import re
import os
import uuid
import PyPDF2
import nltk
import spacy
from django.conf import settings
from django.core.files.base import ContentFile
import logging

# Initialize logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("Could not load spaCy model. Running spaCy download...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file"""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
    return text

def extract_authors(text):
    """
    Extract potential author names from text
    Uses a combination of regex patterns and NLP to identify names
    """
    # Extract author section with regex patterns
    author_section_patterns = [
        r'(?i)AUTHORS?:?\s*(.*?)(?:\n\n|\n\s*ABSTRACT)',
        r'(?i)(?:by|written by)\s+(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
    ]
    
    authors = []
    for pattern in author_section_patterns:
        matches = re.search(pattern, text)
        if matches:
            author_text = matches.group(1)
            # Split by common separators
            potential_authors = re.split(r',|\band\b|;', author_text)
            authors.extend([a.strip() for a in potential_authors if a.strip()])
    
    # If no authors found by patterns, use spaCy for named entity recognition
    if not authors:
        doc = nlp(text[:5000])  # Process first 5000 chars for efficiency
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                authors.append(ent.text)
    
    # Clean up and deduplicate
    cleaned_authors = []
    for author in authors:
        # Remove titles and degrees
        clean_author = re.sub(r'\b(Dr|Prof|PhD|MD|MSc|BSc|BA|MA)\.?\b', '', author)
        clean_author = re.sub(r'\s+', ' ', clean_author).strip()
        if clean_author and len(clean_author.split()) <= 4:  # Names typically have at most 4 parts
            cleaned_authors.append(clean_author)
    
    return list(set(cleaned_authors))  # Remove duplicates

def extract_institutions(text):
    """
    Extract potential institution names from text
    Uses a combination of regex patterns and NLP
    """
    # Common institution indicators
    institution_indicators = [
        r'(?i)(?:university|college|institute|school|department|laboratory|lab|center|centre)',
        r'(?i)(?:corporation|inc\.|incorporated|llc|company|research|foundation)'
    ]
    
    # Try to find institutional affiliations
    affiliation_patterns = [
        r'(?i)(?:affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
        r'(?i)(?:^|\n)(?:\d\s+)?(?:affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
    ]
    
    institutions = []
    
    # Extract using affiliation patterns
    for pattern in affiliation_patterns:
        matches = re.search(pattern, text)
        if matches:
            affiliation_text = matches.group(1)
            # Split by common separators
            potential_institutions = re.split(r';|\n', affiliation_text)
            institutions.extend([i.strip() for i in potential_institutions if i.strip()])
    
    # If no institutions found by patterns, use spaCy for named entity recognition
    if not institutions:
        doc = nlp(text[:10000])  # Process first 10000 chars for efficiency
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # Check if the organization matches institutional indicators
                if any(re.search(pattern, ent.text) for pattern in institution_indicators):
                    institutions.append(ent.text)
    
    # Clean up and deduplicate
    cleaned_institutions = []
    for institution in institutions:
        clean_institution = re.sub(r'\s+', ' ', institution).strip()
        if clean_institution:
            cleaned_institutions.append(clean_institution)
    
    return list(set(cleaned_institutions))  # Remove duplicates

def extract_keywords(text, min_keywords=3, max_keywords=10):
    """
    Extract potential keywords from text using NLP techniques
    """
    try:
        # First look for an explicit keywords section
        keyword_section_patterns = [
            r'(?i)keywords:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
            r'(?i)key\s*words:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
            r'(?i)index\s*terms:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
        ]
        
        explicit_keywords = []
        for pattern in keyword_section_patterns:
            matches = re.search(pattern, text)
            if matches:
                keyword_text = matches.group(1)
                # Split by common separators
                keywords = re.split(r',|;', keyword_text)
                explicit_keywords.extend([k.strip() for k in keywords if k.strip()])
        
        if explicit_keywords:
            return explicit_keywords[:max_keywords]
        
        # If no explicit keywords, extract from abstract using NLP
        abstract_pattern = r'(?i)abstract:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
        abstract_match = re.search(abstract_pattern, text)
        abstract_text = abstract_match.group(1) if abstract_match else text[:3000]
        
        # Process with spaCy
        doc = nlp(abstract_text)
        
        # Extract noun phrases and named entities
        stopwords = set(nltk.corpus.stopwords.words('english'))
        candidates = []
        
        # Get noun phrases
        for chunk in doc.noun_chunks:
            if chunk.text.lower() not in stopwords and len(chunk.text.split()) <= 4:
                candidates.append(chunk.text.lower())
        
        # Get named entities (except people)
        for ent in doc.ents:
            if ent.label_ not in ["PERSON"] and ent.text.lower() not in stopwords:
                candidates.append(ent.text.lower())
        
        # Count frequencies
        from collections import Counter
        keyword_counter = Counter(candidates)
        
        # Get the most common keywords
        keywords = [kw for kw, _ in keyword_counter.most_common(max_keywords)]
        
        # Ensure we have minimum number of keywords
        if len(keywords) < min_keywords:
            # Add individual nouns as fallback
            nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN" 
                    and token.text.lower() not in stopwords]
            noun_counter = Counter(nouns)
            additional_keywords = [kw for kw, _ in noun_counter.most_common(min_keywords - len(keywords))]
            keywords.extend(additional_keywords)
        
        return list(set(keywords))[:max_keywords]  # Remove duplicates and limit
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {e}")
        return []

def anonymize_pdf(pdf_path, anonymize_map):
    """
    Create an anonymized version of the PDF
    This is a simplified approach using PyPDF2 to replace text
    For production, consider using more robust PDF processing libraries
    """
    try:
        # Read the original PDF
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            writer = PyPDF2.PdfWriter()
            
            # Process each page
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                # Replace each item in the anonymization map
                for original, replacement in anonymize_map.items():
                    # Use regex with word boundaries to avoid partial replacements
                    pattern = r'\b' + re.escape(original) + r'\b'
                    text = re.sub(pattern, replacement, text)
                
                # Create a new page with the modified text
                # This is simplified - in production you'd need a more advanced approach
                # to maintain formatting, images, etc.
                # Here we're just demonstrating the concept
                from reportlab.pdfgen import canvas
                from reportlab.lib.pagesizes import letter
                from io import BytesIO
                
                packet = BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                can.drawString(100, 100, "Anonymized document")
                can.drawString(100, 80, text[:100] + "...")  # Display part of the text
                can.save()
                
                packet.seek(0)
                new_pdf = PyPDF2.PdfReader(packet)
                writer.add_page(new_pdf.pages[0])
            
            # Save the anonymized PDF to a temporary file
            temp_path = os.path.join(os.path.dirname(pdf_path), f"anonymized_{os.path.basename(pdf_path)}")
            with open(temp_path, 'wb') as output_file:
                writer.write(output_file)
                
            return temp_path
    
    except Exception as e:
        logger.error(f"Error anonymizing PDF: {e}")
        return None

def create_anonymization_map(authors, institutions):
    """
    Create a mapping of original text to anonymized replacements
    Returns a dictionary {original: anonymized}
    """
    anon_map = {}
    
    # Anonymize authors
    for idx, author in enumerate(authors):
        anon_map[author] = f"Author-{idx+1}"
    
    # Anonymize institutions
    for idx, institution in enumerate(institutions):
        anon_map[institution] = f"Institution-{idx+1}"
    
    return anon_map

def match_referees_by_keywords(article_keywords, referees):
    """
    Find the most suitable referees based on keyword matching
    Returns a list of referees sorted by relevance
    """
    article_keywords = [k.lower() for k in article_keywords]
    
    referee_scores = []
    for referee in referees:
        if not referee.specialization:
            score = 0
        else:
            # Split referee specialization into individual keywords
            referee_keywords = [k.strip().lower() for k in referee.specialization.split(',')]
            
            # Count matches between article keywords and referee specialization
            matches = sum(1 for kw in article_keywords if any(kw in ref_kw or ref_kw in kw for ref_kw in referee_keywords))
            
            # Score is the percentage of article keywords that match referee specialization
            score = matches / len(article_keywords) if article_keywords else 0
        
        referee_scores.append((referee, score))
    
    # Sort by score in descending order
    return sorted(referee_scores, key=lambda x: x[1], reverse=True)