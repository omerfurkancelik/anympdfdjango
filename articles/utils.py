import re
import os
import uuid
import PyPDF2
import nltk
import spacy
from django.conf import settings
from django.core.files.base import ContentFile
import logging
import fitz  # PyMuPDF
import io
from PIL import Image, ImageDraw, ImageFilter
import tempfile
import numpy as np
import cv2

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
        doc = nlp()  # Process first 5000 chars for efficiency
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
            
            
    print("Temiz Olmayan Yazarlar : " + str(authors))
    
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
        doc = nlp(text)  # Process first 10000 chars for efficiency
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
            
            
    print("Temiz Olmayan Endüstriler : " + str(institutions))
    
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
    Simplified approach that focuses on robust text anonymization
    and adds white rectangles over detected faces
    """
    try:
        # Create a temporary file for the anonymized PDF
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        # Process each page
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # 1. Text anonymization - this is the most reliable part
            for original, replacement in anonymize_map.items():
                # Search for text instances and replace them
                text_instances = page.search_for(original)
                for inst in text_instances:
                    # Add redaction annotation
                    annot = page.add_redact_annot(inst, text=replacement)
                    # Apply redactions
                    page.apply_redactions()
            
            # 2. Simple face detection and covering with white rectangles
            try:
                # Get page pixmap (rasterize the page)
                pix = page.get_pixmap()
                img_data = pix.tobytes()
                
                # Convert to PIL image
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
                
                # Detect faces
                img_bytes = io.BytesIO()
                pil_img.save(img_bytes, format="PNG")
                img_bytes.seek(0)
                faces = detect_faces(img_bytes.getvalue())
                
                # If faces detected, add white rectangles over them
                if faces and len(faces):
                    logger.info(f"Found {len(faces)} faces on page {page_idx+1}")
                    
                    for (x, y, w, h) in faces:
                        # Scale face coordinates to match the page dimensions
                        scale_x = page.rect.width / pix.width
                        scale_y = page.rect.height / pix.height
                        
                        # Create rectangle with some padding
                        padding = 5
                        face_rect = fitz.Rect(
                            x * scale_x - padding,
                            y * scale_y - padding,
                            (x + w) * scale_x + padding,
                            (y + h) * scale_y + padding
                        )
                        
                        # Add white rectangle with high opacity
                        page.draw_rect(face_rect, color=(1, 1, 1), fill=(1, 1, 1), opacity=0.9)
            
            except Exception as e:
                logger.error(f"Error processing faces on page {page_idx+1}: {e}")
                # Continue to next page even if this page fails
        
        # Save the anonymized PDF
        doc.save(temp_path)
        doc.close()
        
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




def detect_faces(image_bytes):
    """
    Detect faces in an image using OpenCV
    
    Args:
        image_bytes: Binary image data
        
    Returns:
        List of face rectangles (x, y, width, height)
    """
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Check if image was loaded properly
        if img is None or img.size == 0:
            logger.warning("Failed to decode image")
            return []
        
        # Load the pre-trained face detector model
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        # Verify the cascade loaded correctly
        if face_cascade.empty():
            logger.error("Failed to load face cascade classifier")
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Handle the case where no faces are found
        if isinstance(faces, tuple) and len(faces) == 0:
            return []
        
        # Convert faces to a list if it's a numpy array
        # This avoids the "truth value of an array is ambiguous" error
        faces_list = []
        for (x, y, w, h) in faces:
            faces_list.append((int(x), int(y), int(w), int(h)))
        
        return faces_list
    except Exception as e:
        logger.error(f"Error detecting faces: {e}")
        return []

# Safe image extraction from PDF function
def safely_extract_image(doc, xref):
    """
    Safely extract an image from a PDF document
    
    Args:
        doc: PyMuPDF document
        xref: Image reference
        
    Returns:
        Dictionary with image data or None if extraction failed
    """
    try:
        return doc.extract_image(xref)
    except Exception as e:
        logger.error(f"Failed to extract image (xref {xref}): {e}")
        return None
    
    

def blur_faces(image_bytes, faces):
    """
    Blur detected faces in an image
    
    Args:
        image_bytes: Binary image data
        faces: List of face rectangles from detect_faces
        
    Returns:
        Binary data of image with blurred faces
    """
    try:
        # Convert to PIL image for processing
        img = Image.open(io.BytesIO(image_bytes))
        
        # If we found faces, blur each one
        if len(faces) > 0:
            # Convert to numpy for OpenCV processing
            img_np = np.array(img)
            
            # Convert RGB to BGR for OpenCV if needed
            # Check if the array has 3 dimensions and the last dimension is 3 (RGB)
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # For each face, apply a heavy blur
            for (x, y, w, h) in faces:
                # Make sure coordinates are within image bounds
                x, y = max(0, x), max(0, y)
                w = min(w, img_np.shape[1] - x)
                h = min(h, img_np.shape[0] - y)
                
                if w <= 0 or h <= 0:
                    continue  # Skip if dimensions are invalid
                
                # Get the face region
                face_region = img_np[y:y+h, x:x+w]
                
                # Apply a strong blur
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                
                # Replace the region with the blurred version
                img_np[y:y+h, x:x+w] = blurred_face
            
            # Convert back to RGB for PIL if needed
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL Image
            img = Image.fromarray(img_np)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        img_format = img.format or "PNG"
        
        # Some formats like JPEG2000 might not be supported by PIL's save
        # Fall back to PNG in those cases
        try:
            img.save(output_buffer, format=img_format)
        except (KeyError, IOError):
            img.save(output_buffer, format="PNG")
            
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
    except Exception as e:
        logger.error(f"Error blurring faces: {e}")
        return image_bytes  # Return original if processing fails
    
    
    
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



def safe_operation(operation_func, error_message, default_return=None, *args, **kwargs):
    """
    Wrapper function to execute operations safely within a try-except block
    
    Args:
        operation_func: Function to execute
        error_message: Message to log if an error occurs
        default_return: Value to return if the operation fails
        *args, **kwargs: Arguments to pass to operation_func
        
    Returns:
        Result of operation_func or default_return if an error occurs
    """
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {e}")
        return default_return