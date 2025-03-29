import re
import PyPDF2
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import base64
import json
import io
import spacy
from spacy.matcher import Matcher
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from decimal import Decimal

class PDFAnonymizer:
    def __init__(self, key=None):
        """
        Initialize the anonymizer with an optional AES key.
        If no key is provided, a random one will be generated.
        """
        if key:
            self.key = key
        else:
            self.key = get_random_bytes(32)  # AES-256
        self.authors = []
        self.institutions = []
        self.metadata = {}
        
        # Load English language model for spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise Exception("spaCy English model not found. Run: python -m spacy download en_core_web_sm")
        
        # Initialize matcher for author patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize spaCy matcher patterns for authors and institutions"""
        # Pattern for author names (e.g., "John Smith", "Smith, J.", etc.)
        author_pattern1 = [{"POS": "PROPN"}, {"POS": "PROPN"}]  # John Smith
        author_pattern2 = [{"POS": "PROPN"}, {"TEXT": ","}, {"POS": "PROPN"}]  # Smith, John
        author_pattern3 = [{"POS": "PROPN"}, {"TEXT": ","}, {"IS_ALPHA": True, "LENGTH": 1}]  # Smith, J.
        
        self.matcher.add("AUTHOR_NAME", [author_pattern1, author_pattern2, author_pattern3])
    
    def extract_metadata(self, pdf_path):
        """
        Extract text from PDF and identify authors/institutions using spaCy NLP.
        """
        self.authors = []
        self.institutions = []
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            
            # Extract text from first few pages (where authors usually are)
            text = ""
            for i, page in enumerate(reader.pages):
                if i >= 3:  # Only check first 3 pages
                    break
                text += page.extract_text()
            
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract authors using matcher and custom rules
            self._extract_authors(doc)
            
            # Extract institutions using NER and custom rules
            self._extract_institutions(doc)
            
            # Store original metadata
            self.metadata = {
                'original_authors': self.authors,
                'original_institutions': self.institutions,
                'original_text': text
            }
            
            return self.authors, self.institutions
    
    def _extract_authors(self, doc):
        """Extract author names using NLP patterns"""
        # Find matches using our matcher patterns
        matches = self.matcher(doc)
        
        # Get unique matches
        seen = set()
        for match_id, start, end in matches:
            span = doc[start:end]
            if span.text not in seen:
                self.authors.append(span.text)
                seen.add(span.text)
        
        # Additional heuristic: look for lines with multiple author names
        for sent in doc.sents:
            if any(ent.text in self.authors for ent in sent.ents if ent.label_ == "PERSON"):
                # Count PROPN tokens in sentence
                propn_count = sum(1 for token in sent if token.pos_ == "PROPN")
                if propn_count > 2:  # Probably multiple authors
                    # Split by common separators
                    parts = re.split(r'[,;&]|\band\b', sent.text)
                    for part in parts:
                        part = part.strip()
                        if part and part not in seen:
                            # Simple validation - at least two words
                            if len(part.split()) >= 2:
                                self.authors.append(part)
                                seen.add(part)
    
    def _extract_institutions(self, doc):
        """Extract institutions using NER and custom rules"""
        # First get ORG entities from spaCy
        org_entities = {ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"}
        
        # Add common academic institution indicators
        institution_keywords = {
            'university', 'college', 'institute', 'institution',
            'school', 'laboratory', 'lab', 'department', 'center', 'centre'
        }
        
        # Find noun phrases that contain institution keywords
        for chunk in doc.noun_chunks:
            lower_text = chunk.text.lower()
            if any(keyword in lower_text for keyword in institution_keywords):
                org_entities.add(chunk.text.strip())
        
        # Filter and clean results
        self.institutions = sorted(
            {org for org in org_entities if len(org) > 5 and not org.isdigit()},
            key=len, reverse=True
        )
    
    def encrypt_data(self, data):
        """Encrypt data using AES in CBC mode"""
        cipher = AES.new(self.key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data.encode('utf-8'), AES.block_size))
        iv = cipher.iv
        return base64.b64encode(iv + ct_bytes).decode('utf-8')
    
    def decrypt_data(self, enc_data):
        """Decrypt data using AES in CBC mode"""
        enc_data = base64.b64decode(enc_data)
        iv = enc_data[:AES.block_size]
        ct = enc_data[AES.block_size:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        return pt.decode('utf-8')
    
    def create_anonymized_pdf(self, original_path, output_path):
        """
        Create an anonymized PDF by redacting author/institution information
        and adding encrypted metadata as a hidden comment.
        """
        # Extract metadata if not already done
        if not self.authors or not self.institutions:
            self.extract_metadata(original_path)
        
        # Encrypt the sensitive information
        encrypted_metadata = self.encrypt_data(json.dumps({
            'authors': self.authors,
            'institutions': self.institutions
        }))
        
        # Create a new PDF with redactions
        writer = PyPDF2.PdfWriter()
        reader = PyPDF2.PdfReader(original_path)
        
        for page_num, page in enumerate(reader.pages):
            if page_num == 0:  # Only redact first page (where authors usually are)
                text = page.extract_text()
                doc = self.nlp(text)
                
                # Create a redaction canvas
                packet = io.BytesIO()
                can = canvas.Canvas(packet, pagesize=letter)
                
                # Redact author and institution mentions
                self._redact_text(can, doc, page)
                
                # Add encrypted metadata (hidden)
                can.setFillColorRGB(1, 1, 1)  # white text
                can.setFont("Helvetica", 1)    # tiny font
                can.drawString(1, 1, f"ANONYMIZED_METADATA:{encrypted_metadata}")
                
                can.save()
                packet.seek(0)
                redaction_page = PyPDF2.PdfReader(packet).pages[0]
                
                # Merge original page with redactions
                page.merge_page(redaction_page)
            
            writer.add_page(page)
        
        # Save the result
        with open(output_path, "wb") as output_file:
            writer.write(output_file)
        
        return output_path
    
    def _redact_text(self, canvas, doc, page):
        """Add redaction rectangles for sensitive text"""
        canvas.setFillColorRGB(1, 1, 1)  # White for redaction
        
        # Get page dimensions
        media_box = page.mediabox
        page_width = float(media_box[2] - media_box[0])
        page_height = float(media_box[3] - media_box[1])
        
        # Simple text positioning (this is approximate - for better accuracy consider PDFMiner)
        line_height = 12
        y_position = page_height - 50  # Start near top
        
        # Split text into lines
        lines = doc.text.split('\n')
        
        for line in lines:
            # Redact authors in this line
            for author in self.authors:
                if author in line:
                    start_pos = line.find(author)
                    end_pos = start_pos + len(author)
                    # Calculate approximate position
                    x1 = 50 + (start_pos * 5)
                    x2 = 50 + (end_pos * 5)
                    canvas.rect(
                        float(x1), float(y_position),
                        float(x2 - x1), float(line_height),
                        fill=1, stroke=0
                    )
            
            # Redact institutions in this line
            for inst in self.institutions:
                if inst in line:
                    start_pos = line.find(inst)
                    end_pos = start_pos + len(inst)
                    # Calculate approximate position
                    x1 = 50 + (start_pos * 5)
                    x2 = 50 + (end_pos * 5)
                    canvas.rect(
                        float(x1), float(y_position),
                        float(x2 - x1), float(line_height),
                        fill=1, stroke=0
                    )
            
            y_position -= line_height
    
    def save_key(self, key_path):
        """Save the AES key to a file"""
        with open(key_path, 'wb') as f:
            f.write(self.key)
    
    @classmethod
    def load_key(cls, key_path):
        """Load an AES key from file"""
        with open(key_path, 'rb') as f:
            key = f.read()
        return cls(key)

if __name__ == "__main__":
    print("PDF Anonymization Tool with NLP")
    print("--------------------------------")
    
    # Initialize the anonymizer
    anonymizer = PDFAnonymizer()
    
    # Get input PDF path
    input_pdf = input("Enter path to PDF file: ").strip()
    if not os.path.exists(input_pdf):
        print("Error: File not found")
        exit(1)
    
    output_pdf = os.path.splitext(input_pdf)[0] + "_anonymized.pdf"
    key_file = "encryption_key.bin"
    
    # Process the PDF
    print("\nExtracting metadata...")
    authors, institutions = anonymizer.extract_metadata(input_pdf)
    
    print("\nFound authors:")
    for author in authors:
        print(f"- {author}")
    
    print("\nFound institutions:")
    for inst in institutions:
        print(f"- {inst}")
    
    print("\nCreating anonymized version...")
    anonymizer.create_anonymized_pdf(input_pdf, output_pdf)
    
    # Save the encryption key
    anonymizer.save_key(key_file)
    
    print(f"\nAnonymized PDF created: {output_pdf}")
    print(f"Encryption key saved: {key_file} (keep this secure!)")