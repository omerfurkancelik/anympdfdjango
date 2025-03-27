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

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    """
    PDF'den metin okuyan basit bir fonksiyon örneği.
    PyMuPDF (fitz) veya PyPDF2 kullanılabilir.
    """
    text = []
    try:
        with fitz.open(pdf_path) as doc: #PDF i açar
            for page in doc:
                text.append(page.get_text())
        return "\n".join(text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""
        
        
    with open("denemetext.txt","w",encoding="utf-8") as f: #Bu kısım çalışmıyor
        f.write(text)
    return text

def extract_authors(text): #Yazarları çıkaran fonksiyon (geliştirilecek)
    """
    Metinden potansiyel yazar isimlerini çıkarmaya çalışır.
    1) 'AUTHORS?' veya 'by' gibi kalıpları yakalar.
    2) Bulamazsa spaCy ile PERSON etiketlerini çeker.
    3) Doktora vb. unvanları siler, gereksiz kısımları temizler.
    """
    # Regex ile denenecek kalıplar
    author_section_patterns = [
        r'(?i)AUTHORS?:?\s*(.*?)(?:\n\n|\n\s*ABSTRACT)',
        r'(?i)(?:by|written by)\s+(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
    ]
    
    authors = []
    found_via_regex = False
    for pattern in author_section_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            found_via_regex = True
            author_text = match.group(1)
            # virgül, "and", ";" vb. ayır
            potential_authors = re.split(r',|\band\b|;', author_text)
            authors.extend([a.strip() for a in potential_authors if a.strip()])
    
    # Eğer Regex yoluyla bulamadıysak spaCy PERSON etiketlerini kullanalım
    if not found_via_regex:
        # Çok büyük metinleri kısaltabiliriz (örnek: ilk 5000 karakter)
        doc = nlp(text[:5000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                authors.append(ent.text.strip())
    
    # Unvanları temizle (Dr., Prof. vb.)
    cleaned_authors = []
    for a in authors:
        clean_a = re.sub(r'\b(Dr|Prof|PhD|MD|MSc|BSc|BA|MA)\.?', '', a, flags=re.IGNORECASE)
        # Birden fazla boşluğu teke indir
        clean_a = re.sub(r'\s+', ' ', clean_a).strip()
        # Çok uzun olmayan (örn. en fazla 4 kelime) isimleri al
        if clean_a and len(clean_a.split()) <= 4:
            cleaned_authors.append(clean_a)
    
    # Tekrarlıları temizleyelim
    cleaned_authors = list(set(cleaned_authors))
    logger.info(f"[extract_authors] Found authors: {cleaned_authors}")
    return cleaned_authors

def extract_institutions(text):
    """
    Metinden potansiyel kurum (affiliation) bilgilerini çıkarır.
    1) Regex ile 'affiliation', 'department' vb. kısımlara bakar.
    2) Yoksa spaCy ile ORG etiketlerini çeker.
    3) İçinde 'university', 'institute' vb. geçenleri tercih eder.
    """
    # Denenecek kalıplar (affiliation, institution vs.)
    affiliation_patterns = [
        r'(?i)(affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
        r'(?i)(?:\n)(?:\d\s+)?(affiliation|department|institution)s?:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
    ]
    institutions = []
    
    found_via_regex = False
    for pattern in affiliation_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            found_via_regex = True
            # group(2) varsa oradan alalım, zira group(1) 'affiliation' kelimesinin kendisi olabilir
            affiliation_text = match.group(2) if match.lastindex == 2 else match.group(1)
            potential_insts = re.split(r';|\n|,', affiliation_text)
            institutions.extend([i.strip() for i in potential_insts if i.strip()])
    
    if not found_via_regex:
        # spaCy ile ORG yakalayalım
        doc = nlp(text[:8000])
        # Basit bir tespit kriteri: 'university', 'institute', vs. varsa al
        key_words = ['university','universidade','institute','college','school','department','lab','laboratory',
                     'foundation','company','inc','co.','corp','centre','center']
        for ent in doc.ents:
            if ent.label_ == "ORG":
                # metin küçük harfe dönüştürüp anahtar kelimeler var mı bak
                lower_org = ent.text.lower()
                if any(kw in lower_org for kw in key_words):
                    institutions.append(ent.text.strip())
    
    # Fazla boşlukları temizle
    cleaned_inst = []
    for inst in institutions:
        c = re.sub(r'\s+', ' ', inst).strip()
        if c:
            cleaned_inst.append(c)
    
    cleaned_inst = list(set(cleaned_inst))
    logger.info(f"[extract_institutions] Found institutions: {cleaned_inst}")
    return cleaned_inst

def extract_keywords(text, min_keywords=3, max_keywords=10):
    """
    Metinden (Keywords, Index Terms) bölümlerini veya yoksa
    abstract üzerinden anahtar kelimeleri bulur.
    Burada örneğin spaCy + basit regex yaklaşımı.
    """
    # 1) INDEX TERMS / KEYWORDS bölümünü ara
    keyword_section_patterns = [
        r'(?i)(?:index\s*terms|keywords):?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})',
    ]
    
    explicit_keywords = []
    for pattern in keyword_section_patterns:
        match = re.search(pattern, text, flags=re.DOTALL)
        if match:
            kw_text = match.group(1)
            # virgülle böl
            potential_kws = re.split(r',|;', kw_text)
            explicit_keywords.extend([k.strip() for k in potential_kws if k.strip()])
    
    if explicit_keywords:
        # Bulunan keywordler
        unique_kws = list(set(explicit_keywords))
        return unique_kws[:max_keywords]
    
    # 2) Eğer bulamadıysak, Abstract üzerinden spaCy noun_chunks / named_entities
    abstract_pattern = r'(?i)abstract:?\s*(.*?)(?:\n\n|\n\s*[A-Z]{2,})'
    match_abstract = re.search(abstract_pattern, text, flags=re.DOTALL)
    if match_abstract:
        abstract_text = match_abstract.group(1)
    else:
        # Abstract yoksa metnin ilk ~3000 karakterini kullanıyoruz
        abstract_text = text[:3000]
    
    doc = nlp(abstract_text)
    
    # Noun chunks + ORG, PRODUCT vb. entity
    candidates = []
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:  # fazla uzun olmayan ifadeler
            candidates.append(chunk.text.lower())
    
    for ent in doc.ents:
        if ent.label_ not in ("PERSON", "CARDINAL", "DATE", "TIME"):
            candidates.append(ent.text.lower())
    
    # Bir sıklık hesaplaması
    from collections import Counter
    kw_counter = Counter(candidates)
    most_common = [x for x, _ in kw_counter.most_common(max_keywords*2)]
    
    # min_keywords kadar geri döndür
    final_kws = []
    for mc in most_common:
        if mc not in final_kws:
            final_kws.append(mc)
        if len(final_kws) >= max_keywords:
            break
    
    if len(final_kws) < min_keywords:
        # fallback: en azından min_keywords döndür
        return final_kws + ["keyword"]*(min_keywords - len(final_kws))
    
    logger.info(f"[extract_keywords] Found keywords: {final_kws}")
    return final_kws

def anonymize_pdf(pdf_path, anonymize_map):
    """
    1) Metin redaksiyonu (search_for -> add_redact_annot -> apply_redactions).
    2) Her sayfadaki gömülü resimleri bularak yüzleri bulanıklaştırır.
    3) İşlenmiş PDF'yi geçici bir dosyaya kaydedip onun path'ini döndürür.
    """
    try:
        import fitz
        
        temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
        os.close(temp_fd)
        
        doc = fitz.open(pdf_path)
        
        # -------------------------------------------------
        # A) METİN REDAKSİYONU
        # -------------------------------------------------
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # 1) Metin anonimleştirme (redaction)
            for original, replacement in anonymize_map.items():
                # Tüm eşleşmeleri bul
                text_instances = page.search_for(original)
                for inst in text_instances:
                    annot = page.add_redact_annot(inst, text=replacement, fill=(1,1,1))
                # Redaction uygula
                page.apply_redactions(images=False)  # images=False => resimleri silme
        
        # -------------------------------------------------
        # B) GÖMÜLÜ GÖRSELLERİ BULUP YÜZLERİ BULANIKLAŞTIRMA
        # -------------------------------------------------
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            
            # Bu liste, sayfadaki tüm resimleri (xref vs.) döndürür
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue  # Bu sayfada gömülü resim yok
            
            for img_info in image_list:
                # genelde img_info[0] xref oluyor
                # fitz doc: [xref, smask, width, height, bpc, colorspace, ...]
                xref = img_info[0]
                
                # Resmi bellek olarak çek
                img_dict = safely_extract_image(doc, xref)
                if not img_dict:
                    continue  # extraction başarısız
                
                img_bytes = img_dict["image"]
                
                # Yüz tespiti
                faces = detect_faces(img_bytes)
                if not faces:
                    continue  # Yüz yoksa blur gereksiz
                
                # Blur uygula
                blurred_bytes = blur_faces(img_bytes, faces)
                
                # Şayet blur uygulanmışsa, orijinalle aynı olmayabilir
                if blurred_bytes != img_bytes:
                    # Şimdi page.update_image ile yenisini koyuyoruz
                    try:
                        # Not: update_image() PyMuPDF 1.18+ sürümlerde mevcuttur.
                        page.replace_image(xref, stream=blurred_bytes)
                        print("BLURLANDI")
                    except Exception as e:
                        logger.error(f"Failed to update image xref={xref}: {e}")
        
        # -------------------------------------------------
        # C) Değişiklikleri kaydet
        # -------------------------------------------------
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
            
            
        
        print(faces_list)
        
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
    
    #print(article_keywords)
    
    referee_scores = []
    for referee in referees:
        if not referee.specialization:
            score = 0
        else:
            # Split referee specialization into individual keywords
            referee_keywords = [k.strip().lower() for k in referee.specialization.split(',')]
            
            #print(referee_keywords)
            
            # Count matches between article keywords and referee specialization
            matches = sum(1 for kw in article_keywords if any(kw in ref_kw or ref_kw in kw for ref_kw in referee_keywords))
            
            #print(matches)
            
            # Score is the percentage of article keywords that match referee specialization
            score = matches / len(article_keywords) if article_keywords else 0
        
        referee_scores.append((referee, score))
    
    # Sort by score in descending order
    #print(referee_scores)
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
    
    
    
import fitz

def replace_image_legacy(page, xref, blurred_bytes, bbox):
    """
    Eski PyMuPDF sürümünde page.update_image() yoksa,
    yeni resmi belirlenen 'bbox' bölgesine insert_image ile ekliyoruz.
    'xref'li orijinal resim durabilir, ama görsel olarak kaplamış oluruz.
    """
    # Örneğin, x0,y0,x1,y1 = bbox
    page.insert_image(bbox, stream=blurred_bytes)

def anonymize_pdf_legacy(pdf_path, anonymize_map):
    """
    Eski PyMuPDF sürümleriyle çalışan bir örnek:
    1) Metin redaction
    2) Her sayfadaki resimleri -> detect_faces -> blur -> insert_image
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
    os.close(temp_fd)
    
    doc = fitz.open(pdf_path)
    
    # A) Metin anonimleştirme (redaction)
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for original, replacement in anonymize_map.items():
            text_instances = page.search_for(original)
            for inst in text_instances:
                annot = page.add_redact_annot(inst, text=replacement, fill=(1,1,1))
            page.apply_redactions(images=False)
    
    # B) Resimleri bul -> yüzleri bulanıklaştır -> insert_image
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        if not image_list:
            continue
        
        for img_info in image_list:
            xref = img_info[0]
            # Resmi çıkar
            img_dict = doc.extract_image(xref)
            if not img_dict:
                continue
            
            img_bytes = img_dict["image"]
            faces = detect_faces(img_bytes)
            if not faces:
                continue
            
            blurred_bytes = blur_faces(img_bytes, faces)
            if blurred_bytes != img_bytes:
                # Resmi ekleyeceğimiz dikdörtgeni bulmak gerekiyor
                # Her resmin bounding box'ına erişmek PyMuPDF eski sürümde
                # doğrudan kolay olmayabilir. Tek seçenek: sayfayı rasterize edin 
                # veya tahmini bir rect belirleyin. 
                # Örn. sayfanın tam boyutu:
                rect = page.rect  # Tam sayfa
                # Gerçek bounding box'ı bulacaksanız, 
                #   muhtemelen "img_info"dan width/height alıp 
                #   sayfadaki konumunu hesaplamanız gerek.
                
                replace_image_legacy(page, xref, blurred_bytes, rect)
    
    doc.save(temp_path)
    doc.close()
    return temp_path
