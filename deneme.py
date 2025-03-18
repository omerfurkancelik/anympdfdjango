import re
import spacy

# Eğer metin İngilizceyse ve spaCy İngilizce modeli yoksa:
#   python -m spacy download en_core_web_sm
# Türkçe model kullanacaksanız:
#   python -m spacy download tr_core_news_lg
#   nlp = spacy.load("tr_core_news_lg")
nlp = spacy.load("en_core_web_sm")  # veya "tr_core_news_lg"

def extract_article_info(article_text: str):
    """
    Verilen makale metninden:
     1) Yazar Ad-Soyad bilgileri
     2) Yazar iletişim bilgileri (e-posta)
     3) Yazar kurum bilgileri (affiliation)
     4) Index Terms (Keywords)
    döndürür.
    """
    # 1) E-POSTA ADRESLERİNİ AYIKLAMA
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails_found = re.findall(email_pattern, article_text)
    emails_found = list(set(emails_found))  # Tekrarlı e-postalar varsa temizle
    
    # 2) KEYWORDS (INDEX TERMS) AYIKLAMA
    # Makalede "INDEX TERMS" veya "Keywords" gibi bir ifade geçiyorsa onu arayalım.
    # Örnek metinde "INDEX TERMS Affective computing, CNN, DEAP, ..." şeklinde:
    keywords = []
    keywords_pattern = r'INDEX TERMS(.*?)\n'
    # Bu regex, "INDEX TERMS" ile satır sonu arasındaki metni yakalayacak (non-greedy).
    match_keywords = re.search(keywords_pattern, article_text, re.IGNORECASE | re.DOTALL)
    if match_keywords:
        # Bulduğumuz kısımda virgüllerle veya noktalı virgüllerle ayrılmış kelimeler olabilir
        raw_kw_line = match_keywords.group(1)
        # İçinde varsa iki nokta, tire vb. temizle:
        raw_kw_line = raw_kw_line.replace(':', '')
        raw_kw_line = raw_kw_line.replace(';', ',')
        # Virgüllerle ayırıp strip edelim
        candidate_kw = [k.strip() for k in raw_kw_line.split(',')]
        # Boş olmayanları keywords listesine ekleyelim
        keywords = [k for k in candidate_kw if k]
    
    # 3) spaCy ile NER yaparak PERSON ve ORG etiketlerini toplayalım
    doc = nlp(article_text)
    persons = []
    orgs = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            persons.append(ent.text.strip())
        elif ent.label_ == "ORG":
            orgs.append(ent.text.strip())
    
    # Kişi isimlerini tekrarsız hale getirelim
    persons_unique = list(set(persons))
    # Bazı gereksiz parçalar kalabilir, kısa (<3 karakter) kelimeleri vb. elemek isterseniz:
    persons_cleaned = []
    for p in persons_unique:
        # Örnek filtre: Kişi ismi en az 2 harften oluşsun
        if len(p) > 2:
            persons_cleaned.append(p)
    persons_cleaned.sort()
    
    # Benzer şekilde kurumları da tekrarsız liste halinde tutalım
    orgs_unique = list(set(orgs))
    orgs_unique.sort()
    
    # 4) Yukarıda yakaladığımız ORG içerisinde gerçek "affiliation"ların hangileri olduğunu
    #    bulmak zor olabilir. Basitçe "University", "Institute", "College", "School" vb. içerenleri
    #    "muhtemel kurum" sayabiliriz.
    possible_affiliations = []
    for o in orgs_unique:
        # Örnek basit kontrol:
        if ("univ" in o.lower()) or ("institute" in o.lower()) or ("university" in o.lower()):
            possible_affiliations.append(o)
    
    # Sonuçları bir sözlük halinde dönüyoruz
    return {
        "authors": persons_cleaned,
        "emails": emails_found,
        "affiliations": possible_affiliations,
        "keywords": keywords
    }


# -----------------------------
# ÖRNEK KULLANIM
# -----------------------------
if __name__ == "__main__":
    article_text = """
    Received 20 March 2023, accepted 6 April 2023, date of publication 13 April 2023,
    date of current version 26 April 2023.
    Digital Object Identifier 10.1 109/ACCESS.2023.3266804
    Emotion Recognition Using Temporally Localized Emotional Events in EEG With Naturalistic
    Context: DENS#Dataset
    MOHAMMAD ASIF , (Graduate Student Member, IEEE), SUDHAKAR MISHRA ,
    MAJITHIA TEJAS VINODBHAI, AND UMA SHANKER TIWARY , (Senior Member, IEEE)
    Indian Institute of Information Technology Allahabad, Allahabad, Uttar Pradesh 211012, India
    Corresponding authors: Sudhakar Mishra (rs163@iiita.ac.in), Mohammad Asif (pse2017001@iiita.ac.in),
    and Uma Shanker Tiwary (ust@iiita.ac.in)

    This work was supported by the Ministry of Education, Government of India,
    funded by the acquisition of the EEG system.

    INDEX TERMS Affective computing, CNN, DEAP, DENS, EEG, emotion dataset, emotion recognition,
    LSTM, SEED.
    """
    
    info = extract_article_info(article_text)
    
    print("Yazar Ad-Soyad (Kişi):")
    for i, author_name in enumerate(info["authors"], 1):
        print(f"  {i}) {author_name}")
    
    print("\nYazar İletişim Bilgileri (E-postalar):")
    for i, mail in enumerate(info["emails"], 1):
        print(f"  {i}) {mail}")
    
    print("\nYazar Kurum Bilgileri (affiliations - muhtemel):")
    for i, aff in enumerate(info["affiliations"], 1):
        print(f"  {i}) {aff}")
    
    print("\nMakalenin Keywords (Index Terms):")
    for i, kw in enumerate(info["keywords"], 1):
        print(f"  {i}) {kw}")
