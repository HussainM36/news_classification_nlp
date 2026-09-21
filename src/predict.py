import os
import sys
import re
import joblib
from deep_translator import GoogleTranslator
from translate import Translator as PyTranslate
from transformers import pipeline

# Append src directory path for preprocessing module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_indic_text

# Dynamic Candidate Categories for Gujarati Zero-Shot Classification
GUJARATI_CANDIDATE_LABELS = [
    "Politics", "Sports", "Business", "Entertainment", 
    "Technology", "Health", "Crime", "World", "Lifestyle"
]

# Complete Tamil Murasu 13-Class Mapping
TAMIL_CATEGORY_MAP = {
    0: 'spiritual',        '0': 'spiritual',
    1: 'general news',     '1': 'general news',
    2: 'sports',           '2': 'sports',
    3: 'cinema',           '3': 'cinema',
    4: 'business',         '4': 'business',
    5: 'politics',         '5': 'politics',
    6: 'tamil nadu',       '6': 'tamil nadu',
    7: 'india',            '7': 'india',
    8: 'world',            '8': 'world',
    9: 'regional',         '9': 'regional',
    10: 'lifestyle',       '10': 'lifestyle',
    11: 'politics & public affairs', '11': 'politics & public affairs',
    12: 'technology',      '12': 'technology'
}

def detect_language_script(text: str) -> str:
    """Detects whether the text contains Gujarati or Tamil Unicode characters."""
    has_gujarati = bool(re.search(r'[\u0A80-\u0AFF]', text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
    
    if has_gujarati and not has_tamil:
        return "Gujarati"
    elif has_tamil and not has_gujarati:
        return "Tamil"
    elif has_gujarati and has_tamil:
        return "Mixed"
    return "Unknown"

def translate_to_english(text: str, source_lang: str) -> str:
    """Multi-engine resilient translator."""
    lang_code = 'gu' if source_lang.lower() == 'gujarati' else 'ta'

    # Engine 1: Google Translator
    try:
        translated = GoogleTranslator(source=lang_code, target='en').translate(text)
        if translated and not translated.startswith("Server Error"):
            return translated
    except Exception:
        pass

    # Engine 2: PyTranslate Fallback
    try:
        translated_alt = PyTranslate(from_lang=lang_code, to_lang='en').translate(text)
        if translated_alt and "MYMEMORY WARNING" not in translated_alt:
            return translated_alt
    except Exception:
        pass

    return text

def load_pipelines_and_models():
    """Loads Zero-Shot Transformer pipeline for Gujarati and trained models for Tamil."""
    print("⏳ Loading Zero-Shot Classifier (facebook/bart-large-mnli)...")
    guj_zero_shot = pipeline(
        "zero-shot-classification", 
        model="facebook/bart-large-mnli"
    )

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    try:
        tam_vec = joblib.load(os.path.join(models_dir, "tamil_vectorizer.pkl"))
        tam_model = joblib.load(os.path.join(models_dir, "tamil_model.pkl"))
    except FileNotFoundError:
        print("❌ Error: Tamil model files missing in 'models/'. Please run 'python src/train.py' first.")
        sys.exit(1)

    return guj_zero_shot, (tam_vec, tam_model)

def run_interactive_prediction():
    guj_zero_shot, (tam_vec, tam_model) = load_pipelines_and_models()

    print("\n==================================================")
    print(" 📰 MULTILINGUAL INDIC NEWS CLASSIFIER (CLI) ")
    print("==================================================")

    while True:
        print("\nSelect Language:")
        print("1. Gujarati (ગુજરાતી) [Zero-Shot Transformer - 9 Classes]")
        print("2. Tamil (தமிழ்) [TF-IDF + ML - 13 Classes]")
        print("3. Exit")
        
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == '3':
            print("Exiting prediction CLI. Goodbye!")
            break
        elif choice not in ['1', '2']:
            print("Invalid selection. Please choose 1, 2, or 3.")
            continue

        selected_language = "Gujarati" if choice == '1' else "Tamil"

        headline = input(f"\nPaste/Type {selected_language} News Headline: ").strip()
        if not headline:
            print("Headline cannot be empty!")
            continue

        # Script Validation Check
        detected_script = detect_language_script(headline)
        if detected_script != selected_language:
            print("\n--------------------------------------------------")
            print(f"❌ SCRIPT MISMATCH ERROR!")
            print(f"   You selected : [{selected_language}]")
            print(f"   Detected Text: [{detected_script} Script]")
            print(f"   Please select Option {1 if detected_script == 'Gujarati' else 2} or enter valid {selected_language} text.")
            print("--------------------------------------------------")
            continue

        print("\nAnalyzing & Translating headline...")
        english_headline = translate_to_english(headline, selected_language)

        if selected_language == "Gujarati":
            # Translate headline to English first for optimal zero-shot accuracy
            text_to_classify = english_headline if not english_headline.startswith("[") else headline
            result = guj_zero_shot(text_to_classify, GUJARATI_CANDIDATE_LABELS)
            
            prediction_str = result['labels'][0]
            max_confidence = result['scores'][0] * 100
        else:
            cleaned_text = clean_indic_text(headline)
            vec_text = tam_vec.transform([cleaned_text])
            raw_prediction = tam_model.predict(vec_text)[0]
            probabilities = tam_model.predict_proba(vec_text)[0]
            
            prediction_str = TAMIL_CATEGORY_MAP.get(raw_prediction, str(raw_prediction))
            max_confidence = max(probabilities) * 100

        print("\n==================================================")
        print(f"Language         : [{selected_language}]")
        print(f"Input (Original) : {headline}")
        print(f"Input (English)  : {english_headline}")
        print(f"Category (En)    : 🏷️  {prediction_str.title()}")
        print(f"Confidence       : {max_confidence:.2f}%")
        print("==================================================")

if __name__ == "__main__":
    run_interactive_prediction()