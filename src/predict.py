import os
import sys
import re
import joblib
from deep_translator import GoogleTranslator
from translate import Translator as PyTranslate

# Append src directory path for preprocessing module access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import clean_indic_text

# Category Mappings
GUJARATI_CATEGORY_MAP = {
    0: 'sports', 1: 'entertainment', 2: 'business',
    '0': 'sports', '1': 'entertainment', '2': 'business',
    '0.0': 'sports', '1.0': 'entertainment', '2.0': 'business',
    'sports': 'sports', 'entertainment': 'entertainment', 'business': 'business'
}

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
    """Detects whether the text contains Gujarati, Tamil, or Other Unicode scripts."""
    has_gujarati = bool(re.search(r'[\u0A80-\u0AFF]', text))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', text))
    
    if has_gujarati and not has_tamil:
        return "Gujarati"
    elif has_tamil and not has_gujarati:
        return "Tamil"
    elif has_gujarati and has_tamil:
        return "Mixed"
    else:
        return "Unknown"

def get_category_label(raw_prediction, language: str) -> str:
    if language.lower() == 'gujarati':
        return GUJARATI_CATEGORY_MAP.get(raw_prediction, str(raw_prediction))
    else:
        return TAMIL_CATEGORY_MAP.get(raw_prediction, str(raw_prediction))

def translate_to_english(text: str, source_lang: str) -> str:
    lang_code = 'gu' if source_lang.lower() == 'gujarati' else 'ta'

    try:
        translated = GoogleTranslator(source=lang_code, target='en').translate(text)
        if translated and not translated.startswith("Server Error"):
            return translated
    except Exception:
        pass

    try:
        translator_alt = PyTranslate(from_lang=lang_code, to_lang='en')
        translated_alt = translator_alt.translate(text)
        if translated_alt and "MYMEMORY WARNING" not in translated_alt:
            return translated_alt
    except Exception:
        pass

    return text

def load_trained_models():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models"))
    
    try:
        guj_vec = joblib.load(os.path.join(models_dir, "gujarati_vectorizer.pkl"))
        guj_model = joblib.load(os.path.join(models_dir, "gujarati_model.pkl"))
        tam_vec = joblib.load(os.path.join(models_dir, "tamil_vectorizer.pkl"))
        tam_model = joblib.load(os.path.join(models_dir, "tamil_model.pkl"))
        return (guj_vec, guj_model), (tam_vec, tam_model)
    except FileNotFoundError:
        print("❌ Error: Model files missing in 'models/'. Please run 'python src/train.py' first.")
        sys.exit(1)

def run_interactive_prediction():
    (guj_vec, guj_model), (tam_vec, tam_model) = load_trained_models()

    print("\n==================================================")
    print(" 📰 MULTILINGUAL INDIC NEWS CLASSIFIER WITH TRANSLATION ")
    print("==================================================")

    while True:
        print("\nSelect Language:")
        print("1. Gujarati (ગુજરાતી)")
        print("2. Tamil (தமிழ்)")
        print("3. Exit")
        
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == '3':
            print("Exiting prediction CLI. Goodbye!")
            break
        elif choice not in ['1', '2']:
            print("Invalid selection. Please choose 1, 2, or 3.")
            continue

        selected_language = "Gujarati" if choice == '1' else "Tamil"
        vec = guj_vec if choice == '1' else tam_vec
        model = guj_model if choice == '1' else tam_model

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

        cleaned_text = clean_indic_text(headline)
        vec_text = vec.transform([cleaned_text])

        raw_prediction = model.predict(vec_text)[0]
        probabilities = model.predict_proba(vec_text)[0]
        max_confidence = max(probabilities) * 100

        prediction_str = get_category_label(raw_prediction, selected_language)
        english_headline = translate_to_english(headline, selected_language)
        english_category = prediction_str.title()

        print("\n==================================================")
        print(f"Language         : [{selected_language}]")
        print(f"Input (Original) : {headline}")
        print(f"Input (English)  : {english_headline}")
        print(f"Category (Native): 🏷️  {prediction_str}")
        print(f"Category (En)    : 🏷️  {english_category}")
        print(f"Confidence       : {max_confidence:.2f}%")
        print("==================================================")

if __name__ == "__main__":
    run_interactive_prediction()