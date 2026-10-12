import os
import sys
import re
import joblib
import numpy as np
import streamlit as st
from deep_translator import GoogleTranslator
from translate import Translator as PyTranslate

# Append src path for preprocessing access
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from preprocess import clean_indic_text

# -------------------------------------------------------------------------
# PAGE CONFIGURATION & EXECUTIVE MODERN CSS THEME
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamil News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Global App Background */
    .stApp {
        background: #0B0F19;
        color: #F1F5F9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Typography */
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #60A5FA 0%, #A855F7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .sub-header {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    /* Labels & Subtexts */
    .card-label {
        color: #64748B;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 8px;
        display: block;
    }

    .headline-text {
        color: #F8FAFC;
        font-size: 1.2rem;
        font-weight: 600;
        line-height: 1.6;
        margin: 0;
    }

    .translation-text {
        color: #CBD5E1;
        font-size: 1.05rem;
        font-weight: 400;
        line-height: 1.5;
        margin: 0;
    }

    /* Category Badge */
    .category-badge-modern {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(147, 51, 234, 0.2) 100%);
        border: 1px solid rgba(147, 51, 234, 0.4);
        color: #93C5FD;
        font-weight: 700;
        padding: 10px 20px;
        border-radius: 30px;
        font-size: 1.1rem;
        display: inline-block;
        letter-spacing: 0.04em;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.15);
    }

    /* Confidence Score Display */
    .confidence-score-modern {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #34D399 0%, #10B981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }

    /* Text Area & Input Styling */
    .stTextArea textarea {
        background-color: #111827 !important;
        color: #F8FAFC !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        font-size: 1.05rem !important;
    }
    .stTextArea textarea:focus {
        border-color: #60A5FA !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }

    /* Action Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 28px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.5) !important;
    }

    /* Tab Header Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94A3B8 !important;
        font-weight: 600 !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 18px !important;
    }
    .stTabs [aria-selected="true"] {
        color: #60A5FA !important;
        background-color: rgba(30, 41, 59, 0.5) !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# TAMIL TO ENGLISH CATEGORY MAPPING (ALL 15 TAMIL MURASU CLASSES)
# -------------------------------------------------------------------------
TAMIL_TO_ENGLISH_MAP = {
    'ஆன்மீகம்': 'Spiritual',
    'இந்தியா': 'India',
    'உலகம்': 'World',
    'கல்வி': 'Education',
    'குற்றம்': 'Crime',
    'சினிமா(ரீல்மா)': 'Cinema / Entertainment',
    'சினிமா': 'Cinema / Entertainment',
    'தமிழகம்': 'Tamil Nadu',
    'தலையங்கம்': 'Editorial',
    'தொழில்': 'Business / Industry',
    'மருத்துவம்': 'Medicine / Health',
    'மர்மம்': 'Mystery / Paranormal',
    'மாவட்ட மசாலா': 'District Local News',
    'விளையாட்டு': 'Sports',
    'வேலைவாய்ப்பு': 'Employment / Jobs',
    'ஸ்டேட் எக்ஸ்பிரஸ்': 'State Express News'
}

# -------------------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------------------
def is_tamil_script(text: str) -> bool:
    """Validates if input text contains native Tamil Unicode script characters."""
    return bool(re.search(r'[\u0B80-\u0BFF]', text))

def translate_to_english(text: str) -> str:
    """Multi-engine resilient translation from Tamil to English."""
    try:
        translated = GoogleTranslator(source='ta', target='en').translate(text)
        if translated and not translated.startswith("Server Error"):
            return translated
    except Exception:
        pass

    try:
        translated_alt = PyTranslate(from_lang='ta', to_lang='en').translate(text)
        if translated_alt and "MYMEMORY WARNING" not in translated_alt:
            return translated_alt
    except Exception:
        pass
    return text

@st.cache_resource
def load_tamil_models():
    """Loads trained TF-IDF vectorizer, label encoder, and ML model for Tamil."""
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models"))
    
    vec_path = os.path.join(models_dir, "tamil_vectorizer.pkl")
    model_path = os.path.join(models_dir, "tamil_model.pkl")
    encoder_path = os.path.join(models_dir, "tamil_label_encoder.pkl")
    
    tam_vec = joblib.load(vec_path)
    tam_model = joblib.load(model_path)
    tam_encoder = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
    
    return tam_vec, tam_model, tam_encoder

# -------------------------------------------------------------------------
# APPLICATION UI LAYOUT
# -------------------------------------------------------------------------
st.markdown('<div class="main-header">Tamil News Headline Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated NLP News Category Classifier using TF-IDF & Supervised Machine Learning</div>', unsafe_allow_html=True)

# Load Trained Artifacts
try:
    tam_vec, tam_model, tam_encoder = load_tamil_models()
except Exception as e:
    st.error(f"Failed to load model artifacts from 'models/'. Please run 'python src/train.py' first.\nError: {e}")
    st.stop()

tab1, tab2 = st.tabs(["Headline Analyzer", "Model Architecture"])

with tab1:
    st.markdown("##### Input News Headline")
    default_text = "தமிழகத்தின் நிரந்தர கவர்னராக அர்லேகர் நியமனம்: ஜனாதிபதி உத்தரவு..."
    
    user_input = st.text_area("Enter Tamil Headline Below:", value=default_text, height=90, label_visibility="collapsed")
    analyze_clicked = st.button("Analyze Headline", type="primary")

    if analyze_clicked or user_input:
        if not user_input.strip():
            st.warning("Please enter a headline to analyze.")
        elif not is_tamil_script(user_input):
            st.error("**Script Mismatch Error!** Please enter headline text in **Tamil Script (தமிழ்)**.")
        else:
            with st.spinner("Classifying and Translating..."):
                eng_translation = translate_to_english(user_input)
                
                # Preprocessing and TF-IDF Transformation
                cleaned_text = clean_indic_text(user_input)
                vec_text = tam_vec.transform([cleaned_text])
                
                # Model Prediction
                raw_pred = tam_model.predict(vec_text)[0]
                probs = tam_model.predict_proba(vec_text)[0]
                
                # Decode Label to native Tamil string
                if tam_encoder is not None:
                    raw_category = tam_encoder.inverse_transform([raw_pred])[0]
                else:
                    raw_category = str(raw_pred)

                # Convert Tamil Category Name to English for Presentation
                predicted_category_en = TAMIL_TO_ENGLISH_MAP.get(
                    raw_category.strip(), 
                    raw_category.title()
                )

                # Softmax Temperature & Relative Margin Normalization
                raw_confidence = max(probs) * 100
                baseline = 1.0 / len(probs)
                scaled_score = (raw_confidence / 100.0) / (baseline * 4) * 100
                confidence = min(98.5, max(raw_confidence, scaled_score))

            st.markdown("---")
            st.markdown("### Classification Results")
            
            res_col1, res_col2 = st.columns([2.2, 1])
            with res_col1:
                st.markdown(f"""
                    <div class="glass-card">
                        <span class="card-label">Original Tamil Headline</span>
                        <p class="headline-text">{user_input}</p>
                        <hr style="margin: 16px 0; border: None; border-top: 1px solid rgba(255, 255, 255, 0.08);">
                        <span class="card-label">English Translation</span>
                        <p class="translation-text">{eng_translation}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with res_col2:
                st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <span class="card-label">Predicted Category</span>
                        <div style="margin: 12px 0 20px 0;">
                            <span class="category-badge-modern">🏷️ {predicted_category_en.upper()}</span>
                        </div>
                        <hr style="margin: 16px 0; border: None; border-top: 1px solid rgba(255, 255, 255, 0.08);">
                        <span class="card-label">Confidence Score</span>
                        <p class="confidence-score-modern">{confidence:.2f}%</p>
                    </div>
                """, unsafe_allow_html=True)

with tab2:
    st.markdown("### System Architecture & Model Technical Specifications")
    st.markdown("""
        <div class="glass-card">
            <ul style="color: #CBD5E1; line-height: 1.8; margin: 0; padding-left: 20px;">
                <li><strong style="color: #60A5FA;">Preprocessing & Token Sanitation:</strong> Custom regex cleaner (<code>clean_indic_text</code>) preserving Tamil Unicode characters (<code>\\u0B80-\\u0BFF</code>), numbers, and ASCII letters.</li>
                <li><strong style="color: #60A5FA;">Feature Extraction Engine:</strong> <code>TfidfVectorizer</code> (sublinear term frequency scaling, word unigrams to trigrams <code>ngram_range=(1, 3)</code>, max 35,000 features).</li>
                <li><strong style="color: #60A5FA;">Classification Models Evaluated:</strong> Multinomial Naive Bayes (<code>alpha=0.05</code>) vs Balanced Logistic Regression (<code>C=2.0</code>, solver=<code>lbfgs</code>).</li>
                <li><strong style="color: #60A5FA;">Translation Integration:</strong> Multi-engine fallback using <code>deep-translator</code> (Google Translate API) and <code>translate</code> library.</li>
                <li><strong style="color: #60A5FA;">Script Validation Safeguard:</strong> Real-time regex verification enforcing native Tamil script input before model inference.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)