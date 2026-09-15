# Multilingual Indic News Classifier (Gujarati & Tamil)

An end-to-end Natural Language Processing (NLP) pipeline designed to classify Gujarati and Tamil news headlines into standard news categories using TF-IDF vectorization, Multinomial Naive Bayes / Logistic Regression classifiers, and real-time English translation support for interactive evaluation and viva demonstrations.

---

## 📌 Features

* **Multilingual Input Support**: Interactive CLI accepting headlines in Gujarati (ગુજરાતી) and Tamil (தமிழ்).
* **Script Validation System**: Built-in Unicode regex validation (`\u0A80-\u0AFF` for Gujarati, `\u0B80-\u0BFF` for Tamil) preventing script cross-contamination and false predictions.
* **Dual-Language Dataset Support**: Integrated Kaggle API dataset fetchers for Gujarati (`disisbig/gujarati-news-dataset`) and Tamil (`vijayabhaskar96/tamil-news-classification-dataset-tamilmurasu`).
* **Multi-Engine Translation Fallback**: Translates Indic input text and predicted categories to English in real-time via `deep-translator` and `translate` without breaking on server rate limits.
* **Robust Preprocessing Pipeline**: Strips punctuation, non-Indic characters, normalizes whitespace, and filters short noise tokens.

---

## 📁 Directory Structure

```text
news_classification_nlp/
│
├── data/
│   ├── gujarati/          # Raw Gujarati dataset CSVs
│   ├── tamil/             # Raw Tamil dataset CSVs
│   └── processed/         # Cleaned and class-balanced datasets
│
├── models/                # Saved TF-IDF vectorizers and trained ML models
│   ├── gujarati_vectorizer.pkl
│   ├── gujarati_model.pkl
│   ├── tamil_vectorizer.pkl
│   └── tamil_model.pkl
│
├── notebooks/             # Exploratory Data Analysis & Preprocessing Notebooks
│   └── 01_eda_and_preprocess.ipynb
│
├── src/                   # Source Code Modules
│   ├── download_data.py   # Kaggle dataset acquisition script
│   ├── preprocess.py      # Indic text cleaning helper functions
│   ├── train.py           # Model training and evaluation script
│   └── predict.py         # Interactive prediction & translation CLI
│
├── .gitignore             # Ignored virtual env, cache, binaries, and CSVs
├── requirements.txt       # Python dependencies
└── README.md              # Project Documentation
🏷️ Category Coverage
Gujarati Categories: Sports, Entertainment, Business

Tamil Categories (Tamil Murasu): Spiritual, General News, Sports, Cinema, Business, Politics, Tamil Nadu, India, World, Regional, Lifestyle, Politics & Public Affairs, Technology

⚙️ Installation & Setup
1. Clone Repository & Setup Environment
Bash
git clone [https://github.com/HussainM36/news_classification_nlp.git](https://github.com/HussainM36/news_classification_nlp.git)
cd news_classification_nlp

python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
2. Install Dependencies
Bash
pip install -r requirements.txt
🚀 Execution Pipeline
Step 1: Download Raw Datasets from Kaggle
Ensure your Kaggle API key environment variables are set inside src/download_data.py:

Bash
python src/download_data.py
Step 2: Preprocess and Clean Data
Run Cell 2 inside notebooks/01_eda_and_preprocess.ipynb to clean Indic text, map categorical labels, apply class balancing, and generate output files in data/processed/.

Step 3: Train Machine Learning Models
Bash
python src/train.py
This script evaluates Multinomial Naive Bayes and Logistic Regression models, selects the best-performing model based on validation F1-score, and exports .pkl artifacts to models/.

Step 4: Run Live Interactive Prediction CLI
Bash
python src/predict.py
📊 Sample Prediction Output
Plaintext
==================================================
 📰 MULTILINGUAL INDIC NEWS CLASSIFIER WITH TRANSLATION 
==================================================

Select Language:
1. Gujarati (ગુજરાતી)
2. Tamil (தமிழ்)
3. Exit
Enter choice (1/2/3): 2

Paste/Type Tamil News Headline: மு.க.ஸ்டாலினுக்கு திடீர் உடல்நலக்குறைவு: திமுக முப்பெரும் விழா ஒத்திவைப்பு

Analyzing & Translating headline...

==================================================
Language         : [Tamil]
Input (Original) : மு.க.ஸ்டாலினுக்கு திடீர் உடல்நலக்குறைவு: திமுக முப்பெரும் விழா ஒத்திவைப்பு
Input (English)  : MK Stalin's sudden illness: DMK postpones grand function
Category (Native): 🏷️  politics & public affairs
Category (En)    : 🏷️  Politics & Public Affairs
Confidence       : 41.71%
==================================================
📜 License
Distributed under the MIT License.