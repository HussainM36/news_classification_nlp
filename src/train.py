import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_language(lang_name: str, data_path: str, models_dir: str = "models"):
    print(f"\n==================================================")
    print(f"       OPTIMIZED TRAINING FOR: {lang_name.upper()}")
    print(f"==================================================")

    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at '{data_path}'")
        return

    # 1. Load data
    df = pd.read_csv(data_path).dropna(subset=['clean_text', 'category'])
    print(f"Loaded {len(df)} clean rows across {df['category'].nunique()} classes.")

    # 2. Encode Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    classes = label_encoder.classes_

    # 3. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Optimized TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),      # Word unigrams, bigrams, & trigrams
        sublinear_tf=True,       # Suppresses word count noise
        max_features=35000,
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Model 1: Naive Bayes
    nb = MultinomialNB(alpha=0.05)
    nb.fit(X_train_tfidf, y_train)
    y_pred_nb = nb.predict(X_test_tfidf)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    print(f"\n---> Naive Bayes Accuracy: {acc_nb * 100:.2f}%")

    # 6. Model 2: Balanced Logistic Regression
    lr = LogisticRegression(max_iter=1000, C=2.0, class_weight='balanced', solver='lbfgs')
    lr.fit(X_train_tfidf, y_train)
    y_pred_lr = lr.predict(X_test_tfidf)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    print(f"---> Logistic Regression Accuracy: {acc_lr * 100:.2f}%\n")

    # 7. Select & Save Best Model
    best_name, best_model, best_acc, y_best = ('Logistic Regression', lr, acc_lr, y_pred_lr) if acc_lr >= acc_nb else ('Naive Bayes', nb, acc_nb, y_pred_nb)

    print(f"🏆 Best Model: {best_name.upper()} (Accuracy: {best_acc * 100:.2f}%)\n")
    print(classification_report(y_test, y_best, target_names=classes, zero_division=0))

    # Export artifacts
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, f"{lang_name}_model.pkl"))
    joblib.dump(vectorizer, os.path.join(models_dir, f"{lang_name}_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(models_dir, f"{lang_name}_label_encoder.pkl"))
    print(f"Saved artifacts to '{models_dir}/'")

if __name__ == "__main__":
    train_language("gujarati", "data/processed/gujarati_clean.csv")
    train_language("tamil", "data/processed/tamil_clean.csv")