import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_tamil_model(data_path: str = "data/processed/tamil_clean.csv", models_dir: str = "models"):
    """
    Trains Naive Bayes and Logistic Regression models on preprocessed Tamil news data,
    evaluates classification metrics, and exports vectorizer, label encoder, and model artifacts.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    full_data_path = os.path.join(project_root, data_path)
    full_models_dir = os.path.join(project_root, models_dir)

    print("\n==================================================")
    print("       OPTIMIZED TRAINING FOR: TAMIL")
    print("==================================================")

    if not os.path.exists(full_data_path):
        print(f"Error: Processed Tamil data file not found at '{full_data_path}'")
        print("Please run 'python src/preprocess.py' first.")
        return

    # 1. Load Preprocessed Tamil Data
    df = pd.read_csv(full_data_path).dropna(subset=['clean_text', 'category'])
    print(f"Loaded {len(df)} clean rows across {df['category'].nunique()} classes.")

    # 2. Encode Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['category'])
    classes = label_encoder.classes_

    # 3. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Optimized TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),      # Word unigrams, bigrams, & trigrams
        sublinear_tf=True,       # Suppresses term count frequency noise
        max_features=35000,
        min_df=2
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 5. Model 1: Multinomial Naive Bayes
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

    # 7. Select & Evaluate Winning Model
    if acc_lr >= acc_nb:
        best_name, best_model, best_acc, y_best = 'Logistic Regression', lr, acc_lr, y_pred_lr
    else:
        best_name, best_model, best_acc, y_best = 'Naive Bayes', nb, acc_nb, y_pred_nb

    print(f"Best Model: {best_name.upper()} (Accuracy: {best_acc * 100:.2f}%)\n")
    print(classification_report(y_test, y_best, target_names=classes, zero_division=0))

    # 8. Export Model Artifacts
    os.makedirs(full_models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(full_models_dir, "tamil_model.pkl"))
    joblib.dump(vectorizer, os.path.join(full_models_dir, "tamil_vectorizer.pkl"))
    joblib.dump(label_encoder, os.path.join(full_models_dir, "tamil_label_encoder.pkl"))
    
    print(f"Saved artifacts to '{full_models_dir}/'")
    print("==================================================")

if __name__ == "__main__":
    train_tamil_model()