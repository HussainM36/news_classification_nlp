import os
import re
import pandas as pd

def clean_indic_text(text: str) -> str:
    """
    Cleans Tamil news text by stripping URLs, emails, HTML tags,
    and special characters while preserving Tamil Unicode script (\u0B80-\u0BFF),
    English letters, and numbers.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Strip special punctuation symbols
    text = re.sub(r'[!@#$%^&*()_+\-=\[\]{};:\'",.<>/?\\|`~]', ' ', text)
    
    # 4. Collapse extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataframe(df: pd.DataFrame, text_col: str, label_col: str) -> pd.DataFrame:
    """
    Extracts text and label columns, renames them immediately to 'text' and 'category',
    drops missing values, and applies text cleaning.
    """
    cols = df.columns.tolist()

    # Dynamic column mapping if provided names don't exist exactly
    actual_text = text_col if text_col in cols else cols[0]
    actual_label = label_col if label_col in cols else (cols[1] if len(cols) > 1 else cols[0])

    # Enforce standard column names immediately
    df_temp = pd.DataFrame()
    df_temp['text'] = df[actual_text].astype(str)
    df_temp['category'] = df[actual_label].astype(str).str.strip()

    # Drop empty/missing rows
    df_temp = df_temp.dropna(subset=['text', 'category']).copy()

    # Apply text cleaning
    df_temp['clean_text'] = df_temp['text'].apply(clean_indic_text)

    # Filter out empty or ultra-short text strings after cleaning
    df_temp = df_temp[df_temp['clean_text'].str.len() > 10].copy()

    return df_temp[['text', 'clean_text', 'category']].reset_index(drop=True)

def preprocess_and_save():
    """
    Executes automated preprocessing for the Tamil news dataset
    and exports clean artifacts to data/processed/tamil_clean.csv.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tam_dir = os.path.join(project_root, "data", "tamil")
    processed_dir = os.path.join(project_root, "data", "processed")

    os.makedirs(processed_dir, exist_ok=True)
    tam_file = os.path.join(tam_dir, "tamilmurasu_dataset.csv")

    print("==================================================")
    print(" PREPROCESSING TAMIL NEWS DATASET ")
    print("==================================================")

    if not os.path.exists(tam_file):
        print(f"Error: 'tamilmurasu_dataset.csv' not found at '{tam_file}'.")
        print("Please run 'python src/download_data.py' first.")
        return

    # Load raw dataset
    df_tam_raw = pd.read_csv(tam_file)

    # Apply core cleaning
    df_tam_clean = preprocess_dataframe(df_tam_raw, text_col='news_article', label_col='news_category')
    df_tam_clean = df_tam_clean.drop_duplicates(subset=['clean_text']).reset_index(drop=True)

    # Stratified Downsampling (Max 500 rows per category) using list comprehension to avoid pandas warnings
    df_tam_balanced = pd.concat([
        group.sample(min(len(group), 500), random_state=42)
        for _, group in df_tam_clean.groupby('category')
    ], ignore_index=True)

    tam_out = os.path.join(processed_dir, "tamil_clean.csv")
    df_tam_balanced[['text', 'clean_text', 'category']].to_csv(tam_out, index=False)

    print("Tamil Preprocessing Complete!")
    print(f"   Clean Rows Saved : {len(df_tam_balanced)}")
    print(f"   Total Categories : {df_tam_balanced['category'].nunique()}")
    print(f"   Saved Path       : '{tam_out}'")
    print("==================================================")

if __name__ == "__main__":
    preprocess_and_save()