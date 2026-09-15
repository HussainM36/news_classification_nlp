import re
import pandas as pd

def clean_indic_text(text: str) -> str:
    """
    Cleans Indic news text by keeping native Gujarati and Tamil Unicode scripts.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs and emails
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep Unicode ranges for Gujarati (0A80-0AFF) and Tamil (0B80-0BFF)
    pattern = r'[^\s\u0A80-\u0AFF\u0B80-\u0BFF]'
    text = re.sub(pattern, '', text)
    
    # Collapse extra spaces
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
    df_temp['category'] = df[actual_label].astype(str)

    # Drop empty/missing rows
    df_temp = df_temp.dropna(subset=['text', 'category']).copy()

    # Apply text cleaning
    df_temp['clean_text'] = df_temp['text'].apply(clean_indic_text)

    # Filter out empty or ultra-short text strings after cleaning
    df_temp = df_temp[df_temp['clean_text'].str.len() > 10].copy()

    return df_temp[['text', 'clean_text', 'category']].reset_index(drop=True)

if __name__ == "__main__":
    print("Preprocessing module verified.")