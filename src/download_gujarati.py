import os
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

# Retrieve HF token securely from environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN:
    login(token=HF_TOKEN)

def fetch_real_indic_gujarati_data(output_dir: str = "data/gujarati"):
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "gujarati_expanded_raw.csv")

    print("📥 Connecting to Hugging Face Hub...")

    try:
        dataset = load_dataset(
            "mteb/IndicNLPNewsClassification", 
            "gu", 
            split="train",
            token=HF_TOKEN
        )
        df_raw = pd.DataFrame(dataset)

        LABEL_MAP = {
            '0': 'sports', '1': 'entertainment', '2': 'business',
            0: 'sports', 1: 'entertainment', 2: 'business'
        }

        text_col = 'text' if 'text' in df_raw.columns else df_raw.columns[0]
        label_col = 'label' if 'label' in df_raw.columns else df_raw.columns[1]

        df = pd.DataFrame({
            'headline': df_raw[text_col].astype(str),
            'category': df_raw[label_col].map(LABEL_MAP).fillna(df_raw[label_col].astype(str))
        }).dropna()

        df['headline'] = df['headline'].str.strip()
        df['category'] = df['category'].str.strip().str.lower()
        df = df.drop_duplicates(subset=['headline']).reset_index(drop=True)

        df.to_csv(raw_path, index=False)

        print("\n==================================================")
        print("✅ GUJARATI DATASET DOWNLOAD COMPLETE!")
        print(f"   Total Real Articles Loaded: {len(df)}")
        print(f"   Saved Raw File To          : {raw_path}")
        print("==================================================")

    except Exception as e:
        print(f"\n❌ Download error: {e}")

if __name__ == "__main__":
    fetch_real_indic_gujarati_data()