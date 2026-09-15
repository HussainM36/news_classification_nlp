import os
import glob
import pandas as pd

# 1. Set Kaggle Credentials
os.environ['KAGGLE_USERNAME'] = "hussainm303"
os.environ['KAGGLE_KEY'] = "KGAT_f2c8c9b392cd48f7dac49e37d502f1a7"

from kaggle.api.kaggle_api_extended import KaggleApi

def fetch_kaggle_datasets():
    """
    Downloads raw datasets for Gujarati and Tamil directly from Kaggle API.
    """
    api = KaggleApi()
    api.authenticate()

    datasets = {
        "gujarati": {
            "id": "disisbig/gujarati-news-dataset",
            "dir": "data/gujarati"
        },
        "tamil": {
            "id": "vijayabhaskar96/tamil-news-classification-dataset-tamilmurasu",
            "dir": "data/tamil"
        }
    }

    for lang, info in datasets.items():
        print(f"📥 Downloading {lang.capitalize()} dataset: {info['id']}...")
        os.makedirs(info['dir'], exist_ok=True)
        api.dataset_download_files(info['id'], path=info['dir'], unzip=True)
        print(f"✅ Successfully downloaded to '{info['dir']}'\n")

    print("==================================================")
    print("✅ Kaggle Data Download Complete!")
    print("==================================================")

if __name__ == "__main__":
    fetch_kaggle_datasets()