import os
import glob
import pandas as pd

# Set Kaggle Credentials (if using Kaggle API directly)
os.environ['KAGGLE_USERNAME'] = "hussainm303"
os.environ['KAGGLE_KEY'] = "KGAT_f2c8c9b392cd48f7dac49e37d502f1a7"

def fetch_tamil_dataset():
    """
    Downloads the Tamil Murasu news dataset directly from Kaggle API
    and saves it to data/tamil/.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    tam_dir = os.path.join(project_root, "data", "tamil")
    os.makedirs(tam_dir, exist_ok=True)

    dataset_id = "vijayabhaskar96/tamil-news-classification-dataset-tamilmurasu"
    
    print("==================================================")
    print(" DOWNLOADING TAMIL NEWS DATASET FROM KAGGLE ")
    print("==================================================")

    target_file = os.path.join(tam_dir, "tamilmurasu_dataset.csv")

    if os.path.exists(target_file):
        print(f"Tamil Dataset already exists at: '{target_file}'")
    else:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            print(f"Downloading: {dataset_id}...")
            api.dataset_download_files(dataset_id, path=tam_dir, unzip=True)
            print(f"Successfully downloaded dataset to '{tam_dir}'\n")

        except Exception as e:
            print(f"Kaggle API Download Error: {e}")
            print(f"Manual Setup: Ensure 'tamilmurasu_dataset.csv' is placed inside:\n   {tam_dir}")

    print("==================================================")
    print("Tamil Data Download Complete!")
    print("==================================================")

if __name__ == "__main__":
    fetch_tamil_dataset()