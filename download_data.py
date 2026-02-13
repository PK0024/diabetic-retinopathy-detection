"""
Download and organize Diabetic Retinopathy dataset from Kaggle
Requires: pip install kaggle
"""

import os
import shutil
import zipfile
from pathlib import Path
import subprocess

KAGGLE_COMPETITION = "diabetic-retinopathy-detection"
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
VAL_DIR = DATA_DIR / "val"

def check_kaggle_installed():
    """Check if kaggle package is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        print("Kaggle package not found. Installing...")
        subprocess.run(["pip", "install", "kaggle"], check=True)
        return True

def check_kaggle_credentials():
    """Check if Kaggle API credentials exist"""
    kaggle_dir = Path.home() / ".kaggle"
    cred_file = kaggle_dir / "kaggle.json"
    
    if not cred_file.exists():
        print("\n" + "="*60)
        print("Kaggle API credentials not found!")
        print("="*60)
        print("\nTo download the dataset, you need to:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New Token' - this downloads kaggle.json")
        print("4. Move kaggle.json to ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nAlternatively, you can download manually from:")
        print(f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/data")
        print("="*60 + "\n")
        return False
    return True

def download_dataset():
    """Download dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        
        # Create raw data directory
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download competition files
        print(f"Downloading {KAGGLE_COMPETITION} dataset...")
        api.competition_download_files(KAGGLE_COMPETITION, path=str(RAW_DIR), unzip=False)
        
        print("Download complete!")
        return True
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("\nYou can also download manually from:")
        print(f"https://www.kaggle.com/competitions/{KAGGLE_COMPETITION}/data")
        return False

def extract_and_organize():
    """Extract zip files and organize data"""
    print("\nExtracting and organizing data...")
    
    # Find zip files
    zip_files = list(RAW_DIR.glob("*.zip"))
    
    if not zip_files:
        print("No zip files found in data/raw/")
        print("Please download the dataset first or place zip files in data/raw/")
        return False
    
    # Extract all zip files
    for zip_file in zip_files:
        print(f"Extracting {zip_file.name}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
    
    # Look for train/test images
    # Kaggle DR dataset structure may vary, so we'll look for common patterns
    train_images = list(RAW_DIR.rglob("train*.jpg")) + list(RAW_DIR.rglob("train*.jpeg"))
    test_images = list(RAW_DIR.rglob("test*.jpg")) + list(RAW_DIR.rglob("test*.jpeg"))
    
    if not train_images and not test_images:
        # Try alternative structure - look for any images
        all_images = list(RAW_DIR.rglob("*.jpg")) + list(RAW_DIR.rglob("*.jpeg")) + list(RAW_DIR.rglob("*.png"))
        if all_images:
            print(f"\nFound {len(all_images)} images. Organizing by labels...")
            organize_by_labels(all_images)
            return True
        else:
            print("No images found. Please check the dataset structure.")
            return False
    
    # Organize train images by labels (if labels available)
    if train_images:
        print(f"\nFound {len(train_images)} training images")
        organize_by_labels(train_images, TRAIN_DIR)
    
    # Move test images
    if test_images:
        print(f"\nFound {len(test_images)} test images")
        TEST_DIR.mkdir(parents=True, exist_ok=True)
        for img in test_images[:100]:  # Limit for testing
            shutil.copy2(img, TEST_DIR / img.name)
    
    return True

def organize_by_labels(images, target_dir=None):
    """
    Organize images by DR severity levels (0-4)
    Looks for labels in filenames or separate CSV files
    """
    if target_dir is None:
        target_dir = TRAIN_DIR
    
    # Create class directories
    for i in range(5):
        (target_dir / str(i)).mkdir(parents=True, exist_ok=True)
    
    # Look for trainLabels.csv or similar
    labels_file = RAW_DIR / "trainLabels.csv"
    if not labels_file.exists():
        labels_file = list(RAW_DIR.rglob("*labels*.csv"))[0] if list(RAW_DIR.rglob("*labels*.csv")) else None
    
    if labels_file and labels_file.exists():
        import pandas as pd
        print(f"Reading labels from {labels_file.name}...")
        df = pd.read_csv(labels_file)
        
        # Map images to labels
        image_to_label = {}
        for _, row in df.iterrows():
            image_name = row.iloc[0]  # First column is usually image name
            label = row.iloc[1] if len(row) > 1 else 0  # Second column is label
            image_to_label[image_name] = int(label)
        
        # Copy images to appropriate class folders
        copied = 0
        for img in images:
            img_name = img.stem  # filename without extension
            if img_name in image_to_label:
                label = image_to_label[img_name]
                dest = target_dir / str(label) / img.name
                shutil.copy2(img, dest)
                copied += 1
        
        print(f"Organized {copied} images into class folders")
        
        # Create validation set (20% of training data)
        create_validation_set(target_dir)
        
    else:
        print("No labels file found. Please organize images manually or provide labels CSV.")
        print("Expected structure: data/train/0/, data/train/1/, data/train/2/, data/train/3/, data/train/4/")
        return False
    
    return True

def create_validation_set(train_dir):
    """Create validation set from training data (20% split)"""
    print("\nCreating validation set...")
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    
    import random
    random.seed(42)
    
    for class_dir in range(5):
        class_path = train_dir / str(class_dir)
        if not class_path.exists():
            continue
        
        images = list(class_path.glob("*.jpg")) + list(class_path.glob("*.jpeg")) + list(class_path.glob("*.png"))
        random.shuffle(images)
        
        # Take 20% for validation
        val_count = int(len(images) * 0.2)
        val_images = images[:val_count]
        
        # Create validation class directory
        val_class_dir = VAL_DIR / str(class_dir)
        val_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images to validation
        for img in val_images:
            shutil.move(str(img), str(val_class_dir / img.name))
        
        print(f"Class {class_dir}: {len(val_images)} images moved to validation")

def main():
    """Main function to download and organize dataset"""
    print("="*60)
    print("Diabetic Retinopathy Dataset Setup")
    print("="*60)
    
    # Check prerequisites
    if not check_kaggle_installed():
        return
    
    if not check_kaggle_credentials():
        print("\nPlease set up Kaggle credentials and run again.")
        return
    
    # Download dataset
    if download_dataset():
        # Extract and organize
        if extract_and_organize():
            print("\n" + "="*60)
            print("Dataset setup complete!")
            print("="*60)
            print(f"\nData structure:")
            print(f"  - Training: {TRAIN_DIR}")
            print(f"  - Validation: {VAL_DIR}")
            print(f"  - Test: {TEST_DIR}")
            print("\nYou can now train the model using: python train_model.py")
        else:
            print("\nDataset extraction/organization failed.")
    else:
        print("\nDataset download failed. You can download manually and place files in data/raw/")

if __name__ == "__main__":
    main()
