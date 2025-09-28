#!/usr/bin/env python3
"""
Setup script for Kaggle ASL Fingerspelling Recognition
Configures Kaggle API and downloads the largest ASL dataset
"""

import os
import json
import kaggle
import subprocess
import sys
from pathlib import Path

def setup_kaggle_api():
    """Setup Kaggle API credentials"""
    print("🤟 Setting up Kaggle API for ASL dataset download...")
    
    # Check if Kaggle is installed
    try:
        import kaggle
        print("✅ Kaggle API already installed")
    except ImportError:
        print("📦 Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✅ Kaggle API installed successfully")
    
    # Check for existing credentials
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    
    if credentials_file.exists():
        print("✅ Kaggle credentials already configured")
        return True
    
    print("\n🔑 Kaggle API Setup Required:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New API Token'")
    print("3. Download the kaggle.json file")
    print("4. Place it in ~/.kaggle/kaggle.json")
    print("5. Set permissions: chmod 600 ~/.kaggle/kaggle.json")
    
    # Try to create directory
    kaggle_dir.mkdir(exist_ok=True)
    
    # Check if user has credentials
    response = input("\nDo you have your kaggle.json file ready? (y/n): ").lower()
    if response == 'y':
        print("\nPlease place your kaggle.json file in:")
        print(f"📁 {credentials_file}")
        print("\nThen run this script again.")
        return False
    else:
        print("\n⚠️ Please set up Kaggle API credentials first.")
        return False

def download_asl_dataset():
    """Download the Kaggle ASL Fingerspelling Recognition dataset"""
    print("\n📥 Downloading Kaggle ASL Fingerspelling Recognition dataset...")
    print("📊 Dataset: 3M+ characters from 100+ signers")
    print("🎥 Real-world smartphone recordings")
    
    try:
        # Create dataset directory
        dataset_path = Path("kaggle_asl_dataset")
        dataset_path.mkdir(exist_ok=True)
        
        # Download the dataset
        print("🔄 Downloading dataset files...")
        kaggle.api.dataset_download_files(
            'competition-data/asl-fingerspelling',
            path=str(dataset_path),
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        print(f"📁 Dataset location: {dataset_path.absolute()}")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                print(f"  📄 {file_path.name} ({size_mb:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Manual download instructions:")
        print("1. Visit: https://www.kaggle.com/competitions/asl-fingerspelling")
        print("2. Click 'Download All'")
        print("3. Extract files to: kaggle_asl_dataset/")
        return False

def verify_dataset():
    """Verify the downloaded dataset"""
    print("\n🔍 Verifying dataset...")
    
    dataset_path = Path("kaggle_asl_dataset")
    if not dataset_path.exists():
        print("❌ Dataset directory not found")
        return False
    
    # Check for key files
    expected_files = [
        "train_landmarks",
        "train.csv",
        "character_to_prediction_index.json"
    ]
    
    found_files = []
    for file_path in dataset_path.rglob("*"):
        if file_path.is_file():
            found_files.append(file_path.name)
    
    print(f"📁 Found {len(found_files)} files in dataset")
    
    # Check for important files
    missing_files = []
    for expected_file in expected_files:
        if not any(expected_file in found_file for found_file in found_files):
            missing_files.append(expected_file)
    
    if missing_files:
        print(f"⚠️ Missing expected files: {missing_files}")
        return False
    
    print("✅ Dataset verification completed")
    return True

def create_sample_data():
    """Create sample data for testing without full dataset"""
    print("\n🎯 Creating sample data for testing...")
    
    sample_path = Path("sample_asl_data")
    sample_path.mkdir(exist_ok=True)
    
    # Create sample CSV
    sample_csv = sample_path / "sample_train.csv"
    with open(sample_csv, 'w') as f:
        f.write("file_id,sequence_id,phrase,char\n")
        f.write("123,1,HELLO,H\n")
        f.write("123,1,HELLO,E\n")
        f.write("123,1,HELLO,L\n")
        f.write("123,1,HELLO,L\n")
        f.write("123,1,HELLO,O\n")
    
    # Create sample character mapping
    char_mapping = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
        "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19,
        "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25, "del": 26, "nothing": 27, "space": 28
    }
    
    mapping_file = sample_path / "character_to_prediction_index.json"
    with open(mapping_file, 'w') as f:
        json.dump(char_mapping, f, indent=2)
    
    print("✅ Sample data created")
    print(f"📁 Sample data location: {sample_path.absolute()}")

def main():
    """Main setup function"""
    print("🤟 Kaggle ASL Fingerspelling Recognition Setup")
    print("=" * 50)
    
    # Setup Kaggle API
    if not setup_kaggle_api():
        print("\n⚠️ Please complete Kaggle API setup and run again.")
        return
    
    # Try to download dataset
    print("\n" + "=" * 50)
    if download_asl_dataset():
        verify_dataset()
    else:
        print("\n📝 Creating sample data for testing...")
        create_sample_data()
        print("\n💡 You can manually download the full dataset later")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Run: python3 kaggle_asl_detector.py")
    print("2. Click 'Download Kaggle Dataset' if needed")
    print("3. Click 'Train Model' to train on the dataset")
    print("4. Click 'Start Detection' to begin recognition")

if __name__ == "__main__":
    main()
