"""
Dataset Integration Module for Sign Language Detection
Integrates with Kaggle datasets and pre-trained models
"""

import os
import requests
import zipfile
import json
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class DatasetIntegrator:
    def __init__(self):
        self.datasets_dir = Path("datasets")
        self.models_dir = Path("pretrained_models")
        self.datasets_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Kaggle dataset configurations
        self.kaggle_datasets = {
            "asl_alphabet": {
                "name": "asl-alphabet",
                "description": "ASL Alphabet Dataset",
                "url": "https://www.kaggle.com/datasets/grassknoted/asl-alphabet",
                "local_path": self.datasets_dir / "asl_alphabet",
                "classes": list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                "expected_samples": 87000  # 29 classes * ~3000 samples each
            },
            "sign_language_digits": {
                "name": "sign-language-digits",
                "description": "Sign Language Digits Dataset",
                "url": "https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset",
                "local_path": self.datasets_dir / "sign_digits",
                "classes": list("0123456789"),
                "expected_samples": 2062
            },
            "chalearn_signs": {
                "name": "chalearn-lap-signer-independent-isolated-sign-language-recognition",
                "description": "Chalearn Sign Language Recognition Dataset",
                "url": "https://www.kaggle.com/datasets/jeanmidev/chalearn-lap-signer-independent-isolated-sign-language-recognition",
                "local_path": self.datasets_dir / "chalearn",
                "classes": None,  # Dynamic classes
                "expected_samples": 22000
            }
        }
        
        # Pre-trained model configurations
        self.pretrained_models = {
            "mobilenet_asl": {
                "name": "MobileNetV2 ASL",
                "base_model": MobileNetV2,
                "input_size": (224, 224, 3),
                "description": "MobileNetV2 fine-tuned for ASL recognition"
            },
            "efficientnet_asl": {
                "name": "EfficientNet ASL", 
                "base_model": EfficientNetB0,
                "input_size": (224, 224, 3),
                "description": "EfficientNet fine-tuned for ASL recognition"
            }
        }
        
    def download_kaggle_dataset(self, dataset_name, kaggle_username=None, kaggle_key=None):
        """Download dataset from Kaggle using API"""
        if dataset_name not in self.kaggle_datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_config = self.kaggle_datasets[dataset_name]
        local_path = dataset_config["local_path"]
        
        if local_path.exists():
            print(f"Dataset {dataset_name} already exists at {local_path}")
            return local_path
        
        print(f"Downloading {dataset_config['description']}...")
        
        # Note: This requires Kaggle API setup
        # For now, we'll create a placeholder structure
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Create a download instruction file
        download_instructions = {
            "dataset_name": dataset_name,
            "kaggle_url": dataset_config["url"],
            "local_path": str(local_path),
            "instructions": [
                "1. Install Kaggle API: pip install kaggle",
                "2. Setup Kaggle credentials in ~/.kaggle/kaggle.json",
                f"3. Download dataset: kaggle datasets download -d {dataset_config['name']}",
                f"4. Extract to: {local_path}",
                "5. Run dataset preprocessing"
            ]
        }
        
        with open(local_path / "download_instructions.json", "w") as f:
            json.dump(download_instructions, f, indent=2)
        
        print(f"Download instructions saved to {local_path / 'download_instructions.json'}")
        return local_path
    
    def preprocess_asl_alphabet_dataset(self, dataset_path):
        """Preprocess ASL Alphabet dataset for training"""
        print("Preprocessing ASL Alphabet dataset...")
        
        # Expected structure: dataset_path/A, dataset_path/B, ..., dataset_path/Z
        processed_data = []
        processed_labels = []
        
        for class_name in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            class_path = dataset_path / class_name
            if not class_path.exists():
                print(f"Warning: Class directory {class_path} not found")
                continue
            
            print(f"Processing class: {class_name}")
            image_files = list(class_path.glob("*.jpg")) + list(class_path.glob("*.png"))
            
            for img_file in image_files:
                try:
                    # Load and preprocess image
                    img = cv2.imread(str(img_file))
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to standard size
                    img = cv2.resize(img, (224, 224))
                    
                    # Normalize
                    img = img.astype(np.float32) / 255.0
                    
                    processed_data.append(img)
                    processed_labels.append(ord(class_name) - ord('A'))  # A=0, B=1, etc.
                    
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
                    continue
        
        if processed_data:
            X = np.array(processed_data)
            y = np.array(processed_labels)
            
            # Save processed data
            np.save(dataset_path / "processed_images.npy", X)
            np.save(dataset_path / "processed_labels.npy", y)
            
            print(f"Processed {len(processed_data)} images from ASL Alphabet dataset")
            return X, y
        else:
            print("No data processed from ASL Alphabet dataset")
            return None, None
    
    def create_transfer_learning_model(self, base_model_name, num_classes=29):
        """Create a transfer learning model using pre-trained weights"""
        if base_model_name not in self.pretrained_models:
            raise ValueError(f"Unknown model: {base_model_name}")
        
        model_config = self.pretrained_models[base_model_name]
        
        # Load pre-trained base model
        base_model = model_config["base_model"](
            weights='imagenet',
            include_top=False,
            input_shape=model_config["input_size"]
        )
        
        # Freeze base model layers (optional)
        base_model.trainable = False
        
        # Add custom classification head
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy']
        )
        
        return model
    
    def train_with_kaggle_data(self, dataset_name, model_name, epochs=50):
        """Train model using Kaggle dataset"""
        print(f"Training {model_name} with {dataset_name} dataset...")
        
        # Load processed data
        dataset_path = self.kaggle_datasets[dataset_name]["local_path"]
        
        if not (dataset_path / "processed_images.npy").exists():
            print("Processing dataset first...")
            self.preprocess_asl_alphabet_dataset(dataset_path)
        
        # Load processed data
        X = np.load(dataset_path / "processed_images.npy")
        y = np.load(dataset_path / "processed_labels.npy")
        
        if X is None or len(X) == 0:
            print("No data available for training")
            return None
        
        print(f"Loaded {len(X)} training samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Create model
        model = self.create_transfer_learning_model(model_name, num_classes=26)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
            tf.keras.callbacks.ModelCheckpoint(
                self.models_dir / f"{model_name}_best.h5",
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        test_loss, test_acc, test_top3_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Top-3 Accuracy: {test_top3_acc:.4f}")
        
        # Save final model
        model.save(self.models_dir / f"{model_name}_final.h5")
        
        # Plot training history
        self.plot_training_history(history, model_name)
        
        return model, history
    
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.models_dir / f"{model_name}_training_history.png")
        plt.show()
    
    def download_pretrained_model(self, model_name, source="huggingface"):
        """Download pre-trained models from various sources"""
        if source == "huggingface":
            return self.download_huggingface_model(model_name)
        elif source == "github":
            return self.download_github_model(model_name)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def download_huggingface_model(self, model_name):
        """Download model from Hugging Face Hub"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            model_path = self.models_dir / f"hf_{model_name}"
            model_path.mkdir(exist_ok=True)
            
            print(f"Downloading {model_name} from Hugging Face...")
            
            # Note: This would require specific ASL models on Hugging Face
            # For now, we'll create a placeholder
            placeholder_info = {
                "model_name": model_name,
                "source": "huggingface",
                "status": "placeholder",
                "instructions": [
                    "1. Install transformers: pip install transformers",
                    f"2. Download model: python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{model_name}')\"",
                    f"3. Save to: {model_path}"
                ]
            }
            
            with open(model_path / "download_info.json", "w") as f:
                json.dump(placeholder_info, f, indent=2)
            
            return model_path
            
        except ImportError:
            print("Transformers library not installed. Install with: pip install transformers")
            return None
    
    def create_dataset_downloader_script(self):
        """Create a script to download all datasets"""
        script_content = '''#!/bin/bash
# Sign Language Dataset Downloader Script

echo "🤟 Sign Language Dataset Downloader"
echo "=================================="

# Install required tools
echo "Installing Kaggle API..."
pip install kaggle

echo "Installing additional dependencies..."
pip install transformers datasets

# Create datasets directory
mkdir -p datasets
mkdir -p pretrained_models

echo "📥 Downloading ASL Alphabet Dataset..."
kaggle datasets download -d grassknoted/asl-alphabet -p datasets/asl_alphabet
cd datasets/asl_alphabet
unzip asl-alphabet.zip
cd ../..

echo "📥 Downloading Sign Language Digits Dataset..."
kaggle datasets download -d ardamavi/sign-language-digits-dataset -p datasets/sign_digits
cd datasets/sign_digits
unzip sign-language-digits-dataset.zip
cd ../..

echo "✅ Dataset download complete!"
echo "Run 'python dataset_integration.py' to preprocess the datasets."
'''
        
        with open("download_datasets.sh", "w") as f:
            f.write(script_content)
        
        # Make executable
        os.chmod("download_datasets.sh", 0o755)
        
        print("Created download_datasets.sh script")
        return "download_datasets.sh"
    
    def get_dataset_info(self):
        """Get information about available datasets"""
        info = {
            "kaggle_datasets": self.kaggle_datasets,
            "pretrained_models": self.pretrained_models,
            "download_script": "download_datasets.sh"
        }
        return info

def main():
    """Main function to demonstrate dataset integration"""
    integrator = DatasetIntegrator()
    
    print("🤟 Sign Language Dataset Integration")
    print("====================================")
    
    # Create download script
    integrator.create_dataset_downloader_script()
    
    # Show available datasets
    info = integrator.get_dataset_info()
    print("\n📊 Available Datasets:")
    for name, config in info["kaggle_datasets"].items():
        print(f"  • {name}: {config['description']}")
        print(f"    Classes: {config['classes']}")
        print(f"    Expected samples: {config['expected_samples']}")
    
    print("\n🤖 Available Pre-trained Models:")
    for name, config in info["pretrained_models"].items():
        print(f"  • {name}: {config['description']}")
    
    print("\n📥 To download datasets:")
    print("  1. Run: ./download_datasets.sh")
    print("  2. Or manually download from Kaggle")
    print("  3. Run preprocessing with: python dataset_integration.py")

if __name__ == "__main__":
    main()
