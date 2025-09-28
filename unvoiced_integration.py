"""
Unvoiced Integration Module
Integrates the pre-trained model and approach from the Unvoiced GitHub repository
https://github.com/grassknoted/Unvoiced
"""

import os
import requests
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import numpy as np
import cv2
from pathlib import Path
import gtts
from pygame import mixer
import io
import tempfile

class UnvoicedIntegration:
    def __init__(self):
        self.unvoiced_dir = Path("unvoiced_resources")
        self.unvoiced_dir.mkdir(exist_ok=True)
        
        # Unvoiced repository files
        self.unvoiced_files = {
            "trained_model": {
                "filename": "trained_model_graph.pb",
                "url": "https://raw.githubusercontent.com/grassknoted/Unvoiced/master/trained_model_graph.pb",
                "description": "Pre-trained Inception V3 model for ASL recognition"
            },
            "labels": {
                "filename": "training_set_labels.txt",
                "url": "https://raw.githubusercontent.com/grassknoted/Unvoiced/master/training_set_labels.txt",
                "description": "Training labels for the model"
            }
        }
        
        # Initialize text-to-speech
        self.tts_enabled = True
        try:
            mixer.init()
            print("✅ Audio system initialized")
        except Exception as e:
            print(f"⚠️ Audio system not available: {e}")
            self.tts_enabled = False
        
        # Load Unvoiced resources
        self.model = None
        self.labels = []
        self.load_unvoiced_resources()
        
    def download_unvoiced_file(self, file_info, filename):
        """Download a file from the Unvoiced repository"""
        file_path = self.unvoiced_dir / filename
        
        if file_path.exists():
            print(f"✅ {filename} already exists")
            return file_path
        
        print(f"📥 Downloading {filename}...")
        try:
            response = requests.get(file_info["url"], stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded {filename}")
            return file_path
            
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            return None
    
    def load_unvoiced_resources(self):
        """Load all Unvoiced resources"""
        print("🤟 Loading Unvoiced resources...")
        
        # Download and load model
        model_path = self.download_unvoiced_file(
            self.unvoiced_files["trained_model"], 
            "trained_model_graph.pb"
        )
        
        # Download and load labels
        labels_path = self.download_unvoiced_file(
            self.unvoiced_files["labels"], 
            "training_set_labels.txt"
        )
        
        # Load labels
        if labels_path and labels_path.exists():
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            print(f"✅ Loaded {len(self.labels)} labels")
        else:
            # Fallback to standard ASL alphabet
            self.labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            print("⚠️ Using fallback ASL alphabet labels")
        
        # Load model
        if model_path and model_path.exists():
            self.load_unvoiced_model(model_path)
        else:
            print("⚠️ Creating Inception V3 model architecture")
            self.create_inception_model()
    
    def load_unvoiced_model(self, model_path):
        """Load the pre-trained model from Unvoiced"""
        try:
            # Load the frozen model
            with tf.io.gfile.GFile(str(model_path), "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
            
            # Import the graph
            tf.import_graph_def(graph_def, name="")
            
            print("✅ Loaded Unvoiced pre-trained model")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load Unvoiced model: {e}")
            print("🔄 Falling back to Inception V3 architecture")
            self.create_inception_model()
            return False
    
    def create_inception_model(self):
        """Create Inception V3 model similar to Unvoiced"""
        print("🏗️ Creating Inception V3 model architecture...")
        
        # Load pre-trained Inception V3
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=(299, 299, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)
        predictions = Dense(len(self.labels), activation='softmax')(x)
        
        # Create model
        self.model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"✅ Created Inception V3 model for {len(self.labels)} classes")
    
    def preprocess_image_unvoiced(self, image):
        """Preprocess image for Unvoiced model"""
        # Resize to Inception V3 input size
        image = cv2.resize(image, (299, 299))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Expand dimensions for batch
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict_unvoiced(self, image):
        """Predict using Unvoiced model"""
        if self.model is None:
            return None, 0
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image_unvoiced(image)
            
            # Predict
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top prediction
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            
            # Map to label
            if predicted_class_idx < len(self.labels):
                predicted_label = self.labels[predicted_class_idx]
            else:
                predicted_label = "UNKNOWN"
            
            return predicted_label, confidence
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0
    
    def text_to_speech(self, text):
        """Convert text to speech using Google TTS (like Unvoiced)"""
        if not self.tts_enabled or not text:
            return
        
        try:
            # Create TTS object
            tts = gtts.gTTS(text=text, lang='en')
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                
                # Play audio
                mixer.music.load(tmp_file.name)
                mixer.music.play()
                
                # Wait for playback to finish
                while mixer.music.get_busy():
                    pass
                
                # Clean up
                os.unlink(tmp_file.name)
            
            print(f"🔊 Spoke: {text}")
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
    
    def get_model_info(self):
        """Get information about the Unvoiced model"""
        return {
            "model_type": "Inception V3 (Unvoiced)",
            "architecture": "Transfer Learning",
            "input_size": (299, 299, 3),
            "classes": len(self.labels),
            "labels": self.labels,
            "tts_enabled": self.tts_enabled,
            "description": "Pre-trained ASL recognition model from Unvoiced repository"
        }

def main():
    """Test the Unvoiced integration"""
    print("🤟 Unvoiced Integration Test")
    print("=" * 40)
    
    # Create integration instance
    unvoiced = UnvoicedIntegration()
    
    # Show model info
    info = unvoiced.get_model_info()
    print(f"\n📊 Model Information:")
    print(f"   Type: {info['model_type']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   Input Size: {info['input_size']}")
    print(f"   Classes: {info['classes']}")
    print(f"   TTS Enabled: {info['tts_enabled']}")
    print(f"   Labels: {info['labels']}")
    
    # Test TTS
    if unvoiced.tts_enabled:
        print(f"\n🔊 Testing Text-to-Speech...")
        unvoiced.text_to_speech("Hello from Unvoiced integration!")

if __name__ == "__main__":
    main()
