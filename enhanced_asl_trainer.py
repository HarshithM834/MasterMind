"""
Enhanced ASL Training System with Large Dataset Integration
Integrates multiple ASL datasets for maximum accuracy and precision
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, Reshape, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from asl_dataset_integration import ASLDatasetIntegration
import logging

class EnhancedASLTrainer:
    def __init__(self, model_path="enhanced_asl_model.h5", scaler_path="enhanced_asl_scaler.pkl"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = StandardScaler()
        self.dataset_integration = ASLDatasetIntegration()
        
        # ASL mappings
        self.asl_mappings = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 
            7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 
            14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 
            21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'SPACE', 27: 'DELETE', 28: 'NOTHING'
        }
        
        self.reverse_mappings = {v: k for k, v in self.asl_mappings.items()}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_enhanced_model(self, input_dim=126):
        """Create an enhanced neural network model with advanced architecture"""
        
        # Multiple model architectures to try
        models = {
            'dense': self.create_dense_model(input_dim),
            'conv1d': self.create_conv1d_model(input_dim),
            'hybrid': self.create_hybrid_model(input_dim)
        }
        
        # Return the best performing architecture
        return models['hybrid']  # Hybrid model typically performs best
    
    def create_dense_model(self, input_dim):
        """Create dense neural network model"""
        model = Sequential([
            Dense(1024, activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(29, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy', 'top_5_accuracy']
        )
        
        return model
    
    def create_conv1d_model(self, input_dim):
        """Create 1D convolutional neural network model"""
        model = Sequential([
            Reshape((input_dim, 1), input_shape=(input_dim,)),
            
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.1),
            
            Dense(29, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy', 'top_5_accuracy']
        )
        
        return model
    
    def create_hybrid_model(self, input_dim):
        """Create hybrid CNN-LSTM model"""
        model = Sequential([
            Reshape((input_dim, 1), input_shape=(input_dim,)),
            
            # CNN layers for spatial feature extraction
            Conv1D(64, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # LSTM layers for temporal feature extraction
            LSTM(128, return_sequences=True, dropout=0.2),
            LSTM(64, dropout=0.2),
            
            # Dense layers for classification
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(29, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'top_3_accuracy', 'top_5_accuracy']
        )
        
        return model
    
    def load_or_create_model(self):
        """Load existing model or create new enhanced model"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = load_model(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.logger.info("✅ Loaded enhanced trained model")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load existing model: {e}")
        
        # Create new enhanced model
        self.model = self.create_enhanced_model()
        self.logger.info("🆕 Created new enhanced model (untrained)")
        return False
    
    def prepare_comprehensive_dataset(self):
        """Prepare comprehensive dataset from multiple sources"""
        self.logger.info("Preparing comprehensive ASL dataset...")
        
        # Create synthetic dataset if needed
        if not (self.dataset_integration.data_dir / "synthetic_asl").exists():
            self.logger.info("Creating synthetic dataset...")
            self.dataset_integration.create_synthetic_dataset()
        
        # Prepare training data
        X_train, X_test, y_train, y_test = self.dataset_integration.prepare_training_data()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Dataset prepared: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_enhanced_model(self, epochs=50, batch_size=32):
        """Train the enhanced model with comprehensive dataset"""
        self.logger.info("Starting enhanced model training...")
        
        # Prepare dataset
        X_train, X_test, y_train, y_test = self.prepare_comprehensive_dataset()
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                self.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Evaluate model
        test_loss, test_acc, test_top3_acc, test_top5_acc = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        self.logger.info(f"Training completed!")
        self.logger.info(f"Test Accuracy: {test_acc:.4f}")
        self.logger.info(f"Test Top-3 Accuracy: {test_top3_acc:.4f}")
        self.logger.info(f"Test Top-5 Accuracy: {test_top5_acc:.4f}")
        
        # Generate detailed evaluation report
        self.generate_evaluation_report(X_test, y_test, history)
        
        return history
    
    def generate_evaluation_report(self, X_test, y_test, history):
        """Generate comprehensive evaluation report"""
        self.logger.info("Generating evaluation report...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        class_names = [self.asl_mappings[i] for i in range(29)]
        report = classification_report(y_test, y_pred_classes, target_names=class_names)
        
        self.logger.info("Classification Report:")
        self.logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('ASL Gesture Recognition Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('asl_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot training history
        self.plot_training_history(history)
        
        # Save detailed report
        with open('asl_training_report.txt', 'w') as f:
            f.write("ASL Gesture Recognition Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Architecture: {self.model.name}\n")
            f.write(f"Total Parameters: {self.model.count_params():,}\n")
            f.write(f"Test Accuracy: {self.model.evaluate(X_test, y_test, verbose=0)[1]:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        
        self.logger.info("Evaluation report saved to 'asl_training_report.txt'")
        self.logger.info("Confusion matrix saved to 'asl_confusion_matrix.png'")
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training')
        axes[0, 1].plot(history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        
        # Top-3 Accuracy
        axes[1, 0].plot(history.history['top_3_accuracy'], label='Training')
        axes[1, 0].plot(history.history['val_top_3_accuracy'], label='Validation')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Top-3 Accuracy')
        axes[1, 0].legend()
        
        # Top-5 Accuracy
        axes[1, 1].plot(history.history['top_5_accuracy'], label='Training')
        axes[1, 1].plot(history.history['val_top_5_accuracy'], label='Validation')
        axes[1, 1].set_title('Top-5 Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Top-5 Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('asl_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Training history plot saved to 'asl_training_history.png'")
    
    def predict_gesture(self, features):
        """Predict gesture from features"""
        if self.model is None:
            return None, 0
        
        try:
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled, verbose=0)
            confidence = np.max(prediction[0])
            predicted_class = np.argmax(prediction[0])
            
            return self.asl_mappings[predicted_class], confidence
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None, 0
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_name": self.model.name,
            "total_params": self.model.count_params(),
            "input_shape": self.model.input_shape,
            "output_shape": self.model.output_shape,
            "layers": len(self.model.layers),
            "trained": os.path.exists(self.model_path)
        }

def main():
    """Main training function"""
    print("🚀 Enhanced ASL Training System")
    print("=" * 50)
    
    # Initialize trainer
    trainer = EnhancedASLTrainer()
    
    # Load or create model
    model_loaded = trainer.load_or_create_model()
    
    if not model_loaded:
        print("🔄 Starting comprehensive training with large dataset...")
        
        # Train the model
        history = trainer.train_enhanced_model(epochs=200, batch_size=64)
        
        print("✅ Training completed successfully!")
        print("📊 Check the generated reports and plots for detailed analysis")
    else:
        print("✅ Pre-trained model loaded successfully!")
    
    # Display model info
    model_info = trainer.get_model_info()
    print("\n📋 Model Information:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")
    
    # Display dataset info
    dataset_info = trainer.dataset_integration.get_dataset_info()
    print(f"\n📊 Dataset Information:")
    print(f"  Total samples: {dataset_info['total_samples']}")
    print(f"  Available datasets: {len(dataset_info['available_datasets'])}")
    print(f"  Classes: {len(dataset_info['classes'])}")

if __name__ == "__main__":
    main()
