#!/usr/bin/env python3
"""
Enhanced ASL Sign Language Detector Setup Script
Installs dependencies and sets up the enhanced training system
"""

import os
import sys
import subprocess
import platform

def install_requirements():
    """Install required packages"""
    print("🔧 Installing enhanced ASL detector requirements...")
    
    try:
        # Install basic requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_enhanced.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def setup_kaggle_api():
    """Setup Kaggle API for dataset downloads"""
    print("🔑 Setting up Kaggle API...")
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print("📁 Created Kaggle directory")
    
    kaggle_key_path = os.path.join(kaggle_dir, "kaggle.json")
    if not os.path.exists(kaggle_key_path):
        print("⚠️  Kaggle API key not found!")
        print("📋 To download datasets from Kaggle:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Create API token")
        print("   3. Download kaggle.json")
        print("   4. Place it in ~/.kaggle/kaggle.json")
        print("   5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("✅ Kaggle API key found!")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "asl_datasets",
        "asl_datasets/synthetic_asl",
        "models",
        "reports",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"   Created: {directory}")
    
    print("✅ Directories created!")

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        "cv2",
        "mediapipe",
        "numpy",
        "tensorflow",
        "sklearn",
        "joblib",
        "pandas",
        "matplotlib",
        "seaborn"
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All imports successful!")
    return True

def run_dataset_integration_test():
    """Test the dataset integration system"""
    print("🧪 Testing dataset integration...")
    
    try:
        from asl_dataset_integration import ASLDatasetIntegration
        
        # Initialize dataset integration
        dataset_integration = ASLDatasetIntegration()
        
        # Test synthetic dataset creation
        print("   Creating test synthetic dataset...")
        dataset_integration.create_synthetic_dataset()
        
        # Test dataset info
        info = dataset_integration.get_dataset_info()
        print(f"   ✅ Dataset integration working!")
        print(f"   📊 Total samples: {info['total_samples']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dataset integration test failed: {e}")
        return False

def run_enhanced_trainer_test():
    """Test the enhanced trainer"""
    print("🧪 Testing enhanced trainer...")
    
    try:
        from enhanced_asl_trainer import EnhancedASLTrainer
        
        # Initialize trainer
        trainer = EnhancedASLTrainer()
        
        # Test model creation
        print("   Testing model creation...")
        trainer.create_enhanced_model()
        
        # Test model info
        info = trainer.get_model_info()
        print(f"   ✅ Enhanced trainer working!")
        print(f"   📊 Model parameters: {info.get('total_params', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced trainer test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Enhanced ASL Sign Language Detector Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Setup Kaggle API
    setup_kaggle_api()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        sys.exit(1)
    
    # Test dataset integration
    if not run_dataset_integration_test():
        print("⚠️  Dataset integration test failed, but continuing...")
    
    # Test enhanced trainer
    if not run_enhanced_trainer_test():
        print("⚠️  Enhanced trainer test failed, but continuing...")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Run: python advanced_sign_detector.py")
    print("   2. Click 'Enhanced Train' to train with large dataset")
    print("   3. Start detection and show ASL signs!")
    
    print("\n🔗 Available datasets:")
    print("   • Synthetic ASL dataset (automatically created)")
    print("   • Kaggle ASL datasets (requires API key)")
    print("   • YouTube-ASL dataset (requires manual download)")
    
    print("\n📊 Features:")
    print("   • Real-time ASL recognition")
    print("   • Enhanced training with large datasets")
    print("   • Data augmentation and preprocessing")
    print("   • Comprehensive evaluation reports")
    print("   • Debug monitoring and logging")

if __name__ == "__main__":
    main()
