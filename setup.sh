#!/bin/bash

# Setup script for Diabetic Retinopathy Detection Project

echo "=========================================="
echo "Diabetic Retinopathy Detection - Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install Kaggle API (for data download)
echo "Installing Kaggle API..."
pip install kaggle

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{train,test,val,raw}
mkdir -p models
mkdir -p static/uploads
mkdir -p notebooks

# Set permissions
chmod +x download_data.py
chmod +x train_model.py

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set up Kaggle API credentials:"
echo "   - Go to https://www.kaggle.com/settings"
echo "   - Create API token and save to ~/.kaggle/kaggle.json"
echo "   - Run: chmod 600 ~/.kaggle/kaggle.json"
echo ""
echo "2. Download dataset:"
echo "   python download_data.py"
echo ""
echo "3. Train the model:"
echo "   python train_model.py"
echo ""
echo "4. Run the Flask app:"
echo "   python app.py"
echo ""
