# Deep Learning Fundus Image Analysis for Early Detection of Diabetic Retinopathy

A deep learning project for automated detection and classification of Diabetic Retinopathy using retinal fundus images.

## Project Overview

Diabetic Retinopathy (DR) is a common complication of diabetes mellitus, which causes lesions on the retina that affect vision. Diabetic Retinopathy is the leading cause of blindness among working aged adults around the world and estimated it may affect more than 93 million people. If it is not detected early, it can lead to blindness.

This project uses Transfer Learning techniques like Inception V3, ResNet50, and Xception V3 to classify retinal fundus images into 5 categories ranging from 0 to 4, where 0 is no Diabetic Retinopathy and 4 is proliferative Diabetic Retinopathy.

## Purpose

The main aim of this project is to aid early detection of Diabetic Retinopathy using retinal fundus images which is useful in providing cost-effective way for early detection of DR in millions of people with diabetes to triage those patients who need further care at a time when they have early rather than advanced DR.

## Features

- **Deep Learning Models**: Inception V3, ResNet50, Xception V3
- **Web Interface**: Flask-based web application for easy image upload and prediction
- **5-Level Classification**: Classifies images into 5 stages (0-4) of Diabetic Retinopathy
- **Transfer Learning**: Uses pre-trained models for better accuracy

## Installation

### Quick Setup (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/PK0024/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
```

2. Run the setup script:
```bash
chmod +x setup.sh
./setup.sh
```

3. Set up Kaggle API credentials:
   - Go to [Kaggle Settings](https://www.kaggle.com/settings)
   - Scroll to "API" section and click "Create New Token"
   - This downloads `kaggle.json`
   - Move it to `~/.kaggle/kaggle.json`
   - Run: `chmod 600 ~/.kaggle/kaggle.json`

4. Download and organize the dataset:
```bash
python download_data.py
```

This will automatically:
   - Download the dataset from [Kaggle Competition](https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data)
   - Extract and organize images into train/test/val folders
   - Split data by DR severity levels (0-4)

### Manual Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install kaggle
```

2. Download dataset manually:
   - Visit: https://www.kaggle.com/competitions/diabetic-retinopathy-detection/data
   - Download the dataset files
   - Extract to `data/raw/` directory
   - Organize images into: `data/train/0/`, `data/train/1/`, etc. (for classes 0-4)

3. Train the models:
```bash
python train_model.py
```

4. Run the Flask application:
```bash
python app.py
```

5. Open your browser and navigate to `http://localhost:5000`

## Project Structure

```
diabetic-retinopathy-detection/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── models/                # Saved model files
├── data/                  # Dataset (train/test/val)
├── static/                # CSS, JS, uploaded images
├── templates/             # HTML templates
├── notebooks/             # Jupyter notebooks for analysis
└── requirements.txt       # Python dependencies
```

## Model Accuracy

Average accuracy: **76.3%**

## Technologies Used

- **Backend**: Flask, Python
- **Deep Learning**: TensorFlow, Keras
- **Models**: Inception V3, ResNet50, Xception V3
- **Frontend**: HTML, CSS, JavaScript
- **Image Processing**: OpenCV, Pillow

## Results

The model classifies retinal fundus images into 5 categories:
- **Level 0**: No Diabetic Retinopathy
- **Level 1**: Mild Non-proliferative DR
- **Level 2**: Moderate Non-proliferative DR
- **Level 3**: Severe Non-proliferative DR
- **Level 4**: Proliferative DR

## Future Scope

- Extend work to better prediction by considering other parameters
- Propose treatments corresponding to the severity of the disease
- Deploy in hospitals for real-time screening
- Improve accuracy through additional training data and model optimization

## License

This project is for educational purposes.

## Contact

For questions or contributions, please open an issue on GitHub.
