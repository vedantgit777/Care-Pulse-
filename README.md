# Alzheimer's Classification Flask App

This Flask app classifies Alzheimer's disease stages from brain scan images using a EfficientNetV2B3 

## Classes
- Non Demented
- Very mild Dementia
- Mild Dementia
- Moderate Dementia

## Installation

cd C:\Users\Vedant\OneDrive\Desktop\CAREPULSE\Care-Pulse-
.\.venv\Scripts\Activate.ps1
python app.py

## Usage

1. Open your browser to `http://127.0.0.1:5000/`.
2. Upload a brain scan image (JPG, JPEG, or PNG).
3. The app will display the prediction, confidence, and probabilities.

## Production Notes

- For production deployment, use a WSGI server like Gunicorn.
- Disable debug mode in `app.run(debug=False)`.
- Add proper error logging and monitoring.
- Ensure secure file uploads and validate inputs.
- The app includes Bootstrap for responsive design and error handling for invalid files.