# Alzheimer's Classification Flask App

This Flask app classifies Alzheimer's disease stages from brain scan images using a fine-tuned Inception model.

## Classes
- Non Demented
- Very mild Dementia
- Mild Dementia
- Moderate Dementia

## Installation

1. Create and activate a virtual environment (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the required packages:
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Run the app:
   ```powershell
   python app.py
   ```

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