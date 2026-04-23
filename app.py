from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import requests
import json
import os
from dotenv import load_dotenv
from functools import lru_cache
import time
import re

# ── Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

HARDCODED_USER_EMAIL = 'vedantbhagwani@gmail.com'
HARDCODED_USER_NAME = 'Vedant Bhagwani'

# ── Load the EfficientNetV2B3 model
def load_model():
    return tf.keras.models.load_model(
        'model/efficientnet_alzheimer_94.keras'
    )

model = load_model()

# ── Class labels — must match exact folder names from training dataset
class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# ── API Keys from .env
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found. LLM features will use fallback responses.")
    USE_GROQ = False
else:
    USE_GROQ = True
    try:
        import groq
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
    except ImportError:
        print("ERROR: Please install groq: pip install groq")
        USE_GROQ = False

# ────────────────────────────────────────────────
# FREE LLM API (Groq)
# ────────────────────────────────────────────────

@lru_cache(maxsize=50)
def get_diagnosis_details(diagnosis_class, confidence):
    """Get diagnosis details using free Groq API (cached)"""

    if not USE_GROQ:
        return provide_fallback_diagnosis(diagnosis_class, confidence)

    try:
        prompt = f"""Provide detailed medical information about "{diagnosis_class}" in Alzheimer's disease context.
The diagnosis confidence is {confidence}%.

Respond in EXACTLY this format (keep each section to 2-3 sentences):

1. DESCRIPTION: <clinical description>
2. SYMPTOMS: <key symptoms to monitor>
3. PROGRESSION: <expected disease progression>
4. RECOMMENDED_ACTIONS: <medical recommendations>
5. LIFESTYLE_TIPS: <helpful lifestyle modifications>

Be concise, professional, and clinical in tone."""

        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a clinical AI assistant specializing in neurology and Alzheimer's disease."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=600
        )

        return completion.choices[0].message.content

    except Exception as e:
        app.logger.error(f"Groq API error: {e}")
        return provide_fallback_diagnosis(diagnosis_class, confidence)


def get_groq_response(question, context=""):
    """Get chat response using free Groq API"""

    if not USE_GROQ:
        return "AI assistant is currently unavailable. Please consult a healthcare professional."

    try:
        prompt = f"""
You are a clinical AI assistant supporting a neurologist review of Alzheimer's diagnostics.

Context:
{context}

User question: {question}

Respond clearly, professionally, and with medical sensitivity. Keep the answer concise and appropriate for a patient-facing follow-up.
"""

        completion = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful clinical AI assistant specializing in Alzheimer's disease."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )

        return completion.choices[0].message.content

    except Exception as e:
        app.logger.error(f"Groq chat error: {e}")
        return f"AI assistant is temporarily unavailable. Error: {str(e)}"


# ────────────────────────────────────────────────
# OPENSTREETMAP API
# ────────────────────────────────────────────────

def geocode_city_osm(city):
    """Get coordinates for a city using free Nominatim API"""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": f"{city}, India",
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "AlzheimerDiagnosisApp/1.0",
            "Accept-Language": "en"
        }

        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        return None, None

    except Exception as e:
        app.logger.error(f"Geocoding error for {city}: {e}")
        return None, None


def fetch_hospitals_osm_only(city):
    """Fetch hospitals using OpenStreetMap Overpass API"""

    lat, lon = geocode_city_osm(city)
    if lat is None or lon is None:
        app.logger.error(f"Could not geocode city: {city}")
        return []

    app.logger.info(f"Searching for hospitals near {city} at coordinates ({lat}, {lon})")

    try:
        overpass_url = "https://overpass-api.de/api/interpreter"

        query = f"""
        [out:json][timeout:60];
        (
          node["amenity"="hospital"](around:25000,{lat},{lon});
          way["amenity"="hospital"](around:25000,{lat},{lon});
          relation["amenity"="hospital"](around:25000,{lat},{lon});
          node["amenity"="clinic"](around:25000,{lat},{lon});
          way["amenity"="clinic"](around:25000,{lat},{lon});
          node["healthcare"="hospital"](around:25000,{lat},{lon});
          node["healthcare"="clinic"](around:25000,{lat},{lon});
          node["healthcare"="centre"](around:25000,{lat},{lon});
          node["amenity"="doctors"](around:25000,{lat},{lon});
          node["amenity"="medical_centre"](around:25000,{lat},{lon});
        );
        out body center;
        """

        headers = {"User-Agent": "AlzheimerDiagnosisApp/1.0"}
        response = requests.post(overpass_url, data=query, headers=headers, timeout=45)
        response.raise_for_status()
        data = response.json()

        app.logger.info(f"Overpass API response received with {len(data.get('elements', []))} elements")

        results = []
        seen_names = set()

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name", "")

            if not name or name in seen_names:
                continue
            seen_names.add(name)

            facility_type = tags.get("amenity", "")
            if not facility_type:
                facility_type = tags.get("healthcare", "Medical Facility")

            if facility_type == "hospital":
                facility_display = "Hospital"
            elif facility_type == "clinic":
                facility_display = "Clinic"
            elif facility_type == "doctors":
                facility_display = "Doctor's Office"
            else:
                facility_display = facility_type.capitalize()

            phone = tags.get("phone", "")
            if not phone:
                phone = tags.get("contact:phone", "")
            if not phone:
                phone = tags.get("mobile", "") or tags.get("fax", "")

            website = tags.get("website", "")
            if not website:
                website = tags.get("contact:website", "")

            address_parts = [
                tags.get("addr:street", ""),
                tags.get("addr:area", ""),
                tags.get("addr:city", city),
                tags.get("addr:postcode", "")
            ]
            address = ", ".join(p for p in address_parts if p)

            if not address:
                center = element.get("center", {})
                if center:
                    address = f"Near ({center.get('lat', lat):.4f}, {center.get('lon', lon):.4f}), {city}"
                else:
                    address = f"{city}, India"

            results.append({
                "name": name,
                "address": address,
                "phone": phone if phone else "Contact hospital directly",
                "website": website,
                "type": facility_display,
                "source": "openstreetmap"
            })

        app.logger.info(f"Processed {len(results)} unique hospitals/clinics for {city}")
        return results[:15]

    except requests.exceptions.Timeout:
        app.logger.error(f"OpenStreetMap timeout for {city}")
        return []
    except requests.exceptions.RequestException as e:
        app.logger.error(f"OpenStreetMap request error: {e}")
        return []
    except Exception as e:
        app.logger.error(f"OpenStreetMap error: {e}")
        return []


def get_neurologists_by_city(city):
    """Use ONLY OpenStreetMap API - NO hardcoded data"""

    results = fetch_hospitals_osm_only(city)

    if results:
        app.logger.info(f"Successfully fetched {len(results)} hospitals for {city}")
        return results

    return [{
        "name": f"No medical facilities found in OpenStreetMap for {city}",
        "address": "Try Google Maps search or contact your local health department.",
        "phone": "N/A",
        "website": f"https://www.google.com/maps/search/hospitals+in+{city.replace(' ', '+')}",
        "type": "Data Unavailable in OpenStreetMap",
        "source": "osm_no_data"
    }]


# ────────────────────────────────────────────────
# FALLBACK DIAGNOSIS TEXT
# ────────────────────────────────────────────────

def provide_fallback_diagnosis(diagnosis_class, confidence):
    """Simple fallback when LLM API is unavailable"""

    fallback_texts = {
        'Mild Impairment': f"""
**Diagnosis: Mild Impairment** (Confidence: {confidence}%)

Moderate cognitive decline affecting daily functioning. Patient requires support with complex activities.

**Recommendations:**
- Immediate neurological consultation (within 1 month)
- Comprehensive cognitive assessment
- Medication evaluation
- Caregiver support and education
- Safety assessment of living environment

**Note:** This is an automated analysis. Urgent professional medical evaluation is strongly recommended.
""",
        'Moderate Impairment': f"""
**Diagnosis: Moderate Impairment** (Confidence: {confidence}%)

Severe cognitive decline with significant functional impairment. Patient requires substantial assistance with daily activities.

**Recommendations:**
- Emergency neurological consultation (within 1-2 weeks)
- Full care needs assessment
- Consider specialized dementia care
- Medication management review
- Caregiver support services

**Note:** This is an automated analysis. Immediate professional medical evaluation is essential.
""",
        'No Impairment': f"""
**Diagnosis: No Impairment** (Confidence: {confidence}%)

No significant signs of cognitive decline detected. The patient shows normal cognitive function for their age.

**Recommendations:**
- Continue healthy lifestyle and regular exercise
- Maintain balanced diet rich in antioxidants
- Regular annual check-ups with physician
- Engage in mentally stimulating activities

**Note:** This is an automated analysis based on brain imaging. Please consult a neurologist for comprehensive evaluation.
""",
        'Very Mild Impairment': f"""
**Diagnosis: Very Mild Impairment** (Confidence: {confidence}%)

Early stage cognitive decline detected. Patient may experience occasional memory lapses and mild difficulty with complex tasks.

**Recommendations:**
- Schedule consultation with a neurologist within 3 months
- Engage in cognitive stimulation activities daily
- Regular monitoring every 6 months
- Address cardiovascular risk factors (BP, cholesterol, diabetes)

**Note:** This is an automated analysis. Professional medical evaluation is strongly recommended.
"""
    }

    return fallback_texts.get(
        diagnosis_class,
        f"Diagnosis: {diagnosis_class} ({confidence}% confidence). Please consult a neurologist for detailed information."
    )


# ────────────────────────────────────────────────
# IMAGE PREPROCESSING
# ────────────────────────────────────────────────

def preprocess_image(image):
    """Preprocess image for EfficientNetV2B3 prediction"""
    image = image.convert('RGB')
    image = image.resize((300, 300))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)


# ────────────────────────────────────────────────
# GRAD-CAM
# ────────────────────────────────────────────────

def generate_gradcam(image, model, predicted_idx):
    """Generate Grad-CAM heatmap and return as base64 string"""
    import cv2

    # ── Step 1: Find the last Conv2D layer by searching ALL layers recursively
    def find_last_conv_layer(m):
        """Search through all layers including inside nested models"""
        last_conv_name = None
        last_conv_model = None
        for layer in m.layers:
            # If this layer is itself a model (nested), recurse into it
            if hasattr(layer, 'layers'):
                inner_name, inner_model = find_last_conv_layer(layer)
                if inner_name is not None:
                    last_conv_name = inner_name
                    last_conv_model = layer
            elif isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name
                last_conv_model = m
        return last_conv_name, last_conv_model

    last_conv_name, host_model = find_last_conv_layer(model)

    if last_conv_name is None:
        raise ValueError("Could not find any Conv2D layer in the model")

    app.logger.info(f"Grad-CAM using layer: {last_conv_name} from model: {host_model.name}")

    # ── Step 2: Preprocess image
    img_array = np.array(image.convert('RGB').resize((300, 300)), dtype=np.float32)
    img_preprocessed = tf.keras.applications.efficientnet_v2.preprocess_input(img_array.copy())
    img_input = np.expand_dims(img_preprocessed, axis=0)

    # ── Step 3: Build grad model from the host_model that contains the conv layer
    grad_model = tf.keras.models.Model(
        inputs=host_model.input,
        outputs=[
            host_model.get_layer(last_conv_name).output,
            host_model.output
        ]
    )

    # ── Step 4: Compute gradients
    img_tensor = tf.cast(img_input, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, preds = grad_model(img_tensor)
        # If preds has many classes (host_model = full model), use directly
        # If preds is intermediate features, fall back to full model output
        if preds.shape[-1] == len(class_labels):
            loss = preds[:, predicted_idx]
        else:
            # host_model is the base, run full model to get final class loss
            full_preds = model(img_tensor)
            loss = full_preds[:, predicted_idx]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradient tape returned None — model may not be differentiable at this layer")

    # ── Step 5: Pool gradients and build heatmap
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Handle edge case: scalar heatmap
    if heatmap.shape.rank == 0:
        heatmap = tf.expand_dims(tf.expand_dims(heatmap, 0), 0)

    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    # ── Step 6: Overlay heatmap on original image
    img_cv = np.array(image.convert('RGB').resize((300, 300)), dtype=np.uint8)
    img_cv_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    heatmap_resized = cv2.resize(heatmap, (300, 300))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(img_cv_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    # ── Step 7: Convert to base64
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(superimposed_rgb)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    gradcam_str = base64.b64encode(buffer.getvalue()).decode()

    return gradcam_str


# ────────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        if email == HARDCODED_USER_EMAIL:
            session['user_email'] = HARDCODED_USER_EMAIL
            session['user_name'] = HARDCODED_USER_NAME
            return redirect(url_for('upload_file'))
        error = 'Invalid email. Please use the authorized login email.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html', error="No file selected. Please choose an image file.", user_name=session.get('user_name'))

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('upload.html', error="Invalid file type. Please upload a PNG, JPG, or JPEG image.", user_name=session.get('user_name'))

        try:
            file_bytes = file.read()
            image = Image.open(BytesIO(file_bytes))
            image.verify()
            image = Image.open(BytesIO(file_bytes))

            processed = preprocess_image(image)        # ✅ same level as above
            predictions = model.predict(processed)     # ✅
            predicted_idx = int(np.argmax(predictions, axis=1)[0])  # ✅
            
            # ── Confidence score (FIXED: was undefined before)
            confidence = float(predictions[0][predicted_idx] * 100)
            confidence_pct = round(confidence, 2)

            # Generate Grad-CAM
            try:
                gradcam_str = generate_gradcam(image, model, predicted_idx)
            except Exception as e:
                app.logger.error(f"Grad-CAM error: {e}")
                gradcam_str = None

            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            predicted_label = class_labels[predicted_idx]

            diagnosis_details = get_diagnosis_details(predicted_label, confidence_pct)

            city = request.form.get('city', 'Hyderabad')
            hospitals = get_neurologists_by_city(city)

            result = {
                'class':         predicted_label,
                'confidence':    f"{confidence_pct:.2f}",
                'probabilities': {
                    label: f"{predictions[0][i] * 100:.2f}"
                    for i, label in enumerate(class_labels)
                },
                'details':       diagnosis_details,
                'neurologists':  hospitals,
                'city':          city,
                'gradcam':       gradcam_str,
                'api_status': {
                    'llm': 'Groq' if USE_GROQ else 'Fallback',
                    'places': 'OpenStreetMap API (Real-time Data)'
                }
            }
            return render_template('result.html', result=result, img_str=img_str, user_name=session.get('user_name'))

        except Exception as e:
            app.logger.error(f"Upload error: {e}")
            return render_template('upload.html', error=f"Error processing image: {str(e)}", user_name=session.get('user_name'))

    return render_template('upload.html', user_name=session.get('user_name'))


@app.route('/ask', methods=['POST'])
def ask_groq():
    """Chat endpoint using Groq API"""
    if not session.get('user_email'):
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json() or {}
    question = data.get('question', '').strip()
    context = data.get('context', '').strip()

    if not question:
        return jsonify({'error': 'Please type a question.'}), 400

    reply = get_groq_response(question, context)
    return jsonify({'reply': reply})


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'llm_available': USE_GROQ,
        'api_status': {
            'llm': 'Groq' if USE_GROQ else 'Fallback mode',
            'places': 'OpenStreetMap API'
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)