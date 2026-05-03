from flask import Flask, request, render_template, jsonify, session, redirect, url_for
import tensorflow as tf
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import requests
import os
from dotenv import load_dotenv
from functools import lru_cache
from pymongo import MongoClient
from datetime import datetime
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppresses INFO and WARNING logs
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-secret-key')

# ── MongoDB
MONGO_URI = os.getenv('MONGO_URI')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client['carepulse']
users_collection = db['users']

# ── Admin credentials from env
ADMIN_EMAIL = os.getenv('EMAIL_USER')
ADMIN_PASS  = os.getenv('EMAIL_PASS')

# ── Model
def load_model():
    return tf.keras.models.load_model('model/efficientnet_alzheimer_94.keras')

model = load_model()

# ── Class labels
class_labels = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

# ── Groq
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
USE_GROQ = False
groq_client = None

if GROQ_API_KEY:
    try:
        import groq
        groq_client = groq.Groq(api_key=GROQ_API_KEY)
        USE_GROQ = True
        print("Groq AI ready")
    except ImportError:
        print("ERROR: pip install groq")
else:
    print("WARNING: GROQ_API_KEY not found — using fallback responses")


# ────────────────────────────────────────────────
# GROQ FUNCTIONS
# ────────────────────────────────────────────────

@lru_cache(maxsize=50)
def get_diagnosis_details(diagnosis_class, confidence):
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
            model="llama-3.3-70b-versatile",
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
    if not USE_GROQ:
        return "AI assistant is currently unavailable. Please consult a healthcare professional."
    try:
        prompt = f"""You are a clinical AI assistant supporting a neurologist review of Alzheimer's diagnostics.

Context:
{context}

User question: {question}

Respond clearly, professionally, and with medical sensitivity. Keep the answer concise."""

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
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
        return f"AI assistant temporarily unavailable. Error: {str(e)}"


# ────────────────────────────────────────────────
# OPENSTREETMAP
# ────────────────────────────────────────────────

def geocode_city_osm(city):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"{city}, India", "format": "json", "limit": 1}
        headers = {"User-Agent": "AlzheimerDiagnosisApp/1.0", "Accept-Language": "en"}
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
    lat, lon = geocode_city_osm(city)
    if lat is None:
        return []
    try:
        overpass_url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:60];
        (
          node["amenity"="hospital"](around:25000,{lat},{lon});
          way["amenity"="hospital"](around:25000,{lat},{lon});
          node["amenity"="clinic"](around:25000,{lat},{lon});
          node["healthcare"="hospital"](around:25000,{lat},{lon});
          node["healthcare"="centre"](around:25000,{lat},{lon});
        );
        out body center;
        """
        headers = {"User-Agent": "AlzheimerDiagnosisApp/1.0"}
        response = requests.post(overpass_url, data=query, headers=headers, timeout=45)
        response.raise_for_status()
        data = response.json()

        results = []
        seen_names = set()

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name", "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)

            facility_type = tags.get("amenity", tags.get("healthcare", "Medical Facility"))
            if facility_type == "hospital":
                facility_display = "Hospital"
            elif facility_type == "clinic":
                facility_display = "Clinic"
            else:
                facility_display = facility_type.capitalize()

            phone = tags.get("phone", "") or tags.get("contact:phone", "") or tags.get("mobile", "")
            website = tags.get("website", "") or tags.get("contact:website", "")

            address_parts = [
                tags.get("addr:street", ""),
                tags.get("addr:area", ""),
                tags.get("addr:city", city),
                tags.get("addr:postcode", "")
            ]
            address = ", ".join(p for p in address_parts if p)
            if not address:
                center = element.get("center", {})
                address = f"Near ({center.get('lat', lat):.4f}, {center.get('lon', lon):.4f}), {city}" if center else f"{city}, India"

            results.append({
                "name": name,
                "address": address,
                "phone": phone if phone else "Contact hospital directly",
                "website": website,
                "type": facility_display,
            })

        return results[:15]

    except Exception as e:
        app.logger.error(f"OpenStreetMap error: {e}")
        return []


def get_neurologists_by_city(city):
    results = fetch_hospitals_osm_only(city)
    if results:
        return results
    return [{
        "name": f"No facilities found in OpenStreetMap for {city}",
        "address": "Try Google Maps or contact your local health department.",
        "phone": "N/A",
        "website": f"https://www.google.com/maps/search/hospitals+in+{city.replace(' ', '+')}",
        "type": "Data Unavailable",
    }]


# ────────────────────────────────────────────────
# FALLBACK DIAGNOSIS
# ────────────────────────────────────────────────

def provide_fallback_diagnosis(diagnosis_class, confidence):
    fallback_texts = {
        'Mild Impairment': f"""Diagnosis: Mild Impairment (Confidence: {confidence}%)

Moderate cognitive decline affecting daily functioning. Patient requires support with complex activities.

Recommendations:
- Neurological consultation within 1 month
- Comprehensive cognitive assessment
- Medication evaluation
- Caregiver support and education

Note: This is an automated analysis. Professional medical evaluation is strongly recommended.""",

        'Moderate Impairment': f"""Diagnosis: Moderate Impairment (Confidence: {confidence}%)

Severe cognitive decline with significant functional impairment. Patient requires substantial assistance with daily activities.

Recommendations:
- Neurological consultation within 1-2 weeks
- Full care needs assessment
- Consider specialized dementia care
- Medication management review

Note: This is an automated analysis. Immediate professional medical evaluation is essential.""",

        'No Impairment': f"""Diagnosis: No Impairment (Confidence: {confidence}%)

No significant signs of cognitive decline detected. Normal cognitive function for age.

Recommendations:
- Continue healthy lifestyle and regular exercise
- Maintain balanced diet rich in antioxidants
- Regular annual check-ups
- Engage in mentally stimulating activities

Note: This is an automated analysis. Please consult a neurologist for comprehensive evaluation.""",

        'Very Mild Impairment': f"""Diagnosis: Very Mild Impairment (Confidence: {confidence}%)

Early stage cognitive decline detected. Patient may experience occasional memory lapses.

Recommendations:
- Neurologist consultation within 3 months
- Cognitive stimulation activities daily
- Regular monitoring every 6 months
- Address cardiovascular risk factors

Note: This is an automated analysis. Professional medical evaluation is strongly recommended."""
    }
    return fallback_texts.get(
        diagnosis_class,
        f"Diagnosis: {diagnosis_class} ({confidence}% confidence). Please consult a neurologist."
    )


# ────────────────────────────────────────────────
# PREPROCESSING
# ────────────────────────────────────────────────

def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((300, 300))
    image = np.array(image, dtype=np.float32)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)


# ────────────────────────────────────────────────
# ROUTES — AUTH
# ────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        # Look up user in MongoDB
        user = users_collection.find_one({'email': email})
        if user:
            session['user_email'] = email
            session['user_name']  = user.get('name', email)
            session['role']       = 'user'
            # Record login timestamp
            users_collection.update_one(
                {'email': email},
                {'$set': {'last_login': datetime.utcnow()},
                 '$inc': {'login_count': 1}}
            )
            return redirect(url_for('upload_file'))
        else:
            error = 'Invalid email. Please use an authorized login email.'
    return render_template('login.html', error=error)


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '').strip()

        if email == ADMIN_EMAIL.lower() and password == ADMIN_PASS:
            session['user_email'] = email
            session['user_name']  = 'Admin'
            session['role']       = 'admin'
            return redirect(url_for('admin_dashboard'))
        else:
            error = 'Invalid admin credentials.'
    return render_template('admin_login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ────────────────────────────────────────────────
# ROUTES — ADMIN
# ────────────────────────────────────────────────

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get('role') != 'admin':
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


@app.route('/admin')
@admin_required
def admin_dashboard():
    users = list(users_collection.find({}, {'_id': 0}))
    total_users  = len(users)
    total_logins = sum(u.get('login_count', 0) for u in users)
    return render_template('admin.html',
                           users=users,
                           total_users=total_users,
                           total_logins=total_logins,
                           admin_name=session.get('user_name'))


@app.route('/admin/add_user', methods=['POST'])
@admin_required
def admin_add_user():
    data  = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    name  = data.get('name', '').strip()
    if not email or not name:
        return jsonify({'error': 'Email and name are required.'}), 400
    if users_collection.find_one({'email': email}):
        return jsonify({'error': 'User already exists.'}), 409
    users_collection.insert_one({
        'email':       email,
        'name':        name,
        'created_at':  datetime.utcnow(),
        'login_count': 0,
        'last_login':  None
    })
    return jsonify({'success': True, 'message': f'User {name} added.'})


@app.route('/admin/delete_user', methods=['POST'])
@admin_required
def admin_delete_user():
    data  = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required.'}), 400
    result = users_collection.delete_one({'email': email})
    if result.deleted_count:
        return jsonify({'success': True, 'message': f'User {email} deleted.'})
    return jsonify({'error': 'User not found.'}), 404


@app.route('/admin/users')
@admin_required
def admin_get_users():
    users = list(users_collection.find({}, {'_id': 0}))
    # Convert datetime objects to strings
    for u in users:
        if u.get('created_at'):
            u['created_at'] = u['created_at'].strftime('%Y-%m-%d %H:%M')
        if u.get('last_login'):
            u['last_login'] = u['last_login'].strftime('%Y-%m-%d %H:%M')
    return jsonify(users)


# ────────────────────────────────────────────────
# ROUTES — MAIN APP
# ────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if not session.get('user_email'):
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('upload.html',
                error="No file selected.", user_name=session.get('user_name'))

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('upload.html',
                error="Invalid file type. Please upload PNG, JPG, or JPEG.",
                user_name=session.get('user_name'))

        try:
            file_bytes = file.read()
            image = Image.open(BytesIO(file_bytes))
            image.verify()
            image = Image.open(BytesIO(file_bytes))

            processed     = preprocess_image(image)
            predictions   = model.predict(processed)
            predicted_idx = int(np.argmax(predictions, axis=1)[0])
            confidence    = float(predictions[0][predicted_idx] * 100)
            confidence_pct = round(confidence, 2)
            predicted_label = class_labels[predicted_idx]

            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()

            diagnosis_details = get_diagnosis_details(predicted_label, confidence_pct)
            city      = request.form.get('city', 'Hyderabad')
            hospitals = get_neurologists_by_city(city)

            result = {
                'class':         predicted_label,
                'confidence':    f"{confidence_pct:.2f}",
                'probabilities': {
                    label: f"{predictions[0][i] * 100:.2f}"
                    for i, label in enumerate(class_labels)
                },
                'details':      diagnosis_details,
                'neurologists': hospitals,
                'city':         city,
                'api_status': {
                    'llm':    'Groq' if USE_GROQ else 'Fallback',
                    'places': 'OpenStreetMap API'
                }
            }
            return render_template('result.html', result=result,
                img_str=img_str, user_name=session.get('user_name'))

        except Exception as e:
            app.logger.error(f"Upload error: {e}")
            return render_template('upload.html',
                error=f"Error processing image: {str(e)}",
                user_name=session.get('user_name'))

    return render_template('upload.html', user_name=session.get('user_name'))


@app.route('/ask', methods=['POST'])
def ask_groq():
    if not session.get('user_email'):
        return jsonify({'error': 'Unauthorized'}), 401
    data     = request.get_json() or {}
    question = data.get('question', '').strip()
    context  = data.get('context', '').strip()
    if not question:
        return jsonify({'error': 'Please type a question.'}), 400
    reply = get_groq_response(question, context)
    return jsonify({'reply': reply})


@app.route('/health')
def health_check():
    return jsonify({
        'status':        'healthy',
        'model_loaded':  True,
        'llm_available': USE_GROQ,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)