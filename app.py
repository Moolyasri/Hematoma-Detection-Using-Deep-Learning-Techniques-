import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from PIL import Image
import io
import pymongo
from bson.objectid import ObjectId
import bcrypt
from datetime import datetime
import geopy.distance
from geopy.geocoders import Nominatim
from functools import wraps
import time

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# MongoDB Connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Brain_sdp"]
users_collection = db["users"]
test_results_collection = db["test_results"]
hospitals_collection = db["hospitals"]

# Initialize hospitals collection with some sample data if empty
if hospitals_collection.count_documents({}) == 0:
    sample_hospitals = [
        {"name": "City Hospital", "lat": 12.9716, "lng": 77.5946, "city": "Bangalore", "specialty": "Neurology", "phone": "123-456-7890"},
        {"name": "General Medical Center", "lat": 13.0827, "lng": 80.2707, "city": "Chennai", "specialty": "Neuro-oncology", "phone": "234-567-8901"},
        {"name": "Community Health Center", "lat": 17.3850, "lng": 78.4867, "city": "Hyderabad", "specialty": "Neurosurgery", "phone": "345-678-9012"},
        {"name": "University Hospital", "lat": 18.5204, "lng": 73.8567, "city": "Pune", "specialty": "Neuroimaging", "phone": "456-789-0123"},
        {"name": "Rural Medical Facility", "lat": 28.7041, "lng": 77.1025, "city": "Delhi", "specialty": "Neurology", "phone": "567-890-1234"}
    ]
    hospitals_collection.insert_many(sample_hospitals)

# Load trained model
MODEL_PATH = r"D:\Ai\mysdp1\unet_lstm_model (1).h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
CLASS_NAMES = ["Glioma","Meningioma","No Hematoma", "Pituitary"]

# Function to preprocess the image
def preprocess_image(image):
    """Preprocesses an image: converts to RGB, resizes, normalizes, and reshapes."""
    image = image.convert("RGB")  # Convert to RGB (3 channels)
    image = image.resize((64, 64))  # Resize to match expected model input
    img_array = np.array(image) / 255.0  # Normalize pixel values (0-1 range)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 64, 64, 3)
    return img_array

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Route for the homepage
@app.route('/')
def home():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = users_collection.find_one({'username': username})
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
            session['user_id'] = str(user['_id'])
            session['username'] = user['username']
            session['location'] = user.get('location', {})
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
            return render_template('login.html', error='Invalid username or password')
    
    return render_template('login.html')

# Helper function to get city from coordinates
def get_city_from_coordinates(lat, lng):
    try:
        geolocator = Nominatim(user_agent="brain_scan_app")
        time.sleep(1)  # To avoid hitting rate limits
        location = geolocator.reverse(f"{lat}, {lng}", language="en")
        
        if location and location.raw.get('address'):
            address = location.raw['address']
            # Try different address components to find city name
            city = (address.get('city') or address.get('town') or 
                    address.get('village') or address.get('county') or 
                    address.get('state_district') or address.get('state'))
            return city
        return None
    except Exception as e:
        print(f"Geocoding error: {str(e)}")
        return None

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        lat = request.form.get('lat')
        lng = request.form.get('lng')
        
        # Check if username already exists
        if users_collection.find_one({'username': username}):
            flash('Username already exists')
            return render_template('signup.html', error='Username already exists')
        
        # Hash the password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        # Get city from coordinates if available
        city = None
        if lat and lng:
            city = get_city_from_coordinates(float(lat), float(lng))
        
        # Create user document
        user = {
            'username': username,
            'password': hashed_password,
            'email': email,
            'location': {
                'lat': float(lat) if lat else None,
                'lng': float(lng) if lng else None,
                'city': city
            },
            'created_at': datetime.now()
        }
        
        # Insert user into database
        result = users_collection.insert_one(user)
        
        # Set session variables
        session['user_id'] = str(result.inserted_id)
        session['username'] = username
        session['location'] = user['location']
        
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# Dashboard route
@app.route('/dashboard')
@login_required
def dashboard():
    # Get user's test results
    user_results = list(test_results_collection.find({'user_id': session['user_id']}).sort('created_at', -1))
    
    # Find nearby hospitals based on user location
    nearby_hospitals = []
    city_hospitals = []
    user_city = None
    
    if session.get('location') and session['location'].get('lat') and session['location'].get('lng'):
        user_coords = (session['location']['lat'], session['location']['lng'])
        user_city = session['location'].get('city')
        all_hospitals = list(hospitals_collection.find())
        
        # Get hospitals by distance
        for hospital in all_hospitals:
            hospital_coords = (hospital['lat'], hospital['lng'])
            distance = geopy.distance.distance(user_coords, hospital_coords).kilometers
            if distance <= 50:  # Hospitals within 50 km
                hospital['distance'] = round(distance, 2)
                nearby_hospitals.append(hospital)
        
        # Sort by distance
        nearby_hospitals.sort(key=lambda x: x['distance'])
        
        # Get hospitals in the same city
        if user_city:
            # First try exact city match
            city_hospitals = list(hospitals_collection.find({"city": user_city}))
            
            # If no exact matches, try case-insensitive match
            if not city_hospitals:
                all_hospitals_list = list(hospitals_collection.find())
                city_hospitals = [h for h in all_hospitals_list if h.get('city') and 
                                  h['city'].lower() == user_city.lower()]
    
    return render_template('dashboard.html', 
                          results=user_results, 
                          hospitals=nearby_hospitals, 
                          city_hospitals=city_hospitals, 
                          user_city=user_city)

# Prediction API endpoint
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Validate file in request
        if 'image' not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400

        # Read the image
        image = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(image)

        # Make prediction
        predictions = model.predict(processed_img)[0]
        predicted_class_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_class_index]
        confidence = float(predictions[predicted_class_index])

        # Format class probabilities
        probabilities = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
        
        # Save test result to database
        test_result = {
            'user_id': session['user_id'],
            'prediction': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'created_at': datetime.now()
        }
        test_results_collection.insert_one(test_result)

        return jsonify({
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Update user location
@app.route('/update_location', methods=['POST'])
@login_required
def update_location():
    try:
        data = request.json
        lat = data.get('lat')
        lng = data.get('lng')
        manual_city = data.get('city')
        
        if lat and lng:
            # Use manually entered city or get from coordinates
            city = manual_city if manual_city else get_city_from_coordinates(float(lat), float(lng))
            
            # Update user location in database
            users_collection.update_one(
                {'_id': ObjectId(session['user_id'])},
                {'$set': {'location': {
                    'lat': float(lat), 
                    'lng': float(lng),
                    'city': city
                }}}
            )
            
            # Update session
            session['location'] = {
                'lat': float(lat), 
                'lng': float(lng),
                'city': city
            }
            
            return jsonify({"success": True, "city": city})
        else:
            return jsonify({"success": False, "error": "Invalid location data"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Add a new hospital
@app.route('/add_hospital', methods=['POST'])
@login_required
def add_hospital():
    try:
        data = request.json
        name = data.get('name')
        lat = data.get('lat')
        lng = data.get('lng')
        city = data.get('city')
        specialty = data.get('specialty')
        phone = data.get('phone')
        
        # Validate required fields
        if not name or not city or not specialty:
            return jsonify({"success": False, "error": "Name, city and specialty are required"}), 400
            
        # Create hospital document
        hospital = {
            'name': name,
            'lat': float(lat) if lat else None,
            'lng': float(lng) if lng else None,
            'city': city,
            'specialty': specialty,
            'phone': phone,
            'added_by': session['user_id'],
            'created_at': datetime.now()
        }
        
        # Insert hospital into database
        result = hospitals_collection.insert_one(hospital)
        
        return jsonify({"success": True, "message": "Hospital added successfully"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
