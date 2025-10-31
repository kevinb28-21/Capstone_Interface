"""
Flask API Server for Image Processing
Receives images from Raspberry Pi and processes them using OpenCV/TensorFlow.
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from image_processor import analyze_crop_health, calculate_ndvi
import json

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', './uploads')
PROCESSED_FOLDER = os.getenv('PROCESSED_FOLDER', './processed')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tiff', 'tif'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# In-memory store (replace with PostgreSQL later)
images_db = {}
analyses_db = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'service': 'flask-image-processor'})


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload endpoint for images from Raspberry Pi.
    Accepts: multipart/form-data with 'image' field and optional 'gps' (JSON string)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = str(int(os.path.getmtime(__file__) * 1000)) if os.path.exists(__file__) else str(int(__import__('time').time() * 1000))
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    
    # Get GPS metadata if provided
    gps_data = None
    if 'gps' in request.form:
        try:
            gps_data = json.loads(request.form['gps'])
        except:
            pass
    
    # Process image
    try:
        analysis = analyze_crop_health(filepath, use_tensorflow=False)  # Set True when model ready
        
        # Store in memory (replace with PostgreSQL)
        image_id = str(len(images_db))
        images_db[image_id] = {
            'id': image_id,
            'filename': unique_filename,
            'original_name': filename,
            'path': f'/uploads/{unique_filename}',
            'processed_path': analysis.get('processed_image_path', ''),
            'gps': gps_data,
            'created_at': timestamp
        }
        
        analyses_db[image_id] = {
            'image_id': image_id,
            'analysis': analysis
        }
        
        return jsonify({
            'id': image_id,
            'filename': unique_filename,
            'path': f'/uploads/{unique_filename}',
            'analysis': {
                'ndvi': analysis['ndvi_mean'],
                'summary': analysis['summary'],
                'health_status': analysis['health_status'],
                'stress_zones': analysis['stress_zones']
            },
            'gps': gps_data
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data', methods=['GET'])
def get_data():
    """
    Retrieve processed image data and analyses.
    Query params: ?image_id=<id> for single image, or omit for all
    """
    image_id = request.args.get('image_id')
    
    if image_id:
        if image_id not in images_db:
            return jsonify({'error': 'Image not found'}), 404
        
        img = images_db[image_id]
        analysis = analyses_db.get(image_id, {})
        
        return jsonify({
            'image': img,
            'analysis': analysis.get('analysis', {})
        })
    else:
        # Return all images
        result = []
        for img_id, img in images_db.items():
            analysis = analyses_db.get(img_id, {})
            result.append({
                'id': img_id,
                'filename': img['original_name'],
                'path': img['path'],
                'created_at': img['created_at'],
                'gps': img.get('gps'),
                'analysis': {
                    'ndvi': analysis.get('analysis', {}).get('ndvi_mean', 0),
                    'summary': analysis.get('analysis', {}).get('summary', 'Pending'),
                    'health_status': analysis.get('analysis', {}).get('health_status', 'Unknown')
                }
            })
        
        return jsonify({'images': result})


@app.route('/uploads/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve uploaded images"""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(filepath):
        return send_file(filepath)
    return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    port = int(os.getenv('FLASK_PORT', 5001))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)

