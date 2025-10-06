import os
import time
import threading
from datetime import datetime, timedelta, timezone
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, flash, get_flashed_messages
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from flask_sock import Sock
import simplejson
import logging

# -----------------
# Logging Setup
# -----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Suppress TensorFlow verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# -----------------
# App Config
# -----------------
class Config:
    SECRET_KEY = 'supersecretkey123'  # fixed key for dev
    basedir = os.path.abspath(os.path.dirname(__file__))
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'instance', 'users.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

# Ensure instance folder exists
instance_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'instance')
os.makedirs(instance_path, exist_ok=True)

# -----------------
# Initialize App, DB, WebSocket
# -----------------
app = Flask(__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
sock = Sock(app)

# -----------------
# Database Models
# -----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class Alert(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    message = db.Column(db.String(200), nullable=False)
    category = db.Column(db.String(50), nullable=False, default='Theft')

# -----------------
# Database Initialization
# -----------------
def ensure_database():
    """Create database and tables if missing"""
    db_path = os.path.join(instance_path, 'users.db')
    if not os.path.exists(db_path):
        logging.info("Database not found. Creating new database...")
        with app.app_context():
            db.create_all()
            create_default_admin()
        logging.info("Database created successfully.")
    else:
        logging.info("Using existing database.")

def create_default_admin():
    """Create default admin user if none exists"""
    if not User.query.filter_by(email='admin@example.com').first():
        admin = User(
            username='admin',
            email='admin@example.com',
            password=generate_password_hash('admin123')
        )
        db.session.add(admin)
        db.session.commit()
        logging.info("Default admin user created: admin@example.com / admin123")

# -----------------
# Load AI Models
# -----------------
logging.info("Loading AI models...")
yolo_model = None
tflite_interpreter = None
input_details = None
output_details = None

try:
    yolo_model = YOLO("yolov8n.pt")
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load YOLOv8 model: {e}")

try:
    tflite_interpreter = tf.lite.Interpreter(model_path="models/model_unquant.tflite")
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()
    logging.info("TFLite theft detection model loaded successfully.")
except Exception as e:
    logging.warning(f"TFLite model not loaded: {e}")

# -----------------
# Globals
# -----------------
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    logging.warning("Camera not available. Video feed disabled.")

output_frame = None
frame_lock = threading.Lock()
data_clients = set()  # connected WebSocket clients
last_alert_time = 0  # cooldown for theft alerts

# -----------------
# Background Threads
# -----------------
def capture_frames_thread():
    """Continuously capture frames from the camera"""
    global output_frame
    while True:
        try:
            success, frame = camera.read()
            if success:
                with frame_lock:
                    output_frame = frame.copy()
            time.sleep(1/30)
        except Exception as e:
            logging.error(f"Frame capture error: {e}")
            time.sleep(1)

def theft_detection_thread():
    """Run TFLite theft detection on frames"""
    global output_frame, last_alert_time
    if not tflite_interpreter:
        logging.info("TFLite theft detection disabled.")
        return
    while True:
        try:
            with frame_lock:
                if output_frame is None:
                    time.sleep(1)
                    continue
                frame = output_frame.copy()

            resized = cv2.resize(frame, (224,224)) / 255.0
            input_data = np.expand_dims(resized.astype(np.float32), axis=0)
            tflite_interpreter.set_tensor(input_details[0]['index'], input_data)
            tflite_interpreter.invoke()
            output_data = tflite_interpreter.get_tensor(output_details[0]['index'])
            prediction = output_data[0][0]
            logging.debug(f"Theft prediction: {prediction}")

            # Disabled intentional theft detection to prevent graph inflation
            # if prediction > 0.95 and (time.time() - last_alert_time) > 30:
            #     last_alert_time = time.time()
            #     with app.app_context():
            #         alert = Alert(message="Potential Theft Detected", category="Theft")
            #         db.session.add(alert)
            #         db.session.commit()
            #         logging.info(f"Theft detected at {alert.timestamp}")

            #         # push to connected clients
            #         msg = simplejson.dumps({'type': 'new_alert'})
            #         for client in list(data_clients):
            #             try:
            #                 client.send(msg)
            #             except Exception:
            #                 data_clients.remove(client)

        except Exception as e:
            logging.error(f"Theft detection error: {e}")
        time.sleep(2)

# -----------------
# Routes
# -----------------
@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username','User'))

@app.route('/api/data_summary')
def data_summary():
    if 'user_id' not in session:
        return "Unauthorized", 401
    try:
        total_alerts = Alert.query.count()
        alerts_today = Alert.query.filter(Alert.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24)).count()
        active_users = User.query.count()

        per_day_data = {(datetime.now(timezone.utc).date() - timedelta(days=i)).strftime('%b %d'):0 for i in range(6,-1,-1)}
        for alert in Alert.query.filter(Alert.timestamp >= datetime.now(timezone.utc) - timedelta(days=7)).all():
            day_str = alert.timestamp.strftime('%b %d')
            if day_str in per_day_data:
                per_day_data[day_str] += 1

        per_category_data = {cat: cnt for cat,cnt in db.session.query(Alert.category, db.func.count(Alert.category)).group_by(Alert.category).all()}

        per_hour_data = {(datetime.now(timezone.utc) - timedelta(hours=i)).strftime('%H:00'):0 for i in range(23,-1,-1)}
        for alert in Alert.query.filter(Alert.timestamp >= datetime.now(timezone.utc) - timedelta(hours=24)).all():
            hour_str = alert.timestamp.strftime('%H:00')
            if hour_str in per_hour_data:
                per_hour_data[hour_str] += 1

        return jsonify({
            "total_alerts": total_alerts,
            "alerts_today": alerts_today,
            "active_users": active_users,
            "per_day": per_day_data,
            "per_category": per_category_data,
            "per_hour": per_hour_data
        })
    except Exception as e:
        logging.error(f"Data summary error: {e}")
        return "Internal Server Error", 500

@app.route('/api/get_alerts')
def get_alerts():
    if 'user_id' not in session:
        return "Unauthorized", 401
    try:
        alerts = Alert.query.order_by(Alert.timestamp.desc()).limit(5).all()
        return jsonify([{'message': alert.message, 'timestamp': alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')} for alert in alerts])
    except Exception as e:
        logging.error(f"Get alerts error: {e}")
        return "Internal Server Error", 500

# -----------------
# Authentication
# -----------------
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method=='POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        confirm_password = request.form.get('confirm_password','')

        if not all([username,email,password,confirm_password]):
            flash("All fields required", "danger")
            return redirect(url_for('signup'))

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for('signup'))

        if len(password) < 6:
            flash("Password must be at least 6 characters", "warning")
            return redirect(url_for('signup'))

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for('signup'))

        user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        flash("Signup successful. Login now.", "success")
        return redirect(url_for('login'))
    else:
        # Consume any old flashed messages from other pages
        get_flashed_messages()
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f"Welcome back {user.username}", "success")
            return redirect(url_for('home'))

        flash("Invalid credentials", "danger")
        return redirect(url_for('login'))
    else:
        # Consume any old flashed messages from other pages
        get_flashed_messages()
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for('login'))

# -----------------
# WebSocket
# -----------------
@sock.route('/ws/video_feed')
def ws_video_feed(ws):
    global output_frame
    while True:
        try:
            with frame_lock:
                if output_frame is None:
                    time.sleep(0.01)
                    continue
                frame = output_frame.copy()

            small_frame = cv2.resize(frame, (640,480))
            annotated = small_frame

            if yolo_model:
                results = yolo_model(small_frame, stream=False, verbose=False)
                annotated = results[0].plot()

            ret, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY),65])
            if ret:
                ws.send(buffer.tobytes())

            time.sleep(1/25)
        except Exception:
            break

@sock.route('/ws/data_updates')
def ws_data_updates(ws):
    data_clients.add(ws)
    try:
        while True:
            ws.wait()
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
    finally:
        if ws in data_clients:
            data_clients.remove(ws)

# -----------------
# Run App
# -----------------
if __name__ == '__main__':
    ensure_database()
    threading.Thread(target=capture_frames_thread, daemon=True).start()
    threading.Thread(target=theft_detection_thread, daemon=True).start()
    logging.info("Server running at http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True)
