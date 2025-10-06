# SecurityVision

A Flask-based security application with theft detection and live camera feed.

## Features

- User authentication (login/signup)
- Live camera feed with YOLO object detection
- Theft detection using TensorFlow Lite
- Real-time alerts and dashboard
- WebSocket-based video streaming

## Local Development

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Ensure you have the required model files:
   - `yolov8n.pt` (YOLOv8 model)
   - `models/model_unquant.tflite` (Theft detection model)
   - `models/labels.txt` (Labels)

3. Run the application:
   ```
   python app.py
   ```

4. Open http://127.0.0.1:5000 in your browser.

## Deployment on Vercel

**Important Note:** This application uses camera access and WebSockets, which are not supported on Vercel's serverless platform. The camera functionality will not work in a serverless environment, and WebSockets have limitations. For full functionality, consider deploying to platforms that support persistent connections and hardware access, such as:
- Heroku
- AWS EC2
- DigitalOcean Droplets
- Google Cloud Run (with modifications)

If you still want to deploy the web interface (without camera/WebSocket features):

1. Push this code to a Git repository (e.g., GitHub).

2. Connect the repository to Vercel:
   - Go to Vercel dashboard
   - Import the project
   - Vercel will automatically detect the `vercel.json` configuration

3. The app will deploy, but camera feed and real-time features will be disabled.

## Requirements

- Python 3.8+
- Camera device (for local use only)
- AI model files in appropriate directories

## Project Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates
- `static/`: Static assets (if any)
- `instance/`: Database and instance-specific files
- `models/`: AI model files
- `api/`: Vercel deployment entry point
- `vercel.json`: Vercel configuration