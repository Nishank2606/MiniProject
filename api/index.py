import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app

# Vercel expects the Flask app to be named 'app'
# Note: This app uses camera access and WebSockets which are not supported on Vercel.
# Camera functionality will not work in serverless environment.
# Consider deploying to a platform that supports persistent connections and hardware access like Heroku or AWS EC2.