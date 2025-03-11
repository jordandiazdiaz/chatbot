"""
Flask Web Application for Suspicious Account Activity and Billing Support Chatbot.

This module implements the web interface for the chatbot, handling HTTP requests,
session management, and rendering responses.
"""

import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, send_from_directory
import json
from dotenv import load_dotenv

# Import chatbot module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.chatbot import SupportChatbot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__, 
            static_folder="../static",
            template_folder="../templates")

# Configure app
app.config["DEBUG"] = os.getenv("DEBUG", "False").lower() == "true"
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

# Initialize chatbot
chatbot = SupportChatbot()

@app.route('/')
def index():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve static images."""
    return send_from_directory(os.path.join(app.root_path, '../static/images'), filename)

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    Process chat messages and return responses.
    
    Request format:
    {
        "message": "User's question",
        "reset": false  # Optional, resets conversation if true
    }
    
    Response format:
    {
        "answer": "Chatbot's response",
        "sources": ["source1.pdf", "source2.pdf"],
        "images": ["image1.jpg", "image2.png"],
        "timestamp": "2023-03-10T15:30:45.123456"
    }
    """
    try:
        # Get request data
        data = request.json
        user_message = data.get('message', '')
        reset = data.get('reset', False)
        
        # Log request
        logger.info(f"Chat request: {user_message}")
        
        # Reset conversation if requested
        if reset:
            chatbot.reset_conversation()
            return jsonify({
                "answer": "Conversation has been reset.",
                "sources": [],
                "images": [],
                "timestamp": datetime.now().isoformat()
            })
        
        # Get response from chatbot
        response = chatbot.get_answer(user_message)
        
        # Log response
        logger.info(f"Chat response: {response['answer'][:100]}...")
        logger.info(f"Sources: {response['sources']}")
        logger.info(f"Token usage: {response['token_usage']}")
        
        # Format response for API
        api_response = {
            "answer": response['answer'],
            "sources": [os.path.basename(source) for source in response['sources']],
            "images": [os.path.basename(image) for image in response['images']],
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(api_response)
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return jsonify({
            "answer": "I apologize, but I encountered an error processing your request. Please try again later.",
            "sources": [],
            "images": [],
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

def create_app() -> Flask:
    """Create and configure the Flask app for WSGI servers."""
    return app

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)