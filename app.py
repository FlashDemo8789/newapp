from flask import Flask, jsonify
from flask_cors import CORS
from api import init_routes
import json
import numpy as np
import pandas as pd

# Define Config class locally instead of importing
class Config:
    DEBUG = True
    JSON_SORT_KEYS = False
    SECRET_KEY = "flashdna-development-key"

# Custom JSON encoder class to handle specific objects like pandas DataFrames
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif hasattr(obj, 'to_json'):
            return obj.to_json()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return super().default(obj)

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Set custom JSON encoder
    app.json.encoder = CustomJSONEncoder
    
    # Enable CORS for React frontend
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Initialize routes
    init_routes(app)
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "message": str(e)}), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({"error": "Server error", "message": str(e)}), 500
    
    # Health check endpoint
    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({"status": "ok", "message": "Flash DNA Analysis API is running"})
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to port 5001 to avoid macOS AirPlay conflict
