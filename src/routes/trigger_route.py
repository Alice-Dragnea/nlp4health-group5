import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import logging

# Configure logging to output to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()  # This ensures output goes to console
    ]
)

from flask import Flask, request, jsonify
from flask_cors import CORS
from models.model import isTrigger

trigger_route = Flask(__name__)
CORS(trigger_route)

@trigger_route.route('/isTrigger', methods=['POST'])
def isTriggerRoute():
    try:
        data = request.get_json()

        # Validate input 
        if "side_effect" not in data:
            return jsonify({
                'error': 'Missing required field: side_effect',
                'success': False
            }), 400
        
        text = data["side_effect"]
        trigger_route.logger.info(f"Received data: {data}")

        # Dynamic model selection 
        # If Voiceflow sends "model_id" or "model_name", use it
        # Otherwise default to your best model
        model_id = data.get(
            "model_id",
            data.get("model_name", "Tanvi0212/xlmr-trigger-detection")  # accept both model_id and model_name
        )

        # Predict
        result = isTrigger(text, model_id)
        if result["label"] == "Trigger":
            is_trigger = "True"
        else:
            is_trigger = "False"

        trigger_route.logger.info(f"Result: {result}")
        trigger_route.logger.info(f"Score: {result['confidence']}")

        return jsonify({
            'is_trigger': is_trigger
        }), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    trigger_route.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000)),
        debug=True
    )
