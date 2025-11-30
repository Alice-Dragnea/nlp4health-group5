import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

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

        # Dynamic model selection 
        # If Voiceflow sends "model_id" or "model_name", use it
        # Otherwise default to your best model
        model_id = data.get(
            "model_id",
            data.get("model_name", "Tanvi0212/xlmr-trigger-eng-ar")  # accept both model_id and model_name
        )

        # Predict
        result = isTrigger(text, model_id)

        return jsonify({
            'success': True,
            'model_used': model_id,
            'prediction': result
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
