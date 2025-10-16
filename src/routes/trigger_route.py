import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from ..finetuning.model import isTrigger

trigger_route = Flask(__name__)
CORS(trigger_route)

@trigger_route.route('/isTrigger', methods=['POST'])
def isTriggerRoute():
    try:
        data = request.get_json()

        # validate required fields
        required_fields = ['side_effect']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing or empty required field: {field}',
                    'success': False
                }), 400
        
        # TODO: implement trigger logic/model.
        return jsonify({
            'success': True,
            'is_trigger': isTrigger(data['side_effect'])
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