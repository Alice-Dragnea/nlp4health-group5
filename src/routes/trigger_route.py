from flask import Flask, request, jsonify
from ..finetuning.model import isTrigger

trigger_route = Flask(__name__)

@trigger_route.route('/isTrigger', methods=['POST, GET'])
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
    trigger_route.run(debug=True)