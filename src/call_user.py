import requests
import os
from dotenv import load_dotenv

load_dotenv()

class CallUser:
    def __init__(self):
        self.phone_number = os.getenv('PHONE_NUMBER')
        self.agent_id = os.getenv('AGENT_ID')
        self.model_name = "GPT"
        self.dm_api_key = os.getenv('VOICEFLOW_API_KEY')

    def call(self):
        url = f'https://runtime-api.voiceflow.com/v1alpha1/phone-number/{self.agent_id}/outbound'
        headers = {
            'Authorization': self.dm_api_key,
            'Content-Type': 'application/json'
        }

        data = {
            'to': self.phone_number,
            'variables': {
                'model_name': self.model_name
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.json()

if __name__ == "__main__":
    call_user = CallUser()
    response = call_user.call()
    print(response)