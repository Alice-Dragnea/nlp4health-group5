import requests
import os
import time
from dotenv import load_dotenv
from voiceflow_data.transcript import Transcript
import os

load_dotenv()

class CallUser:
    def __init__(self):
        self.phone_number = os.getenv('PHONE_NUMBER')
        self.agent_id = os.getenv('AGENT_ID')
        self.dm_api_key = os.getenv('VOICEFLOW_API_KEY')
        self.models = ["GPT", "Claude", "Tanvi0212/xlmr-trigger-eng-ar", "Tanvi0212/xlmr-english-downsampled", "Tanvi0212/xlmr-arabic-downsampled"]
        self.model_name = "GPT"
        self.transcript = Transcript()

    def call(self):
        # check if model is valid
        if self.model_name not in self.models:
            print(f"Invalid model: {self.model_name}")
            return None

        # Get current transcript IDs before the call
        initial_ids = self.transcript.get_transcript_ids()

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

        print("Calling user...")
        requests.post(url, headers=headers, json=data)

        # Wait for new transcript to appear
        print("Waiting for transcript ID to be generated...")
        return self._wait_for_transcript_id(initial_ids)

    def _wait_for_transcript_id(self, initial_ids, poll_interval=5, max_wait=300):
        """Poll until a new transcript ID appears"""
        start_time = time.time()

        while time.time() - start_time < max_wait:
            time.sleep(poll_interval)

            current_ids = self.transcript.get_transcript_ids()
            new_ids = [tid for tid in current_ids if tid not in initial_ids]

            if new_ids:
                return new_ids[0]

        print("Timeout waiting for transcript")
        return None

if __name__ == "__main__":
    call_user = CallUser()
    
    # Return Transcript ID
    transcript_id = call_user.call()
    print(f"Transcript ID: {transcript_id}")
    
    # Grab rel path
    relative_path = f"python src/voiceflow_data/transcript.py {transcript_id}"
    print(f"Run the following command to get the transcript: {relative_path}")