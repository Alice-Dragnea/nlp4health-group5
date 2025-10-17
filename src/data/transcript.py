import requests
import os
from dotenv import load_dotenv

load_dotenv()

class Transcript:
    def __init__(self):
        self.api_key = os.getenv('VOICEFLOW_API_KEY').strip()
        self.projectID = os.getenv('VOICEFLOW_PROJECT_ID').strip()
        self.versionID = os.getenv('VOICEFLOW_VERSION_ID').strip()

    def get_transcript_ids(self):
        url = f'https://analytics-api.voiceflow.com/v1/transcript/project/{self.projectID}'
        headers = {
            "Authorization": self.api_key,
            "accept": "application/json"
        }
        
        response = requests.post(url, headers=headers)

        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return []

        data = response.json()
        transcript_ids = [t["id"] for t in data.get("transcripts", [])]
        return transcript_ids

    def get_transcript(self, transcript_id):
        url = f'https://analytics-api.voiceflow.com/v1/transcript/{transcript_id}'
        headers = {
            "Authorization": self.api_key,
            "accept": "application/json"
        }
        
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print("Error:", response.status_code, response.text)
            return []

        data = response.json()
        return Transcript._extract_conversation(data)

    @staticmethod
    def _extract_conversation(transcript_data):
        """Extract just the conversation messages from transcript data"""
        if not transcript_data or 'transcript' not in transcript_data:
            return []
        
        logs = transcript_data['transcript'].get('logs', [])
        conversation = []
        
        for log in logs:
            # Assistant messages (text responses)
            if log.get('type') == 'trace' and log.get('data', {}).get('type') == 'text':
                message = log['data']['payload'].get('message', '')
                if message:  # Only add if there's actual content
                    conversation.append({
                        'speaker': 'assistant',
                        'message': message,
                    })
            
            # User messages (text input from user)
            elif log.get('type') == 'action' and log.get('data', {}).get('type') == 'text':
                message = log['data'].get('payload', '')
                if message:  # Only add if there's actual content
                    conversation.append({
                        'speaker': 'user',
                        'message': message,
                        'timestamp': log.get('createdAt')
                    })
        
        return conversation

if __name__ == "__main__":
    transcript = Transcript()
    transcript_ids = transcript.get_transcript_ids()
    print(f"Here are the transcript ids: {transcript_ids}")
    conversation = transcript.get_transcript(transcript_ids[4])
    print(f"Here is the conversation: {conversation}")