import requests
import os

"""
# PLEASE READ !! 
  Our fine tuned llama3b-Instruct model is too big to call w the HF api for free (16BG). 
  Since we have a free colab pro account, the temporary working solution is to run our model on one of the colab GPUs (hence the colab inference server notebook Llama3B_Inference_Server.ipynb...)
  
  For now, each time we want to start a demo using alicedr/memo-llama3-8b-instruct-arabic, we have to first run the server inference notebook on google colab to generate our new ngrok_url. 
  Then this ngrok_url must be pasted below and saved before running our voiceflow system.

  An alternative to this may be to try using a 4 bit quantized version of our model instead since it may be small enough to make free hf api calls, but I need to do some evals first to check if the 4bit version performance is worse

  --alice
"""

ngrok_url = 'https://YOUR_NGROK_URL/classify' #run Llama3B_Inference_Server.ipnyb first on GOOGLE COLAB to generate ngrok_url. 


#Colab ngrok URL stored as an environment variable
COLAB_API_URL = os.getenv('COLAB_API_URL', ngrok_url)

def isTrigger(text: str):
    """
    Classify patient statement by calling Colab API.
    
    Args:
        text: Patient statement in English or Arabic
        
    Returns:
        dict with 'label' and 'confidence' (probability of the predicted label)
    """
    try:
        response = requests.post(
            COLAB_API_URL,
            json={"text": text},
            timeout=30  #30 second timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            predicted_label = result['label']  # 0 or 1
            
            # Get the probability for the predicted label
            if predicted_label == 1:
                confidence = result['prob_1']  # Probability model assigned to label 1
            else:
                confidence = result['prob_0']  #Probability model assigned to label 0
            
            return {
                "label": "Trigger" if predicted_label == 1 else "Non-trigger",
                "confidence": confidence  #probability of predicted label
            }
        else:
            # Fallback if API fails
            print(f"API error: {response.status_code}")
            return {
                "label": "COLAB API FAILED",
                "confidence": 0.0
            }
    
    except Exception as e:
        #flag label produced if error occurs
        print(f"Error calling Colab API: {e}")
        return {
            "label": "ERROR CALLING COLAB API",
            "confidence": 0.0
        }