# Multilingual Trigger Classification  
## for AI-Driven Patient Follow-Up Conversations  
### Team Name: **Memo**

### Team Members
- AbdulRahman Morsy
- Alice Dragnea
- Chris Gomez
- Tanvi Patel

### Important Links
#### Documents/Spreadsheets
- [📄 **View Project Proposal Here**](https://www.overleaf.com/read/jszhhtpmhzrb#ffa4ec)
- [📄 **View Final Report Here**](https://www.overleaf.com/project/68da8b1df9ea8587fe5f101f)
- [📂 **View Group Colab Drive Here**](https://drive.google.com/drive/folders/1RVIeYD97FFw62jx1QYs6KvorASOFZ7K_?usp=sharing)
- [📂 **View Sample Results/AHD Dataset Analysis Here. Note: This spreadsheet contains sample test set results used for project development and is NOT our final evaluation**](https://docs.google.com/spreadsheets/d/1bBvk63JCNymb4DyvAP-VND0TkycZqlc4Z8M_U7rAlvI/edit?usp=sharing)

#### Resources
- [🤖 **View Voiceflow Agentic Workflow Here (note, you will need to create a Voiceflow account)**](https://creator.voiceflow.com/dashboard?import=68de9e53c3844d1b755d9175)
- [🤗 **View llama3b-8b-Instruct Fine-Tuned Model Here**](https://huggingface.co/alicedr/memo-llama3-8b-instruct-arabic/blob/main/README.md)
- [🤗 **View Fine Tuned xlmr-arabic-downsampled Model Here**](https://huggingface.co/Tanvi0212/xlmr-arabic-downsampled)
- [🤗 **View Fine Tuned xlmr-english-downsampled Model Here**](https://huggingface.co/Tanvi0212/xlmr-english-downsampled)
- [🤗 **View Fine Tuned XLM-Roberta_downsampled Model Here**](https://huggingface.co/Tanvi0212/XLM-Roberta_downsampled/tree/main)


---

## Getting Started

### Prerequisites

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Set up ngrok (if you don't have an account):**
   - Sign up for a free account at [ngrok.com](https://ngrok.com)
   - Get your authtoken from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)
   - Run the following command:
     ```bash
     ngrok config add-authtoken <your_token>
     ```

## Running the Application

### Start the Server

Start both the Python server and ngrok tunnel:
```bash
npm start
```

**Note:** After starting, copy the ngrok server URL (displayed in the terminal) - you'll need it for Voiceflow configuration.

## Running Voiceflow

You have two options for running Voiceflow:

### Option 1: Voiceflow Platform (GPT Model)

**Quick Start:**
1. Visit our [Voiceflow Workflow](https://creator.voiceflow.com/dashboard?import=68de9e53c3844d1b755d9175)
2. Create a free Voiceflow account if you don't have one
3. Start testing the workflow directly in the Voiceflow interface

**Note:** This option uses Voiceflow's built-in GPT model for symptom severity assessment. To use our fine-tuned models, use Option 2.

### Option 2: CLI Phone Call (Fine-Tuned Model)

**Setup Steps:**
1. Complete the prerequisites and start the server (see sections above)
2. Create a `.env` file in the root directory with your Voiceflow credentials
   - Use the `.env.example` file as your `.env` file
3. Run the phone call script:
   ```bash
   # Windows
   python .\src\call_user.py
   
   # Mac/Linux
   python ./src/call_user.py
   ```

This option uses our fine-tuned models for more accurate symptom classification.
**Note:** We recommend calling our API endpoint ONCE before performing a Voiceflow call to ensure the Finetuned model is cached
