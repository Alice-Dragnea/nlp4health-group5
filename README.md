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
- [📄 **View xlmr Fine-Tuned Model Here**](https://huggingface.co/Tanvi0212/xlmr-trigger-detection)
- [📂 **View Results Here**](https://docs.google.com/spreadsheets/d/1bBvk63JCNymb4DyvAP-VND0TkycZqlc4Z8M_U7rAlvI/edit?usp=sharing)

#### Resources
- [🤖 **View Voiceflow Agentic Workflow Here (note, you will need to create a Voiceflow account)**](https://creator.voiceflow.com/dashboard?import=68de9e53c3844d1b755d9175)
- [🤗 **View llama-3b Fine-Tuned Model Here**](https://huggingface.co/alicedr/memo-llama-3b-instruct-arabic)

---

### Install Requirements
`pip install -r requirements.txt`

### Run Server

**Step 1:** Install Node.js dependencies
```bash
npm install
```

**Step 2:** Start both servers (Python server + ngrok)
```bash
npm start
```

#### If you don't have an ngrok account:
Run the following command and follow the instructions to set up your ngrok account:
```bash
ngrok config add-authtoken <your_token>
```
You can get your authtoken by signing up at [ngrok.com](https://ngrok.com) and visiting your [dashboard](https://dashboard.ngrok.com/get-started/your-authtoken).
