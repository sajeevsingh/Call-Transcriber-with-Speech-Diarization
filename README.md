# Call-Transcriber-with-Speech-Diarization

# Speech Diarization App

This is a Flask-based web application for audio diarization and transcription. The app uses Pyannote for speaker diarization and Whisper for transcription, allowing you to upload audio files and view identified speakers with their corresponding transcriptions.

---

## Features

- **Audio Diarization**: Identify speakers in uploaded audio files.
- **Speech Transcription**: Generate text transcriptions from audio.
- **User-Friendly Interface**: Intuitive UI for uploading and processing audio files.
- **GPU Acceleration**: Utilizes GPU for faster processing if available.

---

## Requirements

- Python 3.8 or higher
- Libraries: `Flask`, `Flask-CORS`, `Pyannote`, `Whisper`, `PyDub`, `Torch`, `Ngrok`

---

## Installation and Setup

```bash
# Clone the Repository
git clone https://github.com/sajeevsingh/Call-Transcriber-with-Speech-Diarization.git
cd Call-Transcriber-with-Speech-Diarization
```
---
## Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
---
## Install Dependencies
```bash
pip install -r requirements.txt
```
---

## Generate Required Tokens
### Hugging Face Token
1. Visit https://huggingface.co/
2. Sign up or log in
3. Generate a token from your account settings and replace `hf_token` in the code

### Ngrok Auth Token
1. Visit https://ngrok.com/
2. Sign up or log in
3. Go to "Auth" in your dashboard, copy your token, and authenticate
```bash
ngrok config add-authtoken <your_ngrok_auth_token>
```
---

## File Structure
```bash
speech-diarization-app/
├── app.py                  # Main Flask app
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates
│   ├── landing.html
│   └── Speech-Diarization.html
├── static/                 # Static files
│   ├── css/
│   │   ├── speech-diarization.css
│   │   └── styles.css
│   └── images/
│       ├── image.jpg
│       └── image2.jpg
└── README.md               # Project documentation
```
---

## Usage
```bash
python app.py
```
---

## Access the Application
- After starting the app, an Ngrok public URL will be displayed in the terminal, e.g.:
- Public URL: http://<ngrok_subdomain>.ngrok.io
- Open this URL in your browser

## Upload an Audio File
1. Use the landing page to navigate to the diarization page.
2. Upload an audio file (.mp3 or .wav).
3. View the diarization and transcription results.

---

## Notes
- GPU Utilization: The app uses GPU if available for faster processing.
- Audio Preprocessing: Files are converted to 16 kHz mono .wav format.

