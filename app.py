from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from pyngrok import ngrok
from pyannote.audio import Pipeline
import whisper
from pydub import AudioSegment
import pandas as pd
import os
import torch

# Initialize Flask app
app = Flask(__name__, template_folder='/content/templates', static_folder='/content/static')
CORS(app)

# Base directory
BASE_DIR = "/content"

# Hugging Face token
hf_token = "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"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize diarization pipeline
try:
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=hf_token
    )
    diarization_pipeline.to(device)  # Move diarization pipeline to GPU if available
    print("Diarization pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading diarization pipeline: {str(e)}")

# Initialize Whisper model
try:
    whisper_model = whisper.load_model("small", device=device.type)  # Load Whisper model on GPU
    print("Whisper model loaded successfully.")
except Exception as e:
    print(f"Error loading Whisper model: {str(e)}")

# Landing page route
@app.route("/")
def index():
    return render_template("landing.html")

# Speech diarization page route
@app.route("/upload")
def upload():
    return render_template("Speech-Diarization.html")

# Upload endpoint for processing files
@app.route("/process", methods=["POST"])
def process_file():
    try:
        # Get uploaded file
        file = request.files.get("file")
        if not file:
            print("No file received.")
            return jsonify({"error": "No file uploaded"}), 400

        # Save uploaded file
        uploaded_file_path = os.path.join(BASE_DIR, "uploaded.mp3")
        file.save(uploaded_file_path)
        print(f"File saved at {uploaded_file_path}")

        # Step 1: Preprocess audio
        audio = AudioSegment.from_file(uploaded_file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        processed_audio_path = os.path.join(BASE_DIR, "audio.wav")
        audio.export(processed_audio_path, format="wav")

        # Step 2: Perform diarization
        diarization = diarization_pipeline(processed_audio_path)
        diarization_array = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarization_array.append([turn.start, turn.end, speaker])
        print("Step-1 done")

        # Compress diarization segments
        compressed_diarization = []
        for i in range(len(diarization_array)):
            if i == 0:
                compressed_diarization.append(diarization_array[i])
            else:
                if diarization_array[i][2] == compressed_diarization[-1][2]:
                    compressed_diarization[-1][1] = diarization_array[i][1]
                else:
                    compressed_diarization.append(diarization_array[i])

        # Step 3: Perform transcription
        result = whisper_model.transcribe(processed_audio_path)
        transcription_segments = result["segments"]
        print("Transcription Done")

        # Step 4: Align transcription with diarization
        aligned_data = []
        for segment in transcription_segments:
            start, end = segment["start"], segment["end"]
            best_overlap = 0
            best_speaker = None
            for diarization_segment in compressed_diarization:
                overlap = min(end, diarization_segment[1]) - max(start, diarization_segment[0])
                if overlap > 0:
                    best_overlap = overlap
                    best_speaker = diarization_segment[2]
            if best_speaker:
                aligned_data.append({"speaker": best_speaker, "text": segment["text"]})

        # Return the aligned data
        return jsonify({"transcription": aligned_data})

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    # Start Flask app and ngrok tunnel
    port = 5000
    public_url = ngrok.connect(port).public_url
    print(f"Public URL: {public_url}")
    app.run(host="0.0.0.0", port=port)
