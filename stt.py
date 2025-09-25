import os
import torch
import whisper
import warnings
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress user warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load Whisper model ONCE
print("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("small", device=device)
print(f"Model loaded successfully on {device}!")

# Ensure output folder exists
os.makedirs("Whisper_outputs", exist_ok=True)


def transcribe_audio(file_path):
    """Transcribe audio with automatic language detection."""
    try:
        audio = whisper.load_audio(file_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # Detect language
        _, probs = model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        confidence = probs[detected_language]

        # Transcribe
        result = model.transcribe(
            file_path,
            fp16=False,
            language=detected_language
        )

        return {
            "text": result["text"],
            "language": detected_language,
            "confidence": float(confidence)
        }
    except Exception as e:
        return {"error": str(e)}


@app.route("/transcribe", methods=["POST"])
def api_transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    uploaded_file = request.files["file"]
    if uploaded_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save temporary file
    file_id = str(uuid.uuid4())
    input_path = os.path.join("Whisper_outputs", f"{file_id}.wav")
    uploaded_file.save(input_path)

    # Transcribe
    result = transcribe_audio(input_path)

    # Cleanup
    os.remove(input_path)

    return jsonify(result)


if __name__ == "__main__":
    # Railway will inject PORT env var
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
