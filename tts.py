import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.config import BaseAudioConfig
import os
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from langdetect import detect
import uuid
import warnings
from TTS.api import TTS

# --- FIX: Set environment variable to bypass interactive TOS check ---
# The XTTS model requires confirmation of the Coqui CPML license.
# In non-interactive environments, the prompt fails with EOFError.
# Setting this variable to "1" confirms agreement to the non-commercial terms.
os.environ["COQUI_TOS_AGREED"] = "1"
# ---------------------------------------------------------------------

# Allowlist XTTS classes for torch serialization
# This is a critical step for loading the model correctly.
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig,
    BaseAudioConfig,
])

# Initialize Flask and enable CORS
app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore", category=UserWarning)

# Load XTTS model ONCE at startup (CPU only)
print("Loading XTTS model on CPU...")
try:
    # This call will now skip the interactive input due to the environment variable set above.
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
    print("Model loaded successfully on CPU!")
except Exception as e:
    print(f"Error loading model: {e}")
    tts = None

# Hardcoded speakers
male_speakers = ["Marcos Rudaski", "Luis Moray"]
female_speakers = ["Ana Florence"]

# Create a temporary output folder
os.makedirs("XTTS_outputs", exist_ok=True)

@app.route("/speak", methods=["POST"])
def generate_speech():
    """
    Generates speech from text using the XTTS model.
    """
    if not tts:
        return jsonify({"error": "TTS model failed to load. Check logs for details."}), 500
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        text = data.get("text", "").strip()
        speaker_gender = data.get("chosen_SPEAKER", "").strip().lower()

        if not text:
            return jsonify({"error": "Text is required"}), 400
        if speaker_gender not in ["male", "female"]:
            return jsonify({"error": "chosen_SPEAKER must be 'male' or 'female'"}), 400

        # Select speaker based on gender choice
        chosen_speaker = male_speakers[0] if speaker_gender == "male" else female_speakers[0]

        # Auto-detect language
        try:
            lang_code = detect(text)
            lang_map = {
                'en': 'en', 'es': 'es', 'fr': 'fr', 'de': 'de', 'it': 'it',
                'pt': 'pt', 'pl': 'pl', 'tr': 'tr', 'ru': 'ru', 'nl': 'nl',
                'cs': 'cs', 'ar': 'ar', 'zh-cn': 'zh-cn', 'ja': 'ja',
                'hu': 'hu', 'ko': 'ko'
            }
            language = lang_map.get(lang_code, 'en')
            print(f"Detected language: {language}")
        except:
            language = "en"
            print("Could not detect language, defaulting to English")

        # Generate unique filename
        file_id = str(uuid.uuid4())
        output_path = os.path.join("XTTS_outputs", f"{file_id}.wav")

        print(f"Generating speech for text in '{language}' with speaker '{chosen_speaker}'...")
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker=chosen_speaker,
            language=language,
            split_sentences=True,
        )

        # Send file and set up automatic deletion after the response is sent
        response = send_file(
            output_path,
            mimetype="audio/wav",
            as_attachment=True,
            download_name="speech.wav"
        )

        @response.call_on_close
        def cleanup():
            try:
                os.remove(output_path)
                print(f"Cleaned up file: {output_path}")
            except OSError as e:
                print(f"Error during file cleanup: {e}")
        
        return response

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Note: Use a production WSGI server like gunicorn for deployment:
    # `gunicorn -w 4 -b 0.0.0.0:5000 app:app`
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)