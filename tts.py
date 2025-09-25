import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.config import BaseAudioConfig
import os
from flask import Flask, request, send_file, jsonify
from langdetect import detect
import uuid
import warnings

# Allowlist XTTS classes (needed for torch serialization)
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig,
    BaseAudioConfig,
])

# Initialize Flask
app = Flask(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

from TTS.api import TTS

# Load XTTS model ONCE at startup (CPU only)
print("Loading XTTS model (CPU)...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
print("Model loaded successfully on CPU!")

# Speakers
male_speakers = ["Marcos Rudaski", "Luis Moray"]
female_speakers = ["Ana Florence"]

# Temp output folder
os.makedirs("XTTS_outputs", exist_ok=True)

@app.route("/speak", methods=["POST"])
def generate_speech():
    try:
        # Get JSON body
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        speaker_gender = data.get("chosen_SPEAKER", "").strip().lower()

        if not text:
            return jsonify({"error": "Text is required"}), 400
        if speaker_gender not in ["male", "female"]:
            return jsonify({"error": "chosen_SPEAKER must be 'male' or 'female'"}), 400

        # Select speaker
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
        output_path = f"XTTS_outputs/{file_id}.wav"

        # Generate speech
        print(f"Generating speech with {chosen_speaker}...")
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            speaker=chosen_speaker,
            language=language,
            split_sentences=True,
        )

        # Send file + delete after sending
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
            except:
                pass

        return response

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
