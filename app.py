import os
import json
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask import Flask, request, Response
from google.cloud import texttospeech
import face_recognition

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "tts_key.json"

# --- load your labeled descriptors ---
with open("models/labeled_descriptors.json", encoding="utf-8") as f:
    data = json.load(f)

# build mappings:
descs_by_slug = {}    # slug -> [ np.array(desc), … ]
slug_to_name = {}     # slug -> "Tran Binh Minh"
for entry in data:
    slug = entry["slug"]
    pretty = entry.get("name", slug.replace("_", " ").title())
    slug_to_name[slug] = pretty

    desc = np.array(entry["descriptor"], dtype=np.float32)
    descs_by_slug.setdefault(slug, []).append(desc)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

@app.route("/", methods=["GET"])
def health():
    return jsonify(status="ok"), 200


@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify(error="no_file"), 400

    # detect faces with the CNN model
    img = face_recognition.load_image_file(file)
    face_locs = face_recognition.face_locations(img, model="hog")
    if not face_locs:
        return jsonify(error="no_face"), 400

    # compute encodings with a bit of jitter for stability
    encs = face_recognition.face_encodings(img, face_locs)
    if not encs:
        return jsonify(error="no_encoding"), 400

    probe = encs[0]

    # find best slug match
    best_slug, best_dist = None, float("inf")
    for slug, desc_list in descs_by_slug.items():
        dists = face_recognition.face_distance(desc_list, probe)
        min_dist = float(np.min(dists))
        if min_dist < best_dist:
            best_slug, best_dist = slug, min_dist

    # tune threshold as need
    THRESHOLD = 0.6
    if best_slug and best_dist <= THRESHOLD:
        return jsonify(
            name=slug_to_name[best_slug],
            distance=round(best_dist, 4)
        )
    else:
        return jsonify(error="unknown"), 404
    
@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json() or {}
    text = data.get("text", "")
    if not text:
        return Response("Missing text", status=400)

    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="vi-VN",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )

    return Response(response.audio_content, mimetype="audio/mpeg")

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0") 
    # host = os.getenv('127.0.0.1')
    port = int(os.getenv("PORT", 8080))
    # port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)
