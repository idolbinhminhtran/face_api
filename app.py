import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition

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

@app.route("/recognize", methods=["POST"])
def recognize():
    file = request.files.get("image")
    if not file:
        return jsonify(error="no_file"), 400

    # detect faces with the CNN model
    img = face_recognition.load_image_file(file)
    face_locs = face_recognition.face_locations(img, model="cnn")
    if not face_locs:
        return jsonify(error="no_face"), 400

    # compute encodings with a bit of jitter for stability
    encs = face_recognition.face_encodings(img, face_locs, num_jitters=2)
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

    # tune threshold as needed
    THRESHOLD = 0.6
    if best_slug and best_dist <= THRESHOLD:
        return jsonify(
            name=slug_to_name[best_slug],
            distance=round(best_dist, 4)
        )
    else:
        return jsonify(error="unknown"), 404

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0") 
    port = int(os.getenv("PORT", 5000))
    app.run(host=host, port=port)
