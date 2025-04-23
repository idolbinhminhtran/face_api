#!/usr/bin/env python3
import os
import json
import face_recognition

INPUT_DIR = "known_people"
OUTPUT_FILE = "models/labeled_descriptors.json"

def snake_to_title(s: str) -> str:
    """
    Convert 'tran_binh_minh' → 'Tran Binh Minh'
    """
    return s.replace('_', ' ').title()

def main():
    labeled = []

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # slug is the raw basename, e.g. 'tran_binh_minh'
        slug = os.path.splitext(fname)[0]
        display_name = snake_to_title(slug)
        path = os.path.join(INPUT_DIR, fname)
        print(f"[+] Processing {fname} as '{display_name}'")

        image = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(image)
        if not encs:
            print(f"    ⚠️  No face found in {fname}, skipping.")
            continue

        # take the first face encoding
        descriptor = encs[0].tolist()
        labeled.append({
            "slug": slug,               # machine-key if you need it
            "name": display_name,       # human-readable
            "descriptor": descriptor
        })

    # ensure the models/ folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(labeled, f, ensure_ascii=False, indent=2)

    print(f"[✓] Saved {len(labeled)} descriptors to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()