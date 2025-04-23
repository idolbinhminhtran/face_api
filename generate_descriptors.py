#!/usr/bin/env python3
import os
import json
import face_recognition

INPUT_DIR    = "known_people"
OUTPUT_FILE  = "models/labeled_descriptors.json"
MAPPING_FILE = "slug_to_name.json"   # ← đường dẫn tới file bạn vừa tạo

def snake_to_title(s: str) -> str:
    return s.replace('_', ' ').title()

def main():
    # 1) Load mapping slug→tên có dấu
    with open(MAPPING_FILE, encoding="utf-8") as mf:
        slug_map = json.load(mf)

    labeled = []
    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        slug = os.path.splitext(fname)[0]
        # 2) Ưu tiên tìm trong mapping, không có thì fallback
        display_name = slug_map.get(slug, snake_to_title(slug))

        print(f"[+] Processing {fname} as '{display_name}'")
        image = face_recognition.load_image_file(os.path.join(INPUT_DIR, fname))
        encs  = face_recognition.face_encodings(image)
        if not encs:
            print(f"    ⚠️  No face found in {fname}, skipping.")
            continue

        labeled.append({
            "slug":       slug,
            "name":       display_name,
            "descriptor": encs[0].tolist()
        })

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(labeled, out, ensure_ascii=False, indent=2)

    print(f"[✓] Saved {len(labeled)} descriptors to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
