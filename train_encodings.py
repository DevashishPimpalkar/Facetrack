import face_recognition
import os
import pickle
import cv2

DATASET_DIR = "dataset"
ENCODING_FILE = "encodings/face_encodings.pkl"

known_encodings = []
known_names = []

print("Training face encodings...")

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)

        try:
            image = cv2.imread(img_path)

            if image is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)

            if encodings:
                known_encodings.append(encodings[0])
                known_names.append(person)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

data = {
    "encodings": known_encodings,
    "names": known_names
}

os.makedirs("encodings", exist_ok=True)

with open(ENCODING_FILE, "wb") as f:
    pickle.dump(data, f)

print("✅ Face encodings training completed")
print(f"Total faces encoded: {len(known_encodings)}")
