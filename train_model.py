import cv2
import os
import pickle
import numpy as np

dataset_path = "dataset"
faces = []
labels = []
label_map = {}
current_label = 0

for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    label_map[current_label] = person

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            faces.append(img)
            labels.append(current_label)

    current_label += 1

# 🔥 THIS IS THE KEY FIX
labels = np.array(labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, labels)

os.makedirs("models", exist_ok=True)
model.save("models/lbph_model.xml")

with open("models/labels.pkl", "wb") as f:
    pickle.dump(label_map, f)

print("✅ LBPH model trained successfully")
print(f"Total faces trained: {len(faces)}")
