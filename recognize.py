import cv2
import pickle
import sqlite3
from datetime import datetime

model = cv2.face.LBPHFaceRecognizer_create()
model.read("models/lbph_model.xml")

with open("models/labels.pkl", "rb") as f:
    labels = pickle.load(f)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

conn = sqlite3.connect("database.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    name TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
marked = set()

print("FaceTrack Attendance Started (ESC to exit)")

while True:
    ret, frame = cam.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        label, confidence = model.predict(face_img)

        if confidence < 80:  # lower = better
            name = labels[label]
            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)

            today = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")

            if name not in marked:
                cursor.execute(
                    "INSERT INTO attendance VALUES (?, ?, ?)",
                    (name, today, time)
                )
                conn.commit()
                marked.add(name)
                print(f"✅ Attendance marked for {name}")
        else:
            cv2.putText(frame, "Unknown", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("FaceTrack - Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
conn.close()
cv2.destroyAllWindows()
