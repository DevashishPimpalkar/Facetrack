import cv2
import face_recognition
import pickle
import sqlite3
from datetime import datetime

with open("encodings/face_encodings.pkl", "rb") as f:
    data = pickle.load(f)

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    name TEXT,
    date TEXT,
    time TEXT
)
""")

marked_today = set()
cam = cv2.VideoCapture(0)

print("FaceTrack Attendance Started (ESC to exit)")

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # Mirror the frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    for enc in face_encodings:
        matches = face_recognition.compare_faces(data["encodings"], enc)
        if True in matches:
            index = matches.index(True)
            name = data["names"][index]

            today = datetime.now().strftime("%Y-%m-%d")
            time = datetime.now().strftime("%H:%M:%S")

            if name not in marked_today:
                cursor.execute(
                    "INSERT INTO attendance VALUES (?, ?, ?)",
                    (name, today, time)
                )
                conn.commit()
                marked_today.add(name)
                print(f"✅ Attendance marked for {name}")

    cv2.imshow("FaceTrack - Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
conn.close()
cv2.destroyAllWindows()
