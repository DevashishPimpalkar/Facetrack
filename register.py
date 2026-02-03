import cv2
import os

name = input("Enter Student Name: ")
roll = input("Enter Roll No: ")

folder = f"dataset/{name}_{roll}"
os.makedirs(folder, exist_ok=True)

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # width
cam.set(4, 480)  # height

count = 0
MAX_IMAGES = 20

print("\nInstructions:")
print("- Look at camera normally")
print("- Change angle slightly after every few shots")
print("- Press 's' to save image")
print("- ESC to exit\n")

while True:
    ret, frame = cam.read()
    frame = cv2.flip(frame, 1) # Mirror the frame
    if not ret:
        break

    cv2.putText(frame, f"Images Captured: {count}/{MAX_IMAGES}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)

    cv2.imshow("Face Registration - FaceTrack", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        count += 1
        cv2.imwrite(f"{folder}/{count}.jpg", frame)
        print(f"Saved image {count}")
        
    if count >= MAX_IMAGES or key == 27:
        break

cam.release()
cv2.destroyAllWindows()
print("✅ Dataset creation completed")
