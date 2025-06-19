import cv2
import os
import pickle
import numpy as np
from keras_facenet import FaceNet

# Initialize FaceNet model
embedder = FaceNet()

# External camera index (change if necessary)
camera_index = 0  # Use 0 for the default webcam, 1 for external camera
video = cv2.VideoCapture(camera_index)

# Path to store dataset
DATASET_PATH = "face_dataset.pkl"

# Load existing dataset or create a new one
if os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "rb") as f:
        face_data = pickle.load(f)
else:
    face_data = {}

# Get user name for data collection
nameID = str(input("Enter Your Name: ")).lower()
path = 'images/' + nameID

# Check if directory exists, create if not
if not os.path.exists(path):
    os.makedirs(path)
else:
    print("Name Already Taken. Enter a different name.")
    nameID = str(input("Enter Your Name Again: "))
    path = 'images/' + nameID
    os.makedirs(path)

count = 0  # Initialize image count

# Function to detect and extract face embeddings
def get_face_embedding(image):
    faces = embedder.extract(image, threshold=0.95)  # Detect face
    if len(faces) > 0:
        return faces[0]['embedding']  # Return face embedding
    return None

# Capture and save face images
while True:
    ret, frame = video.read()
    if not ret:
        print("Error accessing camera!")
        break

    # Detect face using FaceNet
    faces = embedder.extract(frame, threshold=0.95)

    for face in faces:
        x, y, w, h = face['box']
        count += 1
        name = f'./images/{nameID}/{count}.jpg'
        print(f"Creating Image: {name}")

        # Save detected face
        cv2.imwrite(name, frame[y:y+h, x:x+w])

        # Draw a bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face embedding and store in dataset
        face_embedding = face['embedding']
        if nameID in face_data:
            face_data[nameID].append(face_embedding)
        else:
            face_data[nameID] = [face_embedding]

    # Display the live video feed
    cv2.imshow("Face Collection", frame)
    
    # Break after collecting 500 images
    if count > 500:
        break

    # Press 'q' to exit manually
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save face embeddings dataset
with open(DATASET_PATH, "wb") as f:
    pickle.dump(face_data, f)

video.release()
cv2.destroyAllWindows()
print("Face dataset collection completed!")
