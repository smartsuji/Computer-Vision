import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing module (optional, for visualization)
mp_drawing = mp.solutions.drawing_utils

# Step 1: Open the camera (or use a video file)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Step 2: Define the region of interest (ROI) for focusing (e.g., center of the frame)
roi_top = 100    # Y-coordinate of the top edge of the ROI
roi_bottom = 400  # Y-coordinate of the bottom edge of the ROI
roi_left = 150    # X-coordinate of the left edge of the ROI
roi_right = 450   # X-coordinate of the right edge of the ROI

# Step 3: Start a timer to limit the recording to 10 seconds
start_time = time.time()
record_duration = 50  # Duration in seconds

# Step 4: Process the camera feed frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    if elapsed_time > record_duration:
        print("Recording complete: 10 seconds elapsed.")
        break

    # Step 5: Crop the region of interest (ROI) from the frame
    roi = frame[roi_top:roi_bottom, roi_left:roi_right]

    # Convert the ROI to RGB (MediaPipe requires RGB input)
    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Step 6: Perform hand landmark detection within the ROI
    result = hands.process(rgb_roi)

    # Step 7: If landmarks are detected, process them
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Optional: Draw landmarks on the ROI (visualization)
            mp_drawing.draw_landmarks(roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Step 8: Display the ROI on the screen
    cv2.imshow('Focused Region - Hand Tracking', roi)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release resources
cap.release()
cv2.destroyAllWindows()
hands.close()
