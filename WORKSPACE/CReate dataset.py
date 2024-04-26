import cv2
import os

# Create a VideoCapture object for the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Background subtraction
background_subtractor = cv2.createBackgroundSubtractorMOG2()

# Directory to save the frames
output_directory = 'Z'
os.makedirs(output_directory, exist_ok=True)

frame_count = 0

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If frame is read correctly
    if ret:
        # Apply background subtraction
        fg_mask = background_subtractor.apply(frame)

        # Apply thresholding to segment hand region
        _, thresh = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected hands and save the frames
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Adjust this threshold based on your requirements
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Save the frame
                cv2.imwrite(os.path.join(output_directory, f'hand_frame_{frame_count}.jpg'), frame[y:y+h, x:x+w])
                frame_count += 1

        # Display the frame
        cv2.imshow('Hand Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Unable to read frame.")
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
