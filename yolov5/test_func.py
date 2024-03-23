# Python program to continuously capture video from webcam until 'q' is pressed to quit

# Import the necessary libraries
import cv2


# Open the webcam
cap = cv2.VideoCapture(1)

# Check if the webcam is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('continuous_video.avi', fourcc, 20.0, (640, 480))

# Loop to continuously capture video
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if the frame is read successfully
    if not ret:
        print("Error: Could not read frame")
        break

    # Write the frame to the output video
    out.write(frame)

    # Display the captured frame
    cv2.imshow('frame', frame)

    # Wait for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and output video
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
