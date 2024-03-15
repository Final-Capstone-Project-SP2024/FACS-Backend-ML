import cv2
import datetime as dt
import time
import threading

# Function to read frames, display on screen, and record video from a camera
def record_video(camera_index, stop_event):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Failed to open camera {camera_index}")
        return

    # Get the frame size from the camera
    size = (int(cap.get(3)), int(cap.get(4)))

    # Initialize variables for recording
    record_time = 10  # Record video every 10 seconds
    record_number = 1  # Initial record number

    while not stop_event.is_set():
        start_time = time.time()
        timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"camera_{camera_index}_{timestamp}_record_{record_number}.mp4"
        result = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, size)

        while time.time() - start_time < record_time and not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to capture frame from camera {camera_index}")
                break

            cv2.imshow(f"Camera {camera_index}", frame)
            result.write(frame)

            # Press 'q' to stop recording and exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        result.release()
        record_number += 1

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Create stop events for each thread
stop_event_internal = threading.Event()
stop_event_external = threading.Event()

# Create threads for each camera
thread_internal = threading.Thread(target=record_video, args=(0, stop_event_internal))
thread_external = threading.Thread(target=record_video, args=(1, stop_event_external))

# Start threads
thread_internal.start()
thread_external.start()

# Let the threads run indefinitely  
try:
    while True:
        time.sleep(0)
except KeyboardInterrupt:
    # If you want to stop the threads manually by pressing Ctrl+C
    stop_event_internal.set()
    stop_event_external.set()   
    thread_internal.join()
    thread_external.join()

print("Recording stopped for both cameras.")
