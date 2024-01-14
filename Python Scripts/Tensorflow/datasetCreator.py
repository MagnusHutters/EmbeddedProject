import cv2
import time
import os

# Define the name of the classification object
object_name = input("Enter the name of the classification object: ")

# Directories for training and validation datasets
train_dir = f'datasets/Training/{object_name}'
validation_dir = f'datasets/Validation/{object_name}'

# Create directories if they don't exist
for directory in [train_dir, validation_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Start the webcam capture
cap = cv2.VideoCapture(0)

# Set the frame rate to 10 fps
fps = 30
frame_interval = 1 / fps

# Initialize the frame count
frame_count = 0
validation_count = 0
training_count = 0

while training_count < 2000 or validation_count < 200:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Determine if the current frame is for training or validation
    if frame_count % 10 == 0 and validation_count < 200:
        # Save frame in the validation folder
        frame_filename = f'{validation_dir}/frame_{validation_count:04d}.png'
        validation_count += 1
    else:
        # Save frame in the training folder
        frame_filename = f'{train_dir}/frame_{training_count:04d}.png'
        training_count += 1

    # Save the frame
    cv2.imwrite(frame_filename, frame)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Increment frame count
    frame_count += 1

    # Wait to maintain the desired frame rate
    time_elapsed = time.time() - start_time
    if time_elapsed < frame_interval:
        time.sleep(frame_interval - time_elapsed)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
