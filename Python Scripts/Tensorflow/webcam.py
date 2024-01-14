import cv2
import serial

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize serial connection (Adjust port and baudrate as per your configuration)
ser = serial.Serial('COM3', 115200)  # Example: higher baud rate

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Scale down the frame
    # Adjust the size as needed for your application
    scaled_frame = cv2.resize(gray_frame, (160, 120))  

    # Serialize the frame (simple conversion to bytes)
    serialized_frame = scaled_frame.tobytes()

    # Send over serial
    ser.write(serialized_frame)

    # Display the resulting frame (optional, for your reference)
    cv2.imshow('Grayscale and Scaled', scaled_frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
ser.close()
