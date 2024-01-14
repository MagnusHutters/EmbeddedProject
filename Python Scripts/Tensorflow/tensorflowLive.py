import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model('object_classification_model.h5')

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale if your model expects grayscale inputs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to your model's expected input size
    resized_frame = cv2.resize(gray, (48, 32))
    

    # Reshape and normalize the image
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = img_array / 255.0

    # Make a prediction
    prediction = model.predict(img_array)

    # Determine the predicted class
    predicted_class = np.argmax(prediction, axis=1)

    # Optional: Map the class index to a class name if you have named classes
    class_names = ['Class1', 'Class2', 'Class3']  # Update with your class names
    predicted_class_name = class_names[predicted_class[0]]

    # Display the prediction on the frame
    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
