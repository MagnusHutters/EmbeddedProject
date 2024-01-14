import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow as tf

# Load the model

#print("Test")


model = load_model('object_classification_model_no_bias.h5')

#Print input shape of layer 0


#model.summary()
#print(model.layers[0].input_shape)
#exit()


# Start the webcam
cap = cv2.VideoCapture(0)

    # Capture frame-by-frame



#Load Validation image
#frame = cv2.imread('datasets/Validation/bolt/frame_0001.png')
#frame = cv2.imread('datasets/Validation/screw/frame_0001.png')
frame = cv2.imread('datasets/Validation/screwbolt/frame_0001.png')




# Convert to grayscale if your model expects grayscale inputs
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Resize the frame to your model's expected input size
resized_frame = cv2.resize(gray, (32, 48))

# Reshape and normalize the image
img_array = np.expand_dims(resized_frame, axis=0)
img_array = img_array / 255.0

# Make a prediction
prediction = model.predict(img_array)

# Print the prediction as decimals - not scientific notation

np.set_printoptions(formatter={'float': '{:0.8f}'.format})

# Print the array
#print(prediction)

#exit()

# Suppose 'model' is your pre-trained model
# And you want to get the output of a layer named 'layer_name'

# Create a new model that will output both the original model's output
# and the desired layer's output


# Assuming 'model' is your pre-trained model

# Get the output of the first layer
layer=8
first_layer_output = model.layers[layer].output
#Print name
print("Layer name: ", model.layers[layer].name)

# Create a new model to output the first layer's output
intermediate_model = Model(inputs=model.input, outputs=first_layer_output)

# Use this model to get the output of the first layer
output_of_first_layer = intermediate_model.predict(img_array)



# flatten and print 10 values from the first layer

output_of_first_layer_flat = output_of_first_layer.flatten()
flat_img_array = img_array.flatten()
for i in range(len(output_of_first_layer_flat)):

print("=====================================")
for i in range(0, 3):
    print(output_of_first_layer_flat[i])
    #print(flat_img_array[i])
print("=====================================")


print(prediction)
exit()



# Determine the predicted class
predicted_class = np.argmax(prediction, axis=1)

# Optional: Map the class index to a class name if you have named classes
class_names = ['Class1', 'Class2', 'Class3']  # Update with your class names
predicted_class_name = class_names[predicted_class[0]]

# Display the prediction on the frame
cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Display the resulting frame





while(True):
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
