import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
import tensorflow as tf

# Load the model

print("Test")


model = load_model('object_classification_model_no_bias.h5')

#Print input shape of layer 0


model.summary()
#print(model.layers[0].input_shape)


# Start the webcam
cap = cv2.VideoCapture(0)

    # Capture frame-by-frame



#Load Validation image
frame = cv2.imread('datasets/Validation/screwbolt/frame_0001.png')
#frame = cv2.imread('datasets/Validation/screw/frame_0000.png')
#frame = cv2.imread('datasets/Validation/screwbolt/frame_0000.png')



#6, (3, 3), activation='relu', input_shape=(48, 32, 1)
def conv2d(depthMul, kernel, activation, input_shape, weights, inputs):
    
    input_shape = np.array(input_shape)
    input_shape[0]=1
    
    output_shape = (1,input_shape[1] - kernel[0] + 1, input_shape[2] - kernel[1] + 1, weights.shape[3])
    
    


    #output = np.zeros(output_shape)
    
    inputs = inputs.reshape(input_shape)

    inputs=inputs.flatten()
    weights=weights.flatten()

    #print("Input shape: ", inputs.shape)
    #print("Output shape: ", output_shape)
    #print("Weights shape: ", weights.shape)

    outputX = output_shape[1]
    outputY = output_shape[2]
    inputX = input_shape[1]
    inputY = input_shape[2]
    outputDepth = output_shape[3]


    output = np.zeros(outputX*outputY*outputDepth)
    print("Output shape: ", output.shape)

    kernelI=kernel[0]
    kernelJ=kernel[1]
    inputDepth=input_shape[3]

    c=0


    for x in range(outputX):
        for y in range(outputY):
            for z in range(outputDepth):
                sum = 0
                c+=1
                for i in range(kernelI):
                    for j in range(kernelJ):
                        for k in range(inputDepth):
                            #print("x: ", x, " y: ", y, " d: ", z, " i: ", i, " j: ", j, " k: ", k)
                            #weight=weights[i][j][k][z]
                            #after flatten
                            weight = weights[\
                                i*(kernelJ*inputDepth*outputDepth) + \
                                j*(inputDepth*outputDepth) + \
                                k*(outputDepth) + \
                                z*(1) \
                            ]
                            
                            
                            
                            #inputTemp = inputs[0][x+i][y+j][k]
                            inputTemp = inputs[ \
                                #k*(inputDepth*inputX*inputY) + \
                                (x+i)*(inputDepth*inputY) + \
                                (y+j)*(inputDepth) + \
                                k*(1) \
                            ]
                            
                            print("x: ", x, " y: ", y, " z: ", z, " i: ", i, " j: ", j, " k: ", k, " weight: ", weight, " inputTemp: ", inputTemp)
                            
                            #][x+i][y+j][0]
                            sum += weight*inputTemp
                if activation == 'relu':
                    if sum < 0:
                        sum = 0
                #print(sum)

                print("x: ", x, " y: ", y, " z: ", z, " sum: ", sum)
                if c == 10:
                    return output
                #output[0][x][y][z] = sum
                output[\
                    x*(outputY*outputDepth) + \
                    y*(outputDepth) + \
                    z*(1) \
                ] = sum
    return output
    
    

def dense_layer(input_array, weights, bias=None, activation_function=None):
    # Matrix multiplication
    output = np.dot(input_array, weights)

    # Adding bias if provided
    if bias is not None:
        output += bias

    # Applying activation function if provided
    if activation_function is not None:
        output = activation_function(output)

    return output

def dense_layer2(input_array, weights, bias=None, activation_function=None):
    # Matrix multiplication



    output = np.dot(input_array, weights)

    #print(output.shape)
    #print(input_array.shape)
    #print(weights.shape)
    #print(bias.shape)

    weightsX = weights.shape[0]
    weightsY = weights.shape[1]
    #flatten weights
    weights = weights.flatten()

    #Zero output
    output = np.zeros(output.shape)

    
    for i in range(weightsX):
        for j in range(weightsY):
            output[0][j] += input_array[0][i] * weights[i*weightsY+j]


    # Adding bias if provided
    for i in range(len(output[0])):
        output[0][i] += bias[i]

    #Build in relu activation
    for i in range(len(output[0])):
        if output[0][i] < 0:
            output[0][i] = 0

    return output

def dense_layer3(input_array, weights, inputsNum, outputsNum):
   
    input_array = input_array.flatten()
    
    output = [0] * outputsNum

    # Matrix multiplication
    for i in range(inputsNum):
        for j in range(outputsNum):
            tempInput = input_array[i]
            
            tempWeight = weights[i*outputsNum+j]
            output[j] += tempInput * tempWeight
            change = tempInput * tempWeight
            
            #Print full calculation
            if(j==2):
                print("i: ", i, " j: ", j, " input: ", tempInput, " weight: ", tempWeight, " Change: ",change," output: ", output[j])


    # Adding bias
    #for i in range(outputsNum):
        #output[i] += bias[i]

    #Build in relu activation
    for i in range(outputsNum):
        if output[i] < 0:
            output[i] = 0

    return output

# Example usage


def dense_layer4(input_array, weights, bias, weightsX, weightsY):
    y = len(weights[0])  # Number of outputs
    output = [0] * y

    print(bias)

    # Matrix multiplication using for loops
    for i in range(y):
        for j in range(len(input_array)):
            output[i] += input_array[j] * weights[j][i]

    # Adding bias if provided
    if bias is not None:
        for i in range(y):
            output[i] += bias[i]

    # Applying activation function if provided
    if activation_function is not None:
        for i in range(y):
            output[i] = activation_function(output[i])

    return output

# You can define your own activation function
def relu(x):
    return np.maximum(0, x)



import numpy as np

def maxPooling2D(input_feature_map, pool_size, strides):
    
    
    stride = strides[0]
    pool = pool_size[0]
    
    input_feature_map=input_feature_map[0]
    # Calculate the dimensions of the output feature map
    output_height = (input_feature_map.shape[0] - pool_size[0]) // strides[0] + 1
    output_width = (input_feature_map.shape[1] - pool_size[1]) // strides[1] + 1
    output_depth = input_feature_map.shape[2]

    # Initialize the output feature map with zeros
    output_feature_map = np.zeros((output_height, output_width, output_depth))

    # Apply the max pooling operation
    for h in range(output_height):
        for w in range(output_width):
            for d in range(output_depth):
                h_start = h * stride
                h_end = h_start + 1
                w_start = w * stride
                w_end = w_start + 1

                # Extract the current window and perform max pooling
                window = input_feature_map[h_start:h_end, w_start:w_end, d]
                
                
                output_feature_map[h, w, d] = np.max(window)

    return output_feature_map


def maxPooling2D_2(input_feature_map, input_height, input_width, depth):
    # Assume the pool size and strides are both 2x2
    pool_size = 2
    stride = 2
    input_feature_map=input_feature_map[0]
    
    #flatten
    input_feature_map=input_feature_map.flatten()
    

    # Calculate the dimensions of the output feature map
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1

    # Initialize the output feature map with zeros
    output_feature_map = np.zeros(output_height * output_width * depth)
    

    # Apply the max pooling operation
    for h in range(output_height):
        for w in range(output_width):
            for d in range(depth):
                max_val = -np.inf

                for i in range(pool_size):
                    for j in range(pool_size):
                        # Expanded input index calculation
                        input_index = (\
                            (h * stride + i) * input_width * depth + \
                            (w * stride + j) * depth + \
                            d * 1 \
                        )
                        max_val = max(max_val, input_feature_map[input_index])

                # Expanded output index calculation
                output_index = (\
                    h * output_width * depth + \
                    w * depth + \
                    d * 1 \
                )
                output_feature_map[output_index] = max_val

    return output_feature_map

# Example usage




# Convert to grayscale if your model expects grayscale inputs
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Resize the frame to your model's expected input size
resized_frame = cv2.resize(gray, (32, 48))

# Reshape and normalize the image
img_array = np.expand_dims(resized_frame, axis=0)
img_array = img_array / 255.0




# Make a prediction
prediction = model.predict(img_array)



# Suppose 'model' is your pre-trained model
# And you want to get the output of a layer named 'layer_name'

# Create a new model that will output both the original model's output
# and the desired layer's output


# Assuming 'model' is your pre-trained model

# Get the output of the first layer

LayerTointercept = 7
first_layer_output = model.layers[0].output
#Print name
#print("Intercepting output at: ", model.layers[LayerTointercept].name)
print("Replacing Layer: ", model.layers[LayerTointercept+1].name)

# Create a new model to output the first layer's output
if(LayerTointercept>=0):
    interceptedLayerOutput = model.layers[LayerTointercept].output
    interceptedModel = Model(inputs=model.input, outputs=interceptedLayerOutput)
    interceptedOutput=interceptedModel.predict(img_array)

replacedLayerOutput = model.layers[LayerTointercept+1].output
replacedModel = Model(inputs=model.input, outputs=replacedLayerOutput)

# Use this model to get the output of the first layer

replacedModelOriginalOutput = replacedModel.predict(img_array)




replacedWeights = model.layers[LayerTointercept+1].weights
#replacedBias = model.layers[LayerTointercept+1].bias
#print("Weights shape: ", replacedWeights)
#print("Bias shape: ", replacedBias)
#print(replacedBias)
#dense_layer_output = dense_layer(interceptedOutput, replacedWeights[0].numpy(), replacedBias.numpy(), relu)




#Flatten weights and bias


#flat_replacedBias = replacedBias.numpy().flatten()
#flat_replacedWeights = replacedWeights[0].numpy().flatten()
#print("flat_replacedBias: ", flat_replacedBias)
#print("flat_replacedWeights: ", flat_replacedWeights)

#testInput=(0.000000, 0.063876, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.514445, 5.231051, 0.139151, 0.266782, 0.000000, 0.000000, 0.000000, 0.455701, 0.000000, 5.056775, 0.000000, 1.827464, 0.197882, 0.000000, 0.967189, 0.000000, 0.011146, 0.076450, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.378570, 5.280768, 0.229329, 0.272958, 0.000000, 0.000000, 0.000000, 0.376497, 0.000000, 5.108150, 0.000000, 1.761386, 0.107411, 0.000000, 0.896981, 0.000000, 0.000000, 0.097451, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.522969, 5.407016, 0.148771, 0.394866, 0.000000, 0.000000, 0.000000, 0.438590, 0.000000, 5.216367, 0.000000, 1.920154, 0.305575, 0.000000, 1.049272, 0.000000, 0.000000, 0.117126, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.459370, 5.494470, 0.218159, 0.382569, 0.000000, 0.000000, 0.000000, 0.423632, 0.000000, 5.248447, 0.000000, 1.913488, 0.312632, 0.000000, 1.064000, 0.000000, 0.000000, 0.741888, 0.000000, 1.258027, 0.000000, 0.000000, 0.000000, 0.756028, 2.196458, 6.345436, 0.000000, 1.033164, 0.000000, 0.000000, 0.229057, 1.953521, 0.083761, 6.015786, 0.000000, 3.025967, 1.650896, 0.000000, 2.907727, 0.000000, 0.129899, 0.560593, 0.000000, 0.645759, 0.000000, 0.000000, 0.000000, 0.180706, 1.544242, 6.527525, 0.000000, 0.878543, 0.000000, 0.000000, 0.034758, 1.490310, 0.253422, 6.215171, 0.000000, 2.520172, 0.978777, 0.000000, 2.213440, 0.000000, 0.000000, 1.226339, 0.000000, 2.483769, 0.000000, 0.000000, 0.000000, 0.844062, 3.810170, 6.510714, 0.000000, 1.794796, 0.000000, 0.000000, 1.153715, 3.298021, 1.145185, 5.813734, 0.000000, 3.461133, 2.312854, 0.000000, 4.080119, 0.000000, 0.238911, 1.208395, 0.000000, 2.166405, 0.000000, 0.000000, 0.000000, 0.520312, 2.905704, 6.662202, 0.000000, 1.867244, 0.000000, 0.000000, 0.944764, 2.595864, 0.951577, 6.258826, 0.000000, 2.892624, 1.880361, 0.000000, 3.641609, 0.000000)

weightsX = replacedWeights[0].numpy().shape[0]
weightsY = replacedWeights[0].numpy().shape[1]
flat_replacedWeights = replacedWeights[0].numpy().flatten()
recreated_output = dense_layer3(interceptedOutput, flat_replacedWeights, weightsX, weightsY)
#recreated_output = conv2d(6, (3, 3), 'relu', (1, 48, 32, 1), model.layers[0].weights[0].numpy(), img_array)

#layer input shape
#recreated_output = conv2d(6, (3, 3), 'relu', model.layers[LayerTointercept+1].input_shape, model.layers[0].weights[0].numpy(), interceptedOutput)

#recreated_output = maxPooling2D_2(interceptedOutput, 46,30,6)


#flatten both
replacedModelOriginalOutput = replacedModelOriginalOutput.flatten()
recreated_output = np.array(recreated_output).flatten()

#Print layer weights
print("=====================================")
print("Weights: ")
#set numpy print to decimals
np.set_printoptions(formatter={'float': '{:0.8f}'.format})
print(model.layers[LayerTointercept+1].weights[0].numpy())

# show first 10 values of original and recreated output side by side
print("=====================================")
for i in range(0, 3):
    original = format(str(replacedModelOriginalOutput[i]), '<20')  # Adjust '10' to change the width
    recreated = format(str(recreated_output[i]), '<20')  # Same width as 'original'
    #recreated2 = format(dense_layer_output2[i], '<20')  # Same width as 'original'
    print("Original: ", original, "Recreated: ", recreated)

#print(dense_layer_output)
#print(replacedModelOriginalOutput)

exit()

#Reshape image to 1x48x32x1
#img_array = np.reshape(img_array, (1,48, 32, 1))
#output=conv2d(6, (3, 3), 'relu', (1, 48, 32, 1), model.layers[0].weights[0].numpy(), img_array)

#print(model.layers[0].weights[0].numpy())
#print(output_of_first_layer)
#print(output)

#Flatten both arrays
#output_of_first_layer = output_of_first_layer.flatten()
#output = output.flatten()

#Save both outputs to file
#np.savetxt("output_of_first_layer_keras.txt", output_of_first_layer, fmt="%s")
#np.savetxt("output_of_first_layer_recreate.txt", output, fmt="%s")







# Determine the predicted class
#predicted_class = np.argmax(prediction, axis=1)

# Optional: Map the class index to a class name if you have named classes
#class_names = ['Class1', 'Class2', 'Class3']  # Update with your class names
#predicted_class_name = class_names[predicted_class[0]]

# Display the prediction on the frame
#cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Display the resulting frame





while(True):
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
