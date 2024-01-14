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



#Export Weights - layer by layer



	## Retrieve network weights after training. Skip layer 0 (input layer)
for w in range(0, len(model.layers)):
    print("=====================================")
    print("Layer: ", w)
    print("Layer name: ", model.layers[w].name)
    #Input shape of layer
    print("Input shape: ", model.layers[w].input_shape)
    #Output shape of layer
    print("Output shape: ", model.layers[w].output_shape)

    #shape of weights
    if len(model.layers[w].weights) > 0:

        print("Weights shape: ", model.layers[w].weights[0].numpy().shape)
    else:
        print("No weights")
    
    


#Save weights to file as c arrays
layersToExport = (0, 1, 2, 3, 4, 5, 6, 7, 8)

#Filename
filename = "weights.h"
open(filename, 'w').close() # clear file
file = open(filename,"a")


#Loop through layers - save parameters to file

def saveVariableToFile(file,varPrefix,varName, varValue):
    
    name = varPrefix + varName
    #Make CAPS
    name = name.upper()
    file.write("#define " + name + " ")
    file.write(str(varValue))
    file.write("\n")
    return
    
    #check if variable is int
    if isinstance(varValue, int):
        file.write("int " + varPrefix + varName + " = " + str(varValue) + ";")
    elif isinstance(varValue, float):
        file.write("float " + varPrefix + varName + " = " + str(varValue) + ";")
    else:
        print("Error: Variable type not supported")
        #type
        
        print("Variable name: ", varValue)


for w in layersToExport:
    layer = model.layers[w]

    dataNamePrefix = f"layer{w}_"

    #Save input shape
    
    #Total input size
    saveVariableToFile(file, dataNamePrefix, "totalInputSize", int(np.prod(layer.input_shape[1:])))
    #total output size
    saveVariableToFile(file, dataNamePrefix, "totalOutputSize", int(np.prod(layer.output_shape[1:])))
    
    
    #check if input shape is 4D
    if len(layer.input_shape) == 2:
        saveVariableToFile(file, dataNamePrefix, "inputSize", layer.input_shape[1])
    elif len(layer.input_shape) == 3:
        saveVariableToFile(file, dataNamePrefix, "inputX", layer.input_shape[1])
        saveVariableToFile(file, dataNamePrefix, "inputY", layer.input_shape[2])
    elif len(layer.input_shape) == 4:
        saveVariableToFile(file, dataNamePrefix, "inputX", layer.input_shape[1])
        saveVariableToFile(file, dataNamePrefix, "inputY", layer.input_shape[2])
        saveVariableToFile(file, dataNamePrefix, "inputDepth", layer.input_shape[3])

    #Save output shape
    #check dimensionality of output shape
    if len(layer.output_shape) == 2:
        saveVariableToFile(file, dataNamePrefix, "outputSize", layer.output_shape[1])
    elif len(layer.output_shape) == 3:
        saveVariableToFile(file, dataNamePrefix, "outputX", layer.output_shape[1])
        saveVariableToFile(file, dataNamePrefix, "outputY", layer.output_shape[2])
    elif len(layer.output_shape) == 4:
        saveVariableToFile(file, dataNamePrefix, "outputX", layer.output_shape[1])
        saveVariableToFile(file, dataNamePrefix, "outputY", layer.output_shape[2])
        saveVariableToFile(file, dataNamePrefix, "outputDepth", layer.output_shape[3])
    
    # if layer type is convolutional
    if isinstance(layer, tf.keras.layers.Conv2D):
        #Save kernel size
        saveVariableToFile(file, dataNamePrefix, "kernelSizeX", layer.kernel_size[0])
        saveVariableToFile(file, dataNamePrefix, "kernelSizeY", layer.kernel_size[1])
        #Save stride
        saveVariableToFile(file, dataNamePrefix, "strideX", layer.strides[0])
        saveVariableToFile(file, dataNamePrefix, "strideY", layer.strides[1])
        #Save use_bias
        #Save name
    file.write("\n\n")
    



#Loop through layers - save weights to file
for w in layersToExport:
    #layer name
    print("=====================================")
    print("Layer: ", w)
    print("Layer name: ", model.layers[w].name)


    #Add weight delcaration to file
    #If has weights
    if len(model.layers[w].weights) > 0:
    
        dataNamePrefix = f"layer{w}_"
        varName = dataNamePrefix+ "weights"
        
        file.write("float " + varName + "[] = {")
        #Flatten weights
        
        weights = model.layers[w].weights[0].numpy().flatten()
        #Save weights to file
        for i in range(len(weights)):
            file.write(str(weights[i]))
            if i != len(weights)-1:
                file.write(", ")
        #Add end of array
        file.write("};\n\n")


    




    

file.close()