
import numpy as np
import cv2

def process_image(input_path, output_path, new_size):
    # Load the image
    frame = cv2.imread(input_path)




    # Convert to grayscale if your model expects grayscale inputs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the frame to your model's expected input size
    resized_frame = cv2.resize(gray, (32, 48))

    # Reshape and normalize the image
    img_array = np.expand_dims(resized_frame, axis=0)
    img_array = img_array / 255.0
    
    #Flatten the array
    img_array = img_array.flatten()
    
    #save to file
    file = open(output_path, "w")
    
    file.write("float input["+str(len(img_array))+"] = {") 
    for i in range(len(img_array)):
        file.write(str(img_array[i]))
        if i != len(img_array)-1:
            file.write(", ")
    file.write("};\n\n")
    

# Example usage

imagePath = 'datasets/Validation/screw/frame_0001.png'

process_image(imagePath, 'output_file.txt', (48, 32))  # Adjust new_size as needed
