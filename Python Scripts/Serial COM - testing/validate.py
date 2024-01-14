

import os
import serial
import numpy as np
import time
import cv2
import struct


ser = serial.Serial('/dev/ttyUSB1', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)





def testImage(img_array):
    
    
    img=img_array.flatten()
    #print(len(img))
    for j in range(len(img)):
        var=int(img[j])

        ser.write(var.to_bytes(1,'little',signed=False))  # send bytearray over UART
        time.sleep(0.001)
    
    data = ser.read(4)

    # Convert bytes to an integer (assuming little-endian format)
    int_value = struct.unpack('<I', data)[0]
    print("Received: " + str(int_value))
    return int_value
    
    #if "output" in nn_res:
    #    print("Received 'output'. Sending data again...")

def process_image(imagePath, new_size):
    frame = cv2.imread(imagePath)
    # Convert to grayscale if your model expects grayscale inputs
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize the frame to your model's expected input size
    resized_frame = cv2.resize(gray, (32, 48))
    # Reshape and normalize the image
    img_array = np.expand_dims(resized_frame, axis=0)
    print(imagePath)
    result = testImage(img_array)
    
    return result
    
    


#
validateFolder = ["datasets/Validation/bolt", "datasets/Validation/screw", "datasets/Validation/screwbolt"]

#Load x images from each folder

results = [[],[],[]]

i=0

for folder in validateFolder:
    #print(folder)
    c=0
    for filename in os.listdir(folder):
        
        #print(filename)
        imagePath = folder+"/"+filename
        #print(imagePath)
        result = process_image(imagePath, (48, 32))  # Adjust new_size as ne
        results[i].append(result)
        c+=1

    i+=1


#print and save results

print(results)
#Print occurrences of each result

for i in range(len(results)):
    #count occurrences of each result
    print("Results for folder: "+validateFolder[i])
    
    unique, counts = np.unique(results[i], return_counts=True)
    
    for j in range(len(unique)):
        print(str(unique[j])+" : "+str(counts[j]))
    

    
    

np.savetxt("results.txt", np.array(results), fmt="%s")

        

