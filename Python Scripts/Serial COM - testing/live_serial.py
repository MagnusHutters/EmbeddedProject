import serial
import os
import random
import cv2
import numpy as np
import struct
import time
import sys

def main():
    dims = (48, 32)  # dimensions of images to train/test with
    cap = cv2.VideoCapture(4)
    
    # define serial connection
    ser = serial.Serial('/dev/ttyUSB1', 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

    while True:  # Continuous loop to handle subsequent communication
        ret, frame = cap.read()
        cv2.imwrite("test.png", frame)
        img = cv2.imread("test.png",0)  # read img as grayscale
        img = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)  # resize img to fit dims
        cv2.imwrite("test2.png",img)
        img = np.asarray(img)

        #print(img)
        # Send data
        img=img.flatten()
        print(len(img))
        for j in range(len(img)):
            var=int(img[j])

            ser.write(var.to_bytes(1,'little',signed=False))  # send bytearray over UART
            time.sleep(0.001)
        
        print("Data sent. Waiting for response...")
        while(not ser.in_waiting):
            pass
        # Read response
        while(ser.in_waiting): # decode received bytes
            nn_res=ser.read(1)
            print(nn_res)  # display prints from U96

        #if "output" in nn_res:
        #    print("Received 'output'. Sending data again...")

if __name__ == "__main__":
    main()
