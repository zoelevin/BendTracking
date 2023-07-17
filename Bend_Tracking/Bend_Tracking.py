from pathlib import Path
from telnetlib import NOP
import cv2
import numpy as np
from pyfirmata import Arduino
import matplotlib.pyplot as plt

path = "C:\\Users\\zoeal\\OneDrive\\Documents\\pinkdots.png"
vid = cv2.VideoCapture("C:\\Users\\zoeal\\OneDrive\\Documents\\MiniTool Video Converter\\Screen Record\\PinkTrackingv3.mp4")
board = Arduino('COM4')

#iterator thread
it = Arduino.util.Iterator(board)
it.start()
while(True):
      

    ret, pinkdots = vid.read()
    
    if ret == False:
        print("frame is empty")

    else:
        into_hsv =cv2.cvtColor(pinkdots,cv2.COLOR_BGR2HSV)
        # changing the color format from BGR to HSV 
        # this is used to create the pink mask
        L_limit=np.array([150,61,205]) # setting the pink lower limit
        U_limit=np.array([190,101,305]) # setting the pink upper limit
         
  
        b_mask=cv2.inRange(into_hsv,L_limit,U_limit)


        # creating the mask using inRange() function
        # this will produce an image where the color of the objects
        # falling in the range will turn white and rest will be black
        pinkdots = cv2.bitwise_and(pinkdots,pinkdots,mask=b_mask)
        
        gray = cv2.cvtColor(pinkdots, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)

        #set param1&2 lower for more relaxed (sensitive) circle detection
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=10, param2=15, minRadius=0, maxRadius=0)


       


        if circles is not None:
            analog_0 = board.get_pin('a:0:i')
            analog_0.read()
             ##order by x value
            circles[0]=circles[0][np.argsort(circles[0][:,0])]
            circles = np.round(circles[0, :]).astype("int")

            for (x, y, r) in circles:
        

                #Draw the circle in the output image
                cv2.circle(pinkdots, (x, y), r, (0, 255, 0), 4)
         
                center = (x, y)
                if (len(circles)==3):
                    pt1 = np.array([circles[0][0], circles[0][1]])
                    pt2 = np.array([circles[1][0], circles[1][1]])
                    pt3 = np.array([circles[2][0], circles[2][1]])
                    ba = pt1-pt2
                    bc = pt3-pt2
                    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                    angle = np.arccos(cosine_angle)
                    angle = np.degrees(angle)
                    cv2.line(pinkdots, pt2, pt1, (0, 0, 255), 2)
                    cv2.line(pinkdots, pt3, pt2, (0, 0, 255), 2)
                    cv2.putText(pinkdots, str(np.round(angle,2)), (50,50),cv2.FONT_HERSHEY_SIMPLEX, 
                               1, (100,255,100), 2)
                    with open('PinkTrackingResults.csv','wb') as file:
                        file.write()
                

        cv2.imshow('Pink Detector',pinkdots)# to display the pink output
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
vid.release()
cv2.waitKey(0)
cv2.destroyAllWindows()