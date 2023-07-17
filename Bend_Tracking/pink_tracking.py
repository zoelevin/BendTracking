import cv2
import numpy as np

path = "C:\\Users\\zoeal\\OneDrive\\Documents\\pinkdots.png"

pinkdots = cv2.imread(path)

    
into_hsv =cv2.cvtColor(pinkdots,cv2.COLOR_BGR2HSV)
        # changing the color format from BGr to HSV 
        # This will be used to create the mask
L_limit=np.array([150,100,100]) # setting the pink lower limit
U_limit=np.array([180,255,255]) # setting the pink upper limit
         
  
b_mask=cv2.inRange(into_hsv,L_limit,U_limit)


        # creating the mask using inRange() function
        # this will produce an image where the color of the objects
        # falling in the range will turn white and rest will be black
pink_frame=cv2.bitwise_and(pinkdots,pinkdots,mask=b_mask)
gray = cv2.cvtColor(pinkdots, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 7)

#pink_frame = cv2.normalize(src=pink_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#set param1&2 lower for more relaxed (sensitive) circle detection
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=5, param2=20, minRadius=0, maxRadius=0)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
            #Draw the circle in the output image
        cv2.circle(pinkdots, (x, y), r, (0, 255, 0), 4)

            #Calculate the center of the circle
        center = (x, y)
        #circles[index][x or y coord]
        for i in range(len(circles)):
            for j in range(i+1, len(circles)):
                for k in range(i+2, len(circles)):
                    print(circles[i]) 
                        #Find the angle between two circles
                    #angle = np.arctan2(circles[j][1] - circles[i][1], circles[j][0] - circles[i][0])
                    #ba = a - b
                    #bc = c - b
                    # #Draw lines connecting the centers of the circles
                    #pt1 = (circles[i][0], circles[i][1])
                    #pt2 = (circles[j][0], circles[j][1])
                    #pt3 = (circles[k][0], circles[k][1])
                    #cv2.line(pink_frame, pt1, pt2, (0, 0, 255), 2)
                    #cv2.line(pink_frame, pt3, pt2, (0, 0, 255), 2)

                    # #Display the angle value
                    #cv2.putText(pink_frame, str(np.rad2deg(angle)), (circles[i][0], circles[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    #circles = np.uint16(np.around(circles))
    #print("circles: ", circles)
    #for i in circles[0, :]:
    #    center = (i[0], i[1])
    #    # circle center
    #    cv2.circle(pink_frame, center, 1, (0, 100, 100), 3)
    #    # circle outline
    #    radius = i[2]
    #    cv2.circle(pink_frame, center, radius, (0, 100, 100), 3)
# this will give the color to mask.
#cv2.imshow('Original',frame) # to display the original frame
        cv2.imshow('Pink Detector',pinkdots) # to display the blue object output

cv2.waitKey(0)
cv2.destroyAllWindows()

