import cv2
import numpy as np

# Define the dimensions for the video feed
frameWidth = 480
frameHeight = 360

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

# Define color ranges in HSV for detection
myColors = [
    [5, 107, 0, 19, 255, 255],   # Example color 1
    [133, 56, 0, 159, 156, 255], # Example color 2
    [57, 76, 0, 100, 255, 255],  # Example color 3
    [0, 0, 0, 180, 255, 50],     # Black
    [100, 150, 0, 140, 255, 255] # Blue
]

# Define BGR values for visualization
myColorsValues = [
    [51, 153, 255], # Example color 1 (BGR)
    [255, 0, 255],  # Example color 2 (Magenta in BGR)
    [0, 255, 0],    # Example color 3 (Green in BGR)
    [0, 0, 0],      # Black in BGR
    [255, 0, 0]     # Blue in BGR
]

# List to store points of detected colors
myPoints = []  # Changed to store points with x, y, colorID

def findColor(img, myColors, myColorsValues):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        cv2.circle(imgResult, (x, y), 10, myColorsValues[count], cv2.FILLED)
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        count += 1
    
    return newPoints

def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = 0, 0, 0, 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            cv2.drawContours(imgResult, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y

def drawOnCanvas(myPoints, myColorsValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorsValues[point[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgResult = img.copy()
    newPoints = findColor(img, myColors, myColorsValues)
    
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorsValues)
    
    cv2.imshow("Video", img)
    cv2.imshow("Result", imgResult)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
