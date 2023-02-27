#1. Preprocessing - input image, closing, noise removal, converting to gray
#2. Find canny edges, create kernel (30, 1), cause signatures have more width and less height, 
#   apply closing using the created kernel
#3. Get the contours(external), sort them based on area in ascending order
#4. Go through each contour, check the width and heights of their bounding rectangle
#5. If met the condition, draw the rectangle.



import cv2
from PIL import Image
import numpy as np
# from skimage import morphology, measure
import matplotlib.pyplot as plt

#Resize input image to a desired size
img = Image.open("./pan/new.jpg")
img_dup = np.array(img)
img = img.resize((800, 500))
#crop image to get the area containing signature
box = (100, 350, 580, 480)
crop_image = img.crop(box=box)
img = np.array(img)
img = np.array(crop_image)
img_cpy = img
#now we have the image cropped and
# img = cv2.dilate(img, (21,21))
# img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)[1]
img = cv2.fastNlMeansDenoisingColored(img_cpy, None, 10, 10, 7, 21)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7)
# img = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# img = cv2.erode(img, (9,9), iterations=5)
# cv2.imshow("thresh", img)
#Now we can perform CCA
#connected disconnected components
#find canny edges
edged = cv2.Canny(img, 30, 200)
shape = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, shape)
#finding contours
contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours = sorted(contours, key= lambda x: cv2.contourArea(x))

try: hierarchy = hierarchy[0]
except: hierarchy = []
# height, width = edged.shape
# min_x, min_y = width, height
# max_x = max_y = 0
#compute the bounding box for the contour, and draws it on the image
flag = False
for contour, hier in zip(contours, hierarchy):
    (x,y,w,h) = cv2.boundingRect(contour)
    if(y < 5):
        continue
    
    # peri = cv2.arcLength(contour, True)
    # approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
    # print('number of sides:',len(approx))
    

    # min_x, max_x = min(x, min_x), min(x+w, max_x)
    # min_y, max_y = min(y, min_y), max(y+h, max_y)
    if (w > 120 and w < 195 and h < 40 and h > 25) or (w > 80 and  h > 40):
        rect = cv2.rectangle(img_cpy, (x,y), (x+w, y+h), (255,0,0), 2)
        flag = True
        print(f"Signature found at : ({x}, {y}) ({x+w}, {y+h})")
        # print(len(approx))
        break

#Now look for signatures with seperations(two small signatures)
if not flag:
    #perform cca
    gray = cv2.cvtColor(img_cpy, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    analysis = cv2.connectedComponentsWithStats(threshold, 8, cv2.CV_32S)
    (total_labels, label_ids, values, centroid) = analysis
    #loop over number of unique cc labels
    a, b = 0, 0
    c, d = 0, 0
    for i in range(0, total_labels):
        if i == 0:
            text = "examining component {}/{} (background)".format(i+1, total_labels)
        #otherwise we are examining an actual connected component
        else:
            text = "examining component {}/{}".format(i+1, total_labels)
        print("[INFO] {}".format(text))

        #extract the connected componenent statistics and centroid for the current label
        x = values[i, cv2.CC_STAT_LEFT]
        y = values[i, cv2.CC_STAT_TOP]
        w = values[i, cv2.CC_STAT_WIDTH]
        h = values[i, cv2.CC_STAT_HEIGHT]
        x2 = x+w
        y2 = y+w
        if(w > 30 and w < 60 and h < 50):
            a, b = x, x2
            c, d = y, y2
    # print(f"({min_x}, {min_y}), ({max_x}, {max_y})")
    # rect = cv2.rectangle(img_cpy, (min_x+200,min_y+20), (max_x, max_y-20), (255,0,0), 2)
    rect = cv2.rectangle(img_cpy, (a+100,b-20), (c,d-30), (255, 0, 0), 2)



print("Number of contours found: "+str(len(contours)))
# cv2.drawContours(img_cpy, contours, -1, color=(0,255,0), thickness=1, lineType=cv2.LINE_AA)
# cv2.imshow("Canny edges after contouring", edged)
cv2.imshow("Signature",img_cpy)
# plt.imshow(img_cpy, "gray")
# plt.show()
cv2.imshow("Input Image",img_dup)
cv2.waitKey(0)