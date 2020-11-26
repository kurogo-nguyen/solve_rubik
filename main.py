import cv2 as cv
import numpy as np
from imutils import contours, resize
from glob import iglob 

colors = {
    'o': ([4, 50, 90], [18, 255, 255]),     # Orange v
    'g': ([50, 90, 60], [80, 255, 255]),        # Green v
    'r': ([151, 100, 60], [180, 255, 255]),        # red
    'R': ([0, 100, 60], [3, 255, 255]),        # red
    'b': ([80, 90, 70], [150, 255, 255]),    # Blue v
    'y': ([21, 100, 600], [41, 255, 255]),   # Yellow v 
    'w':([0, 0, 100], [180, 70, 255]),        #White v
    }

# Color threshold to find the squares

template = cv.imread('template.png')
# template_ = cv.cvtColor(template, cv.COLOR_BGR2GRAY) 
template_canny = cv.Canny(template, 80, 100)
tH, tW, _ = template.shape
cv.imshow("Template", template_canny)
cube = ''
for filepath in iglob(r'test/3/rub*.jpg', recursive=True):
    #load image
    image = cv.imread(filepath)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
        # resize the image
        resized = resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break
        # detect edges then apply template matching to find the template in the image
        resized = cv.GaussianBlur(resized, (5,5), 0)
        edged = cv.Canny(resized, 20, 50)
        result = cv.matchTemplate(edged, template_canny, cv.TM_CCOEFF)
        cv.imshow('edge', edged)
        (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
        # if we have found a new maximum correlation value, then update the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found
    denta = 00
    (startX, startY) = (int(maxLoc[0] * r + denta), int(maxLoc[1] * r + denta))
    (endX, endY) = (int((maxLoc[0] + tW) * r - denta), int((maxLoc[1] + tH) * r - denta))

    image_cube = resize(image[startY:endY, startX:endX], width= 200)
    

    cv.imshow('crop',image_cube)
    # template = cv.imread('template.png')
    template = resize(template, image_cube.shape[0])
    # I want to put logo on top-left corner, So I create a ROI
    rows,cols,channels = template.shape
    roi = image_cube[0:rows, 0:cols]
    # Now create a mask of logo and create its inverse mask also
    img2gray = cv.cvtColor(template,cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
    mask_inv = cv.bitwise_not(mask)
    # Now black-out the area of logo in ROI
    img1_bg = cv.bitwise_and(roi,roi,mask = mask_inv)
    # Take only region of logo from logo image.
    img2_fg = cv.bitwise_and(template,template,mask = mask)
    # Put logo in ROI and modify the main image
    dst = cv.add(img1_bg,img2_fg)
    image_cube = dst

    cv.imshow('res',image_cube)
    cv.waitKey(0)
    cv.destroyAllWindows()





    # image = cv.resize(image, (image.shape[1]%400, image.shape[0]%400))
    # open_kernel = cv.getStructuringElement(cv.MORPH_RECT, (image_cube.shape[1]//20,image_cube.shape[0]//20))
    close_kernel = cv.getStructuringElement(cv.MORPH_CROSS, (image_cube.shape[1]//100,image_cube.shape[0]//100))
    original = image_cube.copy()
    cube_hsv = cv.cvtColor(image_cube, cv.COLOR_BGR2HSV)
    # cube_hsv = cv.GaussianBlur(cube_hsv,(3,3),0)
    # cube_hsv = cv.addWeighted(cube_hsv, 1.7, cube_hsv_blur, -0.5, 0)
    mask = np.zeros(cube_hsv.shape, dtype=np.uint8)
    for color, (lower, upper) in colors.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8) 
        
        color_mask = cv.inRange(cube_hsv, lower, upper)

        # color_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, open_kernel, iterations=2)
        color_mask = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, close_kernel, iterations=1) 
        color_mask = cv.merge([color_mask, color_mask, color_mask])
        mask = cv.bitwise_or(mask, color_mask)

    # mask = cv.morphologyEx(mask, cv.MORPH_OPEN, open_kernel, iterations=2)
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    cnts = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # Sort all contours from top-to-bottom or bottom-to-top
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)
    cnts = cnts[:9]
    (cnts, _) = contours.sort_contours(cnts, method="top-to-bottom")

    # Take each row of 3 and sort from left-to-right or right-to-left
    cube_rows = []
    row = []
    for (i, c) in enumerate(cnts, 1):
        # if c.
        row.append(c)
        if i % 3 == 0:  
            (cnts, _) = contours.sort_contours(row, method="left-to-right")
            cube_rows.append(cnts)
            row = []

    # Draw text
    number = 0
    
    for row in cube_rows:
        for c in row:
            x,y,w,h = cv.boundingRect(c)
            for color, (lower, upper) in colors.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)

                color_check = cube_hsv [int(y+h/2)][int(x+w/2)]
                check = True
                for i in range(3):
                    if color_check[i] not in range(lower[i], upper[i]+1):
                        check = False
                        break
                if check == True:   
                    cv.rectangle(original, (x, y), (x + w, y + h), (36,255,12), 2)
                    cv.putText(original, "#{}".format(color), (x, y + 15), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                    cube += str(color).lower()
                    break


            number += 1


    cv.imshow('mask', mask)
    cv.imshow('original', original)
    cv.waitKey()
    cv.destroyAllWindows()
print(cube)
from rubik_solver import utils

utils.pprint(cube)

# 'Beginner'
# 'CFOP'
# 'Kociemba'
print(utils.solve(cube, 'Kociemba'))