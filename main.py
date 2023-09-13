import cv2
import numpy as np
import pytesseract
import os

# C:\Program Files\Tesseract-OCR
path = 'Query.png'
per = 25
pixelThreshold = 500

# ------------------------------------------------------------
# roi - Region Of Interest
roi = [[(96, 978), (688, 1082), 'text', 'Name'],
       [(736, 978), (1330, 1078), 'text', 'Phone'],
       [(96, 1150), (154, 1206), 'box', 'Sign'],
       [(736, 1148), (796, 1206), 'box', 'Allergic'],
       [(100, 1414), (694, 1518), 'text', 'Email'],
       [(736, 1416), (1332, 1516), 'text', 'ID']]


pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

imgQ = cv2.imread(path)
h,w,c = imgQ.shape
# imgQ = cv2.resize(imgQ, (w//3, h//3))

# Feature extraction using ORB (Oriented FAST and Rotated BRIEF):
# creates an ORB detector with a maximum of 1000 keypoints.
orb = cv2.ORB_create(1000)
kp1, des1 = orb.detectAndCompute(imgQ, None)    # find key point and coresponding descriptors from the resized input image
# impKp1 = cv2.drawKeypoints(imgQ, kp1, None)
# resulting image with key points is stored in the variable

path = 'UserForms'
myPicList = os.listdir(path)
print(myPicList)
for j, y in enumerate(myPicList):
    img = cv2.imread(path + "/" + y)
    # img = cv2.resize(img, (w//3, h//3))
    # cv2.imshow(y, img)

    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)    # Brute-Force Matcher to match the descriptors of the current image (des2) with the descriptors of the reference image (des1).
    matches = bf.match(des2, des1)
    matches = list(matches)
    matches.sort(key= lambda x: x.distance) # matches are sorted by distance in ascending order.
    good = matches[:int (len(matches)*(per/100))]   # Give us 25% of the best matches
    imgMatch = cv2.drawMatches(img, kp2, imgQ, kp1, good[:50], None,flags=2)
    # imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    # cv2.imshow(y, imgMatch)

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(img, M, (w,h))

    # imgScan = cv2.resize(imgScan, (w//3, h//3))
    # cv2.imshow(y, imgScan)

    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    myData = []

    print(f'############# Extracting data from Form {j} ###############')

    for x, r in enumerate(roi):
        cv2.rectangle(imgMask, (r[0][0],r[0][1]),(r[1][0], r[1][1]), (0, 255, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1,0)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]: r[1][0]]
        # cv2.imshow(str(x), imgCrop)

        if r[2] == 'text':
            extracted_text = pytesseract.image_to_string(imgCrop).strip()  # Remove leading/trailing whitespace, including newlines
            print(f'{r[3]} : {extracted_text}')
            myData.append(extracted_text)
        if r[2] == 'box':
            imgGrey = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgGrey, 170, 255, cv2.THRESH_BINARY_INV)[1]
            totalPixels = cv2.countNonZero(imgThresh)

            if totalPixels > pixelThreshold: totalPixels = 1
            else: totalPixels = 0
            print(f'{r[3]} : {totalPixels}')
            myData.append(totalPixels)

    with open('DataOutput.csv','a+') as f:
        for data in myData:
            f.write((str(data)+','))
        f.write(('\n'))

    #imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    print(myData)
    #cv2.imshow(y+"2", imgShow)

# cv2.imshow("KeyPointQuery", impKp1)
# cv2.imshow("Output", imgQ)
cv2.waitKey(0)
