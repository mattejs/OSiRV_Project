import numpy as np
import cv2 as cv

def SkinMask(img):
    HSVIm = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")
    skinRegionHSV = cv.inRange(HSVIm, lower, upper)
    blurred = cv.blur(skinRegionHSV, (2,2))
    ret, thresh = cv.threshold(blurred,0,255,cv.THRESH_BINARY)
    return thresh

def GetContourAndHull(mask_img):
    contours, hierarchy = cv.findContours(mask_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv.contourArea(x))
    hull = cv.convexHull(contours)
    #cv.drawContours(image, [hull], -1, (0, 255, 0), 2)
    return contours, hull


def GetDefects(contours):
    hull = cv.convexHull(contours, returnPoints=False)
    defects = cv.convexityDefects(contours, hull)
    return defects

capture = cv.VideoCapture(0) # '0' for camera, or path to your image/video

while capture.isOpened():
    _, image = capture.read()

    try:
        mask_img = SkinMask(image)
        contours, hull = GetContourAndHull(mask_img)
        cv.drawContours(image, [contours], -1, (255,0,0), 2)
        cv.drawContours(image, [hull], -1, (0, 255, 0), 2)
        defects = GetDefects(contours)

        if defects is not None:
            count = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(contours[s][0])
                end = tuple(contours[e][0])
                far = tuple(contours[f][0])
                a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                if angle <= np.pi / 2:
                    count += 1
                    cv.circle(image, far, 4, [0, 0, 255], -1)
            if count > 0:
                count = count+1
            cv.putText(image, str(count), (30, 150), cv.FONT_HERSHEY_SIMPLEX,3, (255, 0, 0) , 4, cv.LINE_AA)
        cv.imshow("Video", image)

    except:
        pass
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
