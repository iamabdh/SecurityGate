from collections import deque
import cv2
import imutils
import numpy as np
import math
import time

path = dict()


def calculateDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


def structureArray(index):
    if index is not None:
        for i in range(1, 1000):
            if path.get(i) is None:
                index = i
                path.update({i: {
                    'point': [],
                    'pervPoint': deque(maxlen=1),
                    'alerted': False,
                    'storedInDataBase': False
                }})
                break
    return index


def checkingIsItNew(cx, cy):
    iindex = None
    idistance = None
    if len(path) != 0:
        for index in path:
            iindex = index
            pp = list(path.get(index).get('pervPoint'))
            idistance = calculateDistance(pp[0][0], pp[0][1], cx, cy)
            if idistance < 100:
                # not new
                return [True, index, idistance]

        return [False, structureArray(iindex + 1), idistance]
    else:
        return [False, structureArray(0)]


def storeInDataBase(index, color, enterZone=False):
    f = open("database.txt", "a")

    if enterZone:
        f.write(f'\n ***********************************'
                f'\nid: {index}, Unauthorized Person Enter Zone.  '
                f'\n Color: {label}'
                f'\n Time: {time.asctime(time.localtime())}')
    else:
        if len(path.get(index).get('point')) > 10 and not path.get(index).get('storedInDataBase'):

            if color != 'Blue':
                f.write(f'\n ***********************************'
                        f'\nid: {index}, Unauthorized Person Detected. '
                        f'\n Color: {label}'
                        f'\n Time: {time.asctime(time.localtime())}')
            else:
                f.write(f'\n ***********************************'
                        f'\nid: {index}, Authorized Person Detected. '
                        f'\n Color: {label}'
                        f'\n Time: {time.asctime(time.localtime())}')

            path.get(index)['storedInDataBase'] = True
    f.close()


def drawMap(imageFrame, index, cx, cy, color):
    if len(path.get(index).get('point')) > 40:

        if calculateDistance(202, 354, cx, cy) < 560 and not path.get(index).get('alerted'):

            for i in range(1, len(path.get(index).get('point'))):  # for all the points in the deque
                if path.get(index).get('point')[i - 1] is None or path.get(index).get(
                        'point') is None:  # if we have none as the current point or previous
                    continue  # Start the while loop all over again.

                cv2.line(imageFrame, path.get(index).get('point')[i - 1], path.get(index).get('point')[i], (0, 0, 225),
                         4)  # draw a line between
            storeInDataBase(index, color, True)
            path.get(index)['alerted'] = True

            cv2.imshow("detected", imutils.resize(imageFrame, 300, cv2.INTER_CUBIC))


# Select either reading frames from a camera or from a file
Capture = cv2.VideoCapture("dataset_7.mp4")
# Keep acquiring frames

perviousFrame = None

lastFrame = None
toggel = False
crop = None
while True:
    # Acquire a frame from the source
    (success, frame) = Capture.read()
    (success1, frame2) = Capture.read()

    # Check if a frame was successfully acquired
    if not success:
        break

    # process the captured frame
    # frame = imutils.resize(frame, 300, inter=cv2.INTER_LINEAR)
    gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.GaussianBlur(gray1, (43, 43), 0)
    gray2 = cv2.GaussianBlur(gray2, (43, 43), 0)

    # first frame
    frameDelta = cv2.absdiff(gray2, gray1)
    (T, thresh) = cv2.threshold(frameDelta, 30, 255, cv2.THRESH_TRIANGLE)

    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (43, 43))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cloneFrame = frame.copy()

    for index, cnt in enumerate(cnts):
        if cv2.contourArea(cnt) < 10000:
            continue

        # finding the aspect ratio
        (x, y, w, h) = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        aspectRatio = w / float(h)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        index = 0
        if cx is not None and cy is not None:
            status = checkingIsItNew(cx, cy)
            index = status[1]
            path.get(index).get('point').append((cx, cy))
            path.get(index).get('pervPoint').append((cx, cy))

        crop = cloneFrame[y - 50:y + h + 50, x - 50:x + w + 50]
        b = crop[:, :, :1]
        g = crop[:, :, 1:2]
        r = crop[:, :, 2:]

        label = ''
        # computing the mean
        b_mean = np.mean(b)
        g_mean = np.mean(g)
        r_mean = np.mean(r)
        if b_mean > g_mean and b_mean > r_mean:
            label = 'Blue'
            storeInDataBase(index, label)
        #     ignore
        elif g_mean > b_mean and g_mean > r_mean:
            label = 'Green'
            storeInDataBase(index, label)
            drawMap(cloneFrame, index, cx, cy, label)
        elif r_mean > g_mean and r_mean > b_mean:
            label = 'Red'
            storeInDataBase(index, label)
            drawMap(cloneFrame, index, cx, cy, label)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame,
                    f'index: {index}, '
                    f'color: {label},'
                    f' distance: {int(calculateDistance(202, 354, cx, cy))}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display captured frame
    cv2.imshow("Captured frame", imutils.resize(frame, width=400))
    # perviousFrame = None
    key = cv2.waitKey(30) & 0xFF
    # If user presses 'q', then quit
    if key == ord("q"):
        break

# clean up the camera and close any open windows
Capture.release()
cv2.destroyAllWindows()
