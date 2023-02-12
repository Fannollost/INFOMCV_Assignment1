import numpy as np
import constants as const
import cv2 as cv
import glob
import math
import os.path

global clickPoints
global counter
clickPoints = []
counter = 0

def pickColor(column):
    match column:
        case 0:
            return const.RED
        case 1:
            return const.ORANGE
        case 2: 
            return const.YELLOW
        case 3:
            return const.GREEN
        case 4: 
            return const.LBLUE
        case 5: 
            return const.BLUE
        case _:
            return const.RED

def undistortImage(filename, mtx, dist):
    img = cv.imread(filename)
    h , w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    #undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def showImage(name, image, wait = -1):
    cv.imshow(name, image)
    if(wait >= 0):
        cv.waitKey(wait)

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        global counter
        clickPoints.append((x,y))
        counter += 1

def main():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    if(os.path.isfile(const.DATA_PATH) != True):
        #prepare object points
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real wold space
        imgpoints = [] # 2d points in image space

        images = glob.glob(const.IMAGES_PATH)

        for fname in images:

            img = cv.imread(fname, 1)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            global counter
            global clickPoints
            clickPoints = []
            counter = 0
            #find the chessboard corners
            ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
            #if found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(11,11), (-1,-1), criteria)

                imgpoints.append(corners2)
                objpoints.append(objp) 

                # Draw and display the corners
                cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, ret)
                showImage(const.WINDOW_NAME, img, 50)
            else:
                showImage(const.WINDOW_NAME, img)
                while(counter < 4):
                    #Get mouseinput
                    cv.setMouseCallback(const.WINDOW_NAME, click_event)
                    cv.waitKey(1)

                    #visual feedback for mouseclicks
                    if(counter != 0):
                        img = cv.circle(img, clickPoints[counter - 1], 5, const.RED)
                        showImage(const.WINDOW_NAME, img)

                #prepare the pointset
                interpolatedPoints = np.zeros((const.BOARD_SIZE[1], const.BOARD_SIZE[0], 2))
                largest = 0
                smallest = 5000 

                #indexes
                diagonalPoint = 0
                closestPoint = 0

                #find closest and diagonal point
                for j in range(len(clickPoints)):
                    dist = math.dist(clickPoints[0], clickPoints[j])
                    if(dist != 0 and smallest > dist):
                        smallest = dist
                        closestPoint = j
                    if(dist != 0 and largest < dist):
                        largest = dist
                        diagonalPoint = j

                #determine approximate distance between points
                shortSteps = math.dist(clickPoints[0],clickPoints[closestPoint]) / (const.BOARD_SIZE[1])
                longSteps = math.dist(clickPoints[closestPoint], clickPoints[diagonalPoint]) / (const.BOARD_SIZE[0])

                #generate uniform set of points
                interpolatedPoints[0,0] = clickPoints[0]
                orig = clickPoints[0]
                for x in range(const.BOARD_SIZE[0]):
                    for y in range(const.BOARD_SIZE[1]):
                        interpolatedPoints[y,x] = (orig[0] + longSteps * x, orig[1] + shortSteps * y)

                #get uniform corners      
                uniform = np.array((orig, (orig[0] + longSteps * 6, orig[1] + shortSteps * 0),
                (orig[0] + longSteps * 6, orig[1] + shortSteps * 5),
                (orig[0] + longSteps * 0, orig[1] + shortSteps * 5))).astype(np.float32)
                dst = np.array(clickPoints).astype(np.float32)

                #transform uniform set of points to desired cornerpoints
                transform_mat = cv.findHomography(uniform,dst)[0]
                corners2 = cv.perspectiveTransform(interpolatedPoints, transform_mat)
                corners2 = np.array(corners2).reshape(42,2).astype(np.float32)
                corners2 = cv.cornerSubPix(gray,corners2,(20,20), (-1,-1), criteria)

                imgpoints.append(corners2)
                objpoints.append(objp) 

                # Draw and display the corners
                cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, True)
                showImage(const.WINDOW_NAME, img, 1000)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        np.savez(const.DATA_PATH, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    else:
        calibration = np.load(const.DATA_PATH)
        
        #extract calibration values from the file:
        mtx = calibration['mtx']
        dist = calibration['dist']
        rvecs = calibration['rvecs']
        tvecs = calibration['tvecs']

        
    #undist = undistortImage('./pics/left12.jpg', mtx, dist)
    #showImage('undist', undist, 0)
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()