import numpy as np
import cv2 as cv
import glob

def undistortImage(filename, mtx, dist):
    img = cv.imread(filename)
    h , w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    #undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def showImage(name, image, wait):
    cv.imshow(name, image)

def cornersUserInput(name, img, board_size):
    cv.setMouseCallback(name, click_event)

def click_event(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        cv.circle(img, (x,y), 1, (0,0,255))
        cv.imshow('img', img)

def main():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    #redefine for own images! 
    BOARD_SIZE = (7,6)

    #prepare object points
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

    # Arrays to stroe object points and image points from all the images.
    objpoints = [] # 3d point in real wold space
    imgpoints = [] # 2d points in image space

    images = glob.glob('./pics/*.jpg')

    for fname in images:

        img = cv.imread(fname, 1)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, BOARD_SIZE, None)
        #if found, add object points, image points (after refining them)
        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11), (-1,-1), criteria)
    
            imgpoints.append(corners2)
            objpoints.append(objp) 

            # Draw and display the corners
            cv.drawChessboardCorners(img, BOARD_SIZE, corners2, ret)
            showImage('img', img, 100)
        else:
            # showImage('img', img, 0)
            cv.imshow('img', img)
            cv.setMouseCallback('img', click_event)
            cv.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savez('./data/calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

    undist = undistortImage('./pics/left12.jpg', mtx, dist)
    showImage('undist', undist, 0)
    cv.destroyAllWindows()




if __name__ == "__main__":
    main()