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

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    pt1 = (int(corner[0]), int(corner[1]))
    dest1 = tuple(imgpts[0].ravel())
    dest2 = tuple(imgpts[1].ravel())
    dest3 = tuple(imgpts[2].ravel())
    img = cv.line(img, pt1, (int(dest1[0]),int(dest1[1])), (255,0,0), 5)
    img = cv.line(img, pt1, (int(dest2[0]),int(dest2[1])), (0,255,0), 5)
    img = cv.line(img, pt1, (int(dest3[0]),int(dest3[1])), (0,0,255), 5)
    return img

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


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

def checkQuality(gray, corners, limit):
    retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
    return retval[0] > limit

def improveQuality(gray):
    ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
    if ret == True:
        retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
        print("Sharpness : " + str(retval[0]))

    edges = cv.Canny(gray, 150, 400)
    h , w = gray.shape[:2]
    for l in range(h):
        for c in range(w):
            if(edges[l,c] > 250):
                gray[l,c] = 0
    showImage(const.WINDOW_NAME,gray,1500)
    ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)
    if ret == True:
        retval, sharp = cv.estimateChessboardSharpness(gray, const.BOARD_SIZE, corners)
        print("Corrected sharpness : " + str(retval[0] ))
    return gray, ret, corners

def drawOrigin(frame, criteria, objp, mtx,dist):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, const.BOARD_SIZE, None)

    if (ret == True):
        corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

        imgpts, jac = cv.projectPoints(const.AXIS, rvecs, tvecs, mtx, dist)
        cubeimgpts, jac = cv.projectPoints(const.CUBE_AXIS, rvecs, tvecs, mtx, dist)
        img = draw(frame, corners2, imgpts)
        img = drawCube(img, corners2, cubeimgpts)
        return img
    else:
        return frame


def main():
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((const.BOARD_SIZE[0]*const.BOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:const.BOARD_SIZE[0], 0:const.BOARD_SIZE[1]].T.reshape(-1,2)
    if(os.path.isfile(const.DATA_PATH) != True or const.FORCE_CALIBRATION):
        #prepare object points

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real wold space
        imgpoints = [] # 2d points in image space

        images = glob.glob(const.IMAGES_PATH_TEST)

        global counter
        global clickPoints
        for fname in images:
            print(fname)
            img = cv.imread(fname, 1)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            clickPoints = []
            counter = 0

            #find the chessboard corners
            gray, ret, corners = improveQuality(gray)
            #if ret and checkQuality(gray, corners, 5):
            #    continue


            #if found, add object points, image points (after refining them)
            if ret == True:
                corners2 = cv.cornerSubPix(gray,corners,(5,5), (-1,-1), criteria)

                imgpoints.append(corners2)
                objpoints.append(objp) 

                # Draw and display the corners
                cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, ret)
                showImage(const.WINDOW_NAME, img, 300)

                #edges = cv.Canny(img, 150, 400)
                #showImage(const.WINDOW_NAME,edges,5000)
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
                stepFactorX = const.BOARD_SIZE[0] - 1
                stepFactorY = const.BOARD_SIZE[1] - 1

                uniform = np.array((orig, 
                (orig[0] + longSteps * stepFactorX, orig[1] + shortSteps * 0),
                (orig[0] + longSteps * stepFactorX, orig[1] + shortSteps * stepFactorY),
                (orig[0] + longSteps * 0, orig[1] + shortSteps * stepFactorY))).astype(np.float32)
                dst = np.array(clickPoints).astype(np.float32)
  
                #transform uniform set of points to desired cornerpoints
                transform_mat = cv.findHomography(uniform,dst)[0]
                corners2 = cv.perspectiveTransform(interpolatedPoints, transform_mat)
                corners2 = np.array(corners2).reshape(const.BOARD_SIZE[0]*const.BOARD_SIZE[1],2).astype(np.float32)

                edges = cv.Canny(img, 150, 400)
                corners2 = cv.cornerSubPix(edges,corners2,(5,5), (-1,-1), criteria)

                imgpoints.append(corners2)
                objpoints.append(objp) 

                # Draw and display the corners
                cv.drawChessboardCorners(img, const.BOARD_SIZE, corners2, True)
                showImage(const.WINDOW_NAME, img, 3000)

        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        # https://stackoverflow.com/questions/23781089/opencv-calibratecamera-2-reprojection-error-and-custom-computed-one-not-agree?rq=1
        # https://stackoverflow.com/questions/37901806/reprojection-of-calibratecamera-and-projectpoints
        print("Root-mean-square error : "+str(ret)+ " px")
        np.savez(const.DATA_PATH, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    else:
        calibration = np.load(const.DATA_PATH)
        
        #extract calibration values from the file:
        mtx = calibration['mtx']
        dist = calibration['dist']

    print("F")
    #static online phase!
    if(const.WEBCAM == True):
        cap = cv.VideoCapture(0)
        
        if not cap.isOpened():
            raise IOError("Webcam not accessible")

        while True:
            ret, frame = cap.read()
            img = drawOrigin(frame, criteria, objp, mtx, dist)
            cv.imshow(const.WINDOW_NAME, img)

            c = cv.waitKey(1)
            if c == 27:
                break

            try :
                cv.getWindowProperty(const.WINDOW_NAME, 0)
            except :
                break
        cap.release()
    else:    
        frame = cv.imread('./pics/testimg3.jpg',1)
        print(dist)
        img = drawOrigin(frame, criteria, objp, mtx, dist)
        showImage(const.WINDOW_NAME, img, 0)

    #webcam online phase!
    #undist = undistortImage('./pics/left12.jpg', mtx, dist)
    #showImage('undist', undist, 0)

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()