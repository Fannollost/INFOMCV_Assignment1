import numpy as np
#declare constants
RED    = (255,0,0)
ORANGE = (255, 102, 0)
YELLOW = (255, 255, 0)
GREEN  = (0,255,0)
LBLUE  = (102, 153, 255)
BLUE   = (0,0,255)

BOARD_SIZE = (9,6)

WINDOW_NAME = 'img'
WINDOW_SIZE = (60,40)

DATA_PATH   = './data/calibration.npz'
IMAGES_PATH = './pics/*.jpg'
IMAGES_PATH_YANNICK = './pics/yannick/test/*.jpg'
IMAGES_PATH_FABIEN = './pics/fabien/*.jpg'
IMAGES_PATH_DEFAULT = './pics/default/*.jpg'
IMAGES_PATH_FLOOR = './pics/floor/*.jpg'
IMAGES_PATH_TEST_MANUAL = './pics/testingSet/manual*.jpg'
IMAGES_PATH_TEST_ALL = './pics/testingSet/*.jpg'
IMAGES_PATH_TEST_SELECTION = './pics/testingSet/selection*.jpg'
IMAGES_PATH_TEST_SUB_SELECTION = './pics/testingSet/selection_sub*.jpg'

AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
CUBE_AXIS = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])

WEBCAM = False
FORCE_CALIBRATION = True