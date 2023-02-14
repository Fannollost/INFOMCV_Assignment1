import numpy as np
#declare constants
RED    = (255,0,0)
ORANGE = (255, 102, 0)
YELLOW = (255, 255, 0)
GREEN  = (0,255,0)
LBLUE  = (102, 153, 255)
BLUE   = (0,0,255)

BOARD_SIZE = (7,6)

WINDOW_NAME = 'img'

DATA_PATH   = './data/calibration.npz'
IMAGES_PATH = './pics/*.jpg'

AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
CUBE_AXIS = np.float32([[0,0,0], [0,1,0], [1,1,0], [1,0,0],
                   [0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1] ])