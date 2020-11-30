import cv2
from utils.utils import Params
from utils.utils import visualize
import os
import pandas as pd

class Dataprep():
    def __init__(self, config):
        self.config = Params(config)
        self.images_loc = self.config.imgfolder
        self.mask_loc = self.config.maskfolder
        self.train_dataframe = pd.read_csv(self.config.train_dataframepath)
        self.test_dataframe = pd.read_csv(self.config.test_dataframepath)
        self.valid_dataframe = pd.read_csv(self.config.valid_dataframepath)

    def showFirstImage(self, set=0):
        #set to choose which image set 0 for train, 1 for valid and 2 for test
        if set == 0:
            image_fps = os.path.join(self.images_loc, self.train_dataframe['filename'][0])
            image = cv2.imread(image_fps)
            visualize(image)
        if set == 1:
            image_fps = os.path.join(self.images_loc, self.valid_dataframe['filename'][0])
            image = cv2.imread(image_fps)
            visualize(image)
        if set == 2:
            image_fps = os.path.join(self.images_loc, self.test_dataframe['filename'][0])
            image = cv2.imread(image_fps)
            visualize(image)