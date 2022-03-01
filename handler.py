import cv2
import datetime
import numpy as np
import os
import pandas as pd
import pykitti

class KiTTiDataHandler :
    def __init__ (self, date, drive) :
        # self.low_memory = False
        # self.seq_dir = "src/{}/{}_drive_{}_sync/".format(date, date, segment)
        
        # self.pcl_dir = self.seq_dir+'velodyne_points/'
        # self.pcl_data_dir = self.pcl_dir+'data/'
        # self.pcl_data = os.listdir(self.pcl_data_dir)

        # self.left_image_files = os.listdir(self.seq_dir+'image_00')
        # self.right_image_files = os.listdir(self.seq_dir+'image_01')

        # self.num_frames = len(self.pcl_data)

        self.data = pykitti.raw('src/', date, drive)
        self.pointclouds = []
        i = 0
        while True :
            try:
                self.pointclouds.append(self.data.get_velo(i))
                i = i + 1
            except :
                break
        # for i in range(len(self.pcl_data)) :
        #     self.pointclouds.append( np.fromfile(self.pcl_data_dir+self.pcl_data[i], dtype=np.float32, count=-1).reshape((-1,4)))

# basedir = 'src/'
# date = '2011_09_26'
# drive = '0001'