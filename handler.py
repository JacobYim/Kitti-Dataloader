import cv2
import datetime
import numpy as np
import os
import pandas as pd
class KiTTiDataHandler :
    def __init__ (self, date, segment) :
        self.low_memory = False
        self.seq_dir = "src/{}/{}_drive_{}_sync/".format(date, date, segment)
        self.pcl_dir = self.seq_dir+'velodyne_points/'
        self.pcl_data_dir = self.pcl_dir+'data/'
        self.pcl_data = os.listdir(self.pcl_data_dir)
        if self.low_memory :
            self.first_pointcloud = np.fromfile(self.pcl_data_dir+self.pcl_data[0], dtype=np.float32, count=-1).reshape((-1,4))
        else :
            self.pointclouds = []
            for i in range(len(self.pcl_data)) :
                self.pointclouds.append( np.fromfile(self.pcl_data_dir+self.pcl_data[i], dtype=np.float32, count=-1).reshape((-1,4)))