from handler import KiTTiDataHandler
from function import generate_lidar_map
import matplotlib.pyplot as plt
import pykitti
import numpy as np
import os

if __name__ == "__main__" :

    hd = KiTTiDataHandler('2011_09_26', '0001')
    generate_lidar_map(hd)