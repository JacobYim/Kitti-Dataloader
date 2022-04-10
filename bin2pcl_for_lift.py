import numpy as np
import struct
from open3d import *

def convert_kitti_bin_to_pcd(binFilePath):
    size_float = 4
    list_pcd = []
    with open(binFilePath, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = PointCloud()
    pcd.points = Vector3dVector(np_pcd)
    return pcd

pcd = convert_kitti_bin_to_pcd("test.bin")
io.write_point_cloud("test.pcd", pcd)
