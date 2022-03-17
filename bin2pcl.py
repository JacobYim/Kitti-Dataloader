import numpy as np
import open3d as o3d
import os

# file_dir = os.path.dirname(os.path.realpath(__file__))
file_dir = os.curdir
print(file_dir)
src_dir = file_dir+"/src/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data"
dist_dir =  file_dir+"/pcl"

if not os.path.isdir(dist_dir) :
	os.mkdir(dist_dir) 

files = os.listdir(src_dir)
for file in files :
	# Load binary point cloud
	print("working for {} ...".format(src_dir+'/'+file))
	bin_pcd = np.fromfile(src_dir+'/'+file, dtype=np.float32)

	# Reshape and drop reflection values
	points = bin_pcd.reshape((-1, 4))[:, 0:3]

	# Convert to Open3D point cloud
	o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

	# Save to whatever format you like
	o3d.io.write_point_cloud(dist_dir+"/"+file.split('.')[0]+".pcd", o3d_pcd)
