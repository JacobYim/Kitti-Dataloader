from handler import KiTTiDataHandler
import matplotlib.pyplot as plt
import numpy as np
import os
if __name__ == "__main__" :

    i = 0
    show = False 
    hd = KiTTiDataHandler('2011_09_26', '0001')
    # Visualizing a lidar pointcloud with matplotlib.
    for i in range(len(hd.pointclouds)) :
        pointcloud = hd.pointclouds[i]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        xs = pointcloud[:, 0]#[::20]   # Uncomment if 3d plot runs too slow, takes every 20th point
        ys = pointcloud[:, 1]#[::20]
        zs = pointcloud[:, 2]#[::20]

        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.scatter(xs, ys, zs, s=0.01)
        ax.grid(False)
        ax.axis('off')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=90, azim=180)
        if not os.path.exists('pcl'):
            os.makedirs('pcl')
        plt.savefig("pcl/{}_{}_{}".format('2011_09_26', '0001', str(i)))
        if show :
            plt.show()

