import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pykitti
import tensorflow as tf
from sklearn.linear_model import RANSACRegressor
from scipy import stats
from time import sleep
import zipfile

def load_data(data,idx):
  img_raw = np.array(data.get_cam2(idx))
  lidar_raw = np.array(data.get_velo(idx))[:,:3]
  lidar_raw = lidar_raw[lidar_raw[:,2]<=0,:]
  dist = np.linalg.norm(lidar_raw,axis=1)
  lidar_raw = lidar_raw[dist >= 2.5]
  return img_raw,lidar_raw

def transform_coordinate(lidar_points,extrinsic_matrix):
  inp = lidar_points.copy()
  inp = np.concatenate((inp,np.ones((inp.shape[0],1))),axis=1)
  inp = np.matmul(extrinsic_matrix,inp.T).T
  return inp[:,:3]

def project_lidar2cam(lidar_in_cam,camera_intrinsic,img_raw_size):
  lidar_in_cam = np.concatenate((lidar_in_cam,np.ones((lidar_in_cam.shape[0],1))),axis=1)
  lidar_in_cam = lidar_in_cam[lidar_in_cam[:,2]>0]

  lidar_2d = np.matmul(camera_intrinsic,lidar_in_cam[:,:3].T).T
  lidar_2d = np.divide(lidar_2d,lidar_2d[:,2].reshape((-1,1)))
  lidar_2d = lidar_2d.astype(int)

  maskH = np.logical_and(lidar_2d[:,0]>=0,lidar_2d[:,0]<img_raw_size[1])
  maskV = np.logical_and(lidar_2d[:,1]>=0,lidar_2d[:,1]<img_raw_size[0])
  mask = np.logical_and(maskH,maskV)
  lidar_2d = lidar_2d[mask,:]
  lidar_in_cam = lidar_in_cam[mask,:]

  return lidar_2d,lidar_in_cam[:,:3]

def crop_data(img_in,lidar_2d_in,lidar_in_cam_in,rh,rw):
  lidar_2d = lidar_2d_in.copy()
  lidar_in_cam = lidar_in_cam_in.copy()
  img = img_in.copy()

  dim_ori = np.array(img.shape)
  cent = (dim_ori/2).astype(int)
  if dim_ori[0]/dim_ori[1] == rh/rw:
      crop_img = img
    
  elif dim_ori[0] <= dim_ori[1]:
      cH2 = dim_ori[0]
      cW2 = cH2*rw/rh
      cW = int(cW2/2)
      crop_img = img[:,cent[1]-cW:cent[1]+cW+1]

  else:
      cW2 = dim_ori[1]
      cH2 = cW2*rh/rw
      cH = int(cH2/2)
      crop_img = img[cent[0]-cH:cent[0]+cH+1,:]

  cW = cW2/2
  cH = cH2/2
  centH = cent[0]
  centW = cent[1]
  maskH = np.logical_and(lidar_2d[:,1]>=centH-cH,lidar_2d[:,1]<=centH+cH)
  maskW = np.logical_and(lidar_2d[:,0]>=centW-cW,lidar_2d[:,0]<=centW+cW)
  mask = np.logical_and(maskH,maskW)
  lidar_2d = lidar_2d[mask,:]
  lidar_in_cam = lidar_in_cam[mask,:]
  cent = np.array((centW-cW,centH-cH,0)).reshape((1,3))
  lidar_2d = lidar_2d - cent

  return crop_img, lidar_2d.astype(int), lidar_in_cam

def process_images(img_in, sess, target_size=513, probability_threshold=0.5):
  INPUT_TENSOR_NAME = 'ImageTensor:0'
  PROB_TENSOR_NAME = 'SemanticProbabilities:0'
  INPUT_SIZE = target_size

  image = img_in.copy()
  sz = image.shape
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  if INPUT_SIZE == 0:
    resized_image = image.copy()
  else:
    resized_image = cv2.resize(image,(INPUT_SIZE,INPUT_SIZE))

  batch_seg_map = sess.run(
      PROB_TENSOR_NAME,
      feed_dict={INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
  seg_map = (batch_seg_map[0][:,:,1]*255).astype(int)
  prob = np.array(seg_map, dtype=np.uint8)
  prob = cv2.resize(prob,(sz[1],sz[0]))

  pred = prob.copy()
  msk_bin = prob >= (probability_threshold*255)
  pred[msk_bin] = 1
  pred[np.logical_not(msk_bin)] = 0

  _,segm_reg = cv2.connectedComponents(pred)
  segm_reg = segm_reg.astype(float)
  segm_reg[segm_reg==0] = np.nan
  modes,_ = stats.mode(segm_reg,axis=None)
  mode = modes[0]
  # pred[segm_reg!=mode] = 0
  
  return prob,(pred*255).astype(np.uint8)

def get_road_model_ransac(img_pred,lidar_in_cam,lidar_2d):
  lidar_in_road_lbl = [True if img_pred[pt[1],pt[0]] == 255 else False for pt in lidar_2d]
  lidar_in_road = lidar_in_cam[lidar_in_road_lbl,:]
  road_model = RANSACRegressor().fit(lidar_in_road[:,[0,2]],lidar_in_road[:,1])
  return road_model

def filter_road_points(road_model,lidar_in,threshold=0.5):
  x = lidar_in[:,[0,2]]
  y_true = lidar_in[:,1]
  y_pred = road_model.predict(x)
  delta_y = np.absolute(y_true-y_pred).flatten()
  is_not_road = delta_y > threshold
  lidar_out = lidar_in[is_not_road,:].copy()
  return lidar_out

def load_vehicle_pose_vel(data,idx,old_pose,old_idx):
  delta_t = (data.timestamps[idx]-data.timestamps[old_idx]).total_seconds()
  packet = data.oxts[idx].packet
  vf = packet.vf
  vr = -packet.vl
  pose_f = old_pose[0] + (vf*delta_t)
  pose_r = old_pose[1] + (vr*delta_t)
  pose_y = packet.yaw - data.oxts[0].packet.yaw
  return (pose_f,pose_r,pose_y)

def generate_measurement_ogm(lidar_in,ogm_shape):
  rphi_meas = np.zeros((lidar_in.shape[0],2))
  rphi_meas[:,1] = np.sqrt(np.add(np.square(lidar_in[:,0]),np.square(lidar_in[:,1])))/ALPHA
  rphi_meas[:,0] = (np.arctan2(lidar_in[:,1],lidar_in[:,0])+np.pi)/BHETA
  rphi_meas = np.unique(rphi_meas.astype(int),axis=0)
  rphi_meas = rphi_meas[rphi_meas[:,1]<int(MAX_RANGE/ALPHA),:]
  rphi_meas = rphi_meas[rphi_meas[:,0]<int(2*np.pi/BHETA),:]

  sg_ang_bin = int(2*np.pi/BHETA)
  sg_rng_bin = int(MAX_RANGE/ALPHA)
  scan_grid = np.ones((sg_ang_bin,sg_rng_bin))*0.5
  scan_grid[tuple(rphi_meas.T)] = 0.7
  
  for ang in range(sg_ang_bin):
    ang_arr = rphi_meas[rphi_meas[:,0]==ang,1]
    if len(ang_arr) == 0:
      scan_grid[ang,:] = 0.3
    else:
      min_r = np.min(ang_arr)
      scan_grid[ang,:min_r] = 0.3
  
  ogm_sz = (ogm_shape[1],ogm_shape[0])
  ogm_cen = (int(ogm_shape[1]/2),int(ogm_shape[0]/2))
  radius = (MAX_RANGE/RESOLUTION) + SPHERICAL2CARTESIAN_BIAS
  ogm_step = cv2.warpPolar(scan_grid,ogm_sz,ogm_cen,radius,cv2.WARP_INVERSE_MAP)
  ogm_step[OOR_MASK] = 0.5
  ogm_step = cv2.rotate(ogm_step, cv2.ROTATE_90_CLOCKWISE)
  return ogm_step

def logit(m):
  return np.log(np.divide(m, np.subtract(1, m)))

def inverse_logit(m):
  return np.divide(np.exp(m),np.add(1,np.exp(m)))

def update_ogm(prior_ogm,new_ogm):
  logit_map = logit(new_ogm) + logit(prior_ogm)
  out_ogm = inverse_logit(logit_map)
  out_ogm[out_ogm>=0.98] = 0.98
  out_ogm[out_ogm<=0.02] = 0.02
  return out_ogm
  
def shift_pose_ogm(ogm, init, fin):
  ogm_o = ogm.copy()
  theta = init[2] /180 * np.pi
  rot_m = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
  trs_m = np.array([[init[0]],[init[1]]])
  point = np.array(fin[:2]).reshape((-1,1))
  point_1 = (point - trs_m)
  point_2 = np.dot(rot_m,-point_1)
  delta_theta = (fin[2] - init[2])
  delta = np.array([point_2[1,0]/RESOLUTION,point_2[0,0]/RESOLUTION,0])

  M = np.array([[1,0,delta[0]],[0,1,-delta[1]]])
  dst = cv2.warpAffine(ogm_o,M,(ogm_o.shape[1],ogm_o.shape[0]),borderValue=0.5)
  M = cv2.getRotationMatrix2D((ogm_o.shape[1]/2+0.5,ogm_o.shape[0]/2+0.5),delta_theta,1)
  dst = cv2.warpAffine(dst,M,(ogm_o.shape[1],ogm_o.shape[0]),borderValue=0.5)
  return dst

def generate_measurement_dgm(lidar_in,dgm_shape):
  rphi_meas = np.zeros((lidar_in.shape[0],2))
  rphi_meas[:,1] = np.sqrt(np.add(np.square(lidar_in[:,0]),np.square(lidar_in[:,1])))/ALPHA
  rphi_meas[:,0] = (np.arctan2(lidar_in[:,1],lidar_in[:,0])+np.pi)/BHETA
  rphi_meas = np.unique(rphi_meas.astype(int),axis=0)
  rphi_meas = rphi_meas[rphi_meas[:,1]<int(MAX_RANGE/ALPHA),:]
  rphi_meas = rphi_meas[rphi_meas[:,0]<int(2*np.pi/BHETA),:]

  sg_ang_bin = int(2*np.pi/BHETA)
  sg_rng_bin = int(MAX_RANGE/ALPHA)
  scan_grid = np.zeros((sg_ang_bin,sg_rng_bin,3))
  scan_grid[:,:,0] = 1 
  scan_grid[tuple(rphi_meas.T)] = (1-OCC_CONF,OCC_CONF,0)
  
  for ang in range(sg_ang_bin):
    ang_arr = rphi_meas[rphi_meas[:,0]==ang,1]
    if len(ang_arr) == 0:
      scan_grid[ang,:] = (1-FREE_CONF,0,FREE_CONF)
    else:
      min_r = np.min(ang_arr)
      scan_grid[ang,:min_r] = (1-FREE_CONF,0,FREE_CONF)
  
  dgm_sz = (dgm_shape[1],dgm_shape[0])
  dgm_cen = (int(dgm_shape[1]/2),int(dgm_shape[0]/2))
  radius = (MAX_RANGE/RESOLUTION) + SPHERICAL2CARTESIAN_BIAS
  dgm_step = cv2.warpPolar(scan_grid,dgm_sz,dgm_cen,radius,cv2.WARP_INVERSE_MAP)
  dgm_step[OOR_MASK] = (1,0,0)
  dgm_step = cv2.rotate(dgm_step, cv2.ROTATE_90_CLOCKWISE)
  return dgm_step

def update_dgm(prior_dgm,new_dgm):
  conflict_mass = np.multiply(prior_dgm[:,:,2],new_dgm[:,:,1])
  conflict_mass = np.add(conflict_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,2]))

  free_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,2])
  free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,0]))
  free_mass = np.add(free_mass,np.multiply(prior_dgm[:,:,2],new_dgm[:,:,2]))
  free_mass = np.divide(free_mass,1-conflict_mass)

  occ_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,1])
  occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,0]))
  occ_mass = np.add(occ_mass,np.multiply(prior_dgm[:,:,1],new_dgm[:,:,1]))
  occ_mass = np.divide(occ_mass,1-conflict_mass)

  unknown_mass = np.multiply(prior_dgm[:,:,0],new_dgm[:,:,0])
  unknown_mass = np.divide(unknown_mass,1-conflict_mass)

  updated_dgm = np.stack((unknown_mass,occ_mass,free_mass),axis=2)
  return updated_dgm,conflict_mass

def predict_dgm(dgm,dynamic_mass):
  max_mass = np.argmax(dgm,axis=2)
  pred_map = np.zeros(dgm.shape)
  pred_map[max_mass==0] = (123,123,123)
  pred_map[max_mass==1] = (0,0,0)
  pred_map[max_mass==2] = (255,255,255)
  pred_map[dynamic_mass>=DYNAMIC_THRESHOLD] = (0,0,255)
  return pred_map.astype(np.uint8)

def shift_pose_dgm(dgm, init, fin):
  dgm_o = dgm.copy()
  theta = init[2] /180 * np.pi
  rot_m = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta),np.cos(theta)]])
  trs_m = np.array([[init[0]],[init[1]]])
  point = np.array(fin[:2]).reshape((-1,1))
  point_1 = (point - trs_m)
  point_2 = np.dot(rot_m,-point_1)
  delta_theta = (fin[2] - init[2])
  delta = np.array([point_2[1,0]/RESOLUTION,point_2[0,0]/RESOLUTION,0])

  M = np.array([[1,0,delta[0]],[0,1,-delta[1]]])
  dst = cv2.warpAffine(dgm_o,M,(dgm_o.shape[1],dgm_o.shape[0]),borderValue=0.5)
  M = cv2.getRotationMatrix2D((dgm_o.shape[1]/2+0.5,dgm_o.shape[0]/2+0.5),delta_theta,1)
  dst = cv2.warpAffine(dst,M,(dgm_o.shape[1],dgm_o.shape[0]),borderValue=0.5)
  return dst

### Processed according to the step-by-step tutorial
def single_loop_ogm(data,idx,tf_sess,ogm):
  '''
  Args:
    data = pykitti object that has been loaded
    idx = index of the processed frame
    tf_sess = TensorFlow session with loaded DeepLabv3+ model
    ogm = the latest estimated OGM
  Returns:
    updated_ogm = the updated OGM
    pose = the latest pose of the vehicle
    crop_img = the cropped camera image
  Note:
    Other parameters are defined globally
  '''

  img_raw,lidar_raw = load_data(data,idx)
  img_raw_size = img_raw.shape
  lidar_raw = transform_coordinate(lidar_raw,LIDAR2CAM_EXTRINSIC)
  lidar_2d,lidar_in_cam = project_lidar2cam(lidar_raw,CAMERA_INTRINSIC,img_raw_size)
  crop_img,lidar_2d,lidar_in_cam = crop_data(img_raw,lidar_2d,lidar_in_cam,CROP_RH,CROP_RW)
  _,segm_pred = process_images(crop_img, tf_sess, DEEPLAB_INPUT_SIZE, 0.5)
  road_model = get_road_model_ransac(segm_pred,lidar_in_cam,lidar_2d)
  lidar_nonroad = filter_road_points(road_model,lidar_raw,ROAD_HEIGHT_THRESHOLD)
  lidar_ogm = lidar_nonroad[:,[2,0]]

  pose = load_vehicle_pose_vel(data,idx,OLD_POSE,OLD_IDX)
  shifted_ogm = shift_pose_ogm(ogm,OLD_POSE,pose)
  ogm_step = generate_measurement_ogm(lidar_ogm,ogm.shape)
  updated_ogm = update_ogm(shifted_ogm,ogm_step)

  return updated_ogm,pose,crop_img

### Processed according to the step-by-step tutorial
def single_loop_dgm(data,idx,tf_sess,dgm):
  '''
  Args:
    data = pykitti object that has been loaded
    idx = index of the processed frame
    tf_sess = TensorFlow session with loaded DeepLabv3+ model
    dgm = the latest estimated DGM
  Returns:
    updated_dgm = the updated DGM (:,:,3)
    dynamic_mass = the conflicting mass map (:,:,1)
    pose = the latest pose of the vehicle
    crop_img = the cropped camera image
  Note:
    Other parameters are defined globally
  '''

  img_raw,lidar_raw = load_data(data,idx)
  img_raw_size = img_raw.shape
  lidar_raw = transform_coordinate(lidar_raw,LIDAR2CAM_EXTRINSIC)
  lidar_2d,lidar_in_cam = project_lidar2cam(lidar_raw,CAMERA_INTRINSIC,img_raw_size)
  crop_img,lidar_2d,lidar_in_cam = crop_data(img_raw,lidar_2d,lidar_in_cam,CROP_RH,CROP_RW)
  _,segm_pred = process_images(crop_img, tf_sess, DEEPLAB_INPUT_SIZE, 0.5)
  road_model = get_road_model_ransac(segm_pred,lidar_in_cam,lidar_2d)
  lidar_nonroad = filter_road_points(road_model,lidar_raw,ROAD_HEIGHT_THRESHOLD)
  lidar_dgm = lidar_nonroad[:,[2,0]]

  pose = load_vehicle_pose_vel(data,idx,OLD_POSE,OLD_IDX)
  shifted_dgm = shift_pose_dgm(dgm,OLD_POSE,pose)
  dgm_step = generate_measurement_dgm(lidar_dgm,dgm.shape)
  updated_dgm,dynamic_mass = update_dgm(shifted_dgm,dgm_step)

  return updated_dgm,dynamic_mass,pose,crop_img

### Load KITTI data
basedir = 'src/'
date = '2011_09_26'
drive = '0001'
data = pykitti.raw(basedir, date, drive)
NUMBER_DATA = len(data.oxts)

### Global parameters (Perception)
LIDAR2CAM_EXTRINSIC = data.calib.T_cam2_velo
CAMERA_INTRINSIC = data.calib.K_cam2
CROP_RH = 3
CROP_RW = 4
DEEPLAB_MODEL_PATH = 'deeplab_model.pb'
DEEPLAB_INPUT_SIZE = 513
ROAD_HEIGHT_THRESHOLD = 0.15

### Global parameters (DGM)
ALPHA = 1
BHETA = 1*np.pi/180
RESOLUTION = 0.1
MAX_RANGE = 50
MAP_WIDTH = 100
SPHERICAL2CARTESIAN_BIAS = 6
MAP_SIZE_X = int(MAP_WIDTH/RESOLUTION)
MAP_SIZE_Y = int(MAP_WIDTH/RESOLUTION)
xarr = np.arange(-MAP_WIDTH/2,MAP_WIDTH/2,RESOLUTION)
yarr = np.arange(-MAP_WIDTH/2,MAP_WIDTH/2,RESOLUTION)
MAP_XX, MAP_YY = np.meshgrid(xarr, -yarr)
rgrid = np.sqrt(np.add(np.square(MAP_XX),np.square(MAP_YY)))
OOR_MASK = rgrid >= MAX_RANGE
FREE_CONF = 0.7
OCC_CONF = 0.7
DYNAMIC_THRESHOLD = 0.1

is_save = True
save_dir = 'results_ogm/'
if not os.path.exists(save_dir): os.makedirs(save_dir)

### Load DeepLab v3+ model
with open(DEEPLAB_MODEL_PATH, "rb") as f:
    graph_def = tf.compat.v1.GraphDef.FromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def=graph_def, name="")
sess = tf.compat.v1.Session(graph=graph)

### Initiate OGM
ogm = np.ones((MAP_SIZE_Y,MAP_SIZE_X)) * 0.5

### Process all the data in sequence
idx = 0
OLD_IDX = 0
OLD_POSE = (0,0,0)
frequency = 1

while True:
  if idx >= NUMBER_DATA: break
  ogm,pose,camera_img = single_loop_ogm(data,idx,sess,ogm)
  OLD_IDX = idx
  OLD_POSE = pose
  idx = idx + frequency

  ### Visualize
  fig,axs = plt.subplots(1,2,figsize=(16,8))
  ogm_img = ((1-ogm)*255).astype(np.uint8)
  ogm_img = cv2.resize(ogm_img,(500,500))
  ogm_img = cv2.cvtColor(ogm_img,cv2.COLOR_GRAY2RGB)
  center = (int(ogm_img.shape[1]/2),int(ogm_img.shape[0]/2)) 
  cv2.circle(ogm_img,tuple(center[:2]),5,(255,0,0),-1)
  axs[0].imshow(ogm_img,cmap='gray',vmin=0,vmax=255)
  axs[1].imshow(camera_img)
  axs[0].set_axis_off()
  axs[1].set_axis_off()
  if is_save:
    plt.savefig(f'{save_dir}{OLD_IDX:03d}.png')
    plt.close(fig)
  else:
    # plt.show()
    clear_output(wait=True)

if is_save:
  zip_file = f'ogm_results.zip'
  with zipfile.ZipFile(zip_file, 'w') as z:
    list_f = os.listdir(save_dir)
    for fl in list_f:
      z.write(save_dir+fl,fl)

