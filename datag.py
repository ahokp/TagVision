import math
import apriltag as at
import cv2
import numpy as np
from mpl_toolkits import mplot3d 
from matplotlib import pyplot as plt
from scipy.optimize import least_squares

def create_world_rotation_matrix(x, y, z):
    R_x = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    R_y = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    R_z = np.array([[np.cos(z), -np.sin(z), 0],
                    [np.sin(z), np.cos(z), 0],
                    [0, 0, 1]])
    
    R = np.matmul(R_z, np.matmul(R_y, R_x))
    return R.T

def model_func(extr, Rt, tt, og):
    R = np.array([[extr[0], extr[1], extr[2]],
                  [extr[3], extr[4], extr[5]],
                  [extr[6], extr[7], extr[8]]]).astype(np.float32)
    t = np.array([[extr[9], extr[10], extr[11]]]).T.astype(np.float32)

    residuals = []
    for i in range(len(Rt)):
        d = np.linalg.norm(np.matmul(Rt[i], og[i]) + tt[i] + np.matmul(R.T, t))
        residuals.append(d)
    
    return residuals

tag_size = 0.158
# World position(xyz) of the tags
tag_ts = [np.array([[0.611], [0.079], [0]]),
          np.array([[0.079], [0.388], [0]]),
          np.array([[0.079], [0.079], [0]]),
          np.array([[0.611], [0.388], [0]])]

# Orientation of the tags in radian
tag_w_ori = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [0, 0, 0]]

fig = plt.figure()
ax = plt.axes(projection="3d")

tag_Rs = []
total_tags = len(tag_w_ori)
for i in range(total_tags):
    # Create tag-to-world rotations
    rot = tag_w_ori[i]
    tag_Rs.append(create_world_rotation_matrix(rot[0], rot[1], rot[2]))

    # Plot tag locations
    tag_pos = tag_ts[i]
    ax.scatter3D(tag_pos[0], tag_pos[1], tag_pos[2])

# Camera params
K = np.array([])
dist = None

for n in range(15):

    imgpath = f'/home/pool/Documents/detection/apriltag/images/img{n+1}.png'
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    img_size = (img.shape[1], img.shape[0])

    detector = at.apriltag('tag36h11')
    detections = detector.detect(img)
    tag_count = len(detections)

    ipts = []
    opts = []
    ids = []

    obj_points = np.array([
                        -1, -1, 0,
                        1, -1, 0,
                        1, 1, 0,
                        -1, 1, 0]).astype(np.float32).reshape(1, -1, 3)*tag_size/2

    for i in range(tag_count):
        tag = detections[i]
        img_points = tag['lb-rb-rt-lt'].reshape(1, -1, 2).astype(np.float32)
        
        ipts.append(img_points)
        opts.append(obj_points)
        ids.append(int(tag['id']))
    
    if K.size == 0:
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                                        opts, ipts,
                                        img_size, None, None)   

    tag_Rts = []
    tag_tts = []
    tag_og = []
    cam_R = np.zeros((3, 3))
    cam_t = np.zeros((3, 1))

    for i in range(tag_count):
        ret, rvec, tvec, _ = cv2.solvePnPRansac(opts[i][0], ipts[i][0], K, dist)

        id = ids[i]
        Rw = tag_Rs[id]
        tw = tag_ts[id]

        Rt, _ = cv2.Rodrigues(np.float32(rvec))
        t = tvec
        Rc = Rt.T
        tc = -t

        # Rotation to align with world coord system
        R = np.matmul(Rw, Rc)
        # xyz position in tag coord system
        pos_t = np.matmul(R, tc)
        # Position in world coord system
        p = pos_t + tw
        #ax.scatter3D(p[0], p[1], p[2])

        # Translation
        tcw = np.matmul(-R, p)
        # Object pose?
        o_pos = np.matmul(R.T, tcw)
        #ax.scatter3D(o_pos[0], o_pos[1], o_pos[2])

        # Camera coord system?
        #ax.scatter3D(0, 0, 0)
        tposc = np.matmul(R.T, -tw) + tcw
        #ax.scatter3D(tposc[0], tposc[1], tposc[2])
        
        tag_og.append(np.matmul(Rw, tw))
        tag_Rts.append(Rt)
        tag_tts.append(t)
        cam_R += R
        cam_t += tcw

    # Camera pose avg
    Rcm = cam_R/tag_count
    tcm = cam_t/tag_count
    avg_pos = np.matmul(Rcm.T, -tcm)
    ax.scatter3D(avg_pos[0], avg_pos[1], avg_pos[2])

    # Initial camera extrinsic
    init_extrinsic = np.concatenate((Rcm, tcm.T), axis=0).reshape(12)
    
    #init_extrinsic = [-0.5, 0.6, -0.6, 0.8, 0.5, -0.3, 0.1, -0.6, -0.8, 0.6, -0.9, 0.8]
    # Optimize camera extrinsic
    extrinsic = least_squares(model_func, init_extrinsic,
                              args=(tag_Rts, tag_tts, tag_og))
                              
    extr = extrinsic['x']
    R = np.array([[extr[0], extr[1], extr[2]],
                  [extr[3], extr[4], extr[5]],
                  [extr[6], extr[7], extr[8]]]).astype(np.float32)
    t = np.array([[extr[9], extr[10], extr[11]]]).T.astype(np.float32)
    

    '''
    print(R)
    print(cam_R/tag_count)
    print()
    print(t)
    print(cam_t/tag_count)
    '''
    
    p = np.matmul(R.T, -t)
    #ax.scatter3D(p[0], p[1], p[2])
    #break
    
plt.show()
