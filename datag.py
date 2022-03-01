import apriltag as at
import cv2
import numpy as np
from mpl_toolkits import mplot3d 
from matplotlib import pyplot as plt
import LMA

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

def model_func(extr, args):
    """ Model function
    Note: This function is used as an optimization/error function in
    Levenberg-Marquardt algorithm.
    :param extr: values to be optimized
    :param args: values used in optimization
    :return: Distances calculated using the given parameters
    """
    rv = np.array([extr[0], extr[1], extr[2]]).astype(np.float32)
    # Rotation matrix
    R, _ = cv2.Rodrigues(rv)
    R_inv = np.linalg.inv(R)
    # Translation vector
    t = np.array([[extr[3], extr[4], extr[5]]]).T.astype(np.float32)
    
    Rt, tt, og = args
    residuals = []
    for i in range(len(Rt)):
        d = np.linalg.norm(np.matmul(Rt[i], og[i]) + tt[i] + np.matmul(R_inv, t))
        residuals.append(d)
    
    return np.array(residuals)

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

tag_Rs = []
total_tags = len(tag_w_ori)
for i in range(total_tags):
    # Create tag-to-world rotations
    rot = tag_w_ori[i]
    tag_Rs.append(create_world_rotation_matrix(rot[0], rot[1], rot[2]))

    # Plot tag locations
    tag_pos = tag_ts[i]
    #ax.scatter3D(tag_pos[0], tag_pos[1], tag_pos[2])

# Camera params
K = np.array([])
dist = None

for n in range(15):

    imgpath = f'/home/pool/Documents/detection/TagVision/images/img{n+1}.png'
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
        # Tag-to-world rotation
        Rw = tag_Rs[id]
        # Tag origin coords in world coordinate system
        tw = tag_ts[id]

        # Rotation and translation of the tag
        Rt, _ = cv2.Rodrigues(np.float32(rvec))
        t = tvec
        # Rotation and translation of the camera in tag coord system
        Rc = Rt.T
        tc = -t
        
        # Rotation to align with world coord system
        R = np.matmul(Rw, Rc)
        # Camera position in tag coord system
        pos_t = np.matmul(R, tc)
        # Camera position in world coord system
        p = pos_t + tw

        # Global origin in camera coordinate system with tag params
        og = -np.matmul(Rw, tw)
        orig = np.matmul(Rt, og) + t

        # Global origin in camera coord system with cam params
        origC = np.matmul(R.T, -p)

        tag_og.append(og)
        tag_Rts.append(Rt)
        tag_tts.append(t)

        cam_R += R
        cam_t += p
        '''
        # Drawing tag centers to the image
        rvec, _ = cv2.Rodrigues(R.T)
        io, _ = cv2.projectPoints(pos_t-tw, rvec, -origC, K, dist)

        io = io[0][0]
        iox = int(io[0])
        ioy = int(io[1])
        img = cv2.circle(img, (iox, ioy), radius=10, color=(0, 0, 255), thickness=-1)
        '''
        
    Rcm = cam_R/tag_count
    Rcm, _ = cv2.Rodrigues(Rcm)
    tcm = cam_t/tag_count
    # Initial camera extrinsic
    init_extrinsic = np.concatenate((Rcm.T, tcm.T), axis=0).reshape(-1)

    args = (tag_Rts, tag_tts, tag_og)
    # Optimize camera extrinsic with Levenberg-Marquardt method
    err, extr, _ = LMA.LM(init_extrinsic, args, model_func, kmax=100)
    
    rvo = np.array([extr[0], extr[1], extr[2]]).astype(np.float32)
    Ro, _ = cv2.Rodrigues(rvo)
    to = np.array([[extr[3], extr[4], extr[5]]]).T.astype(np.float32)
    
    # World origin in camera coord system
    po = np.matmul(Ro.T, -to)
    
    rvec, _ = cv2.Rodrigues(Ro.T)
    # Draw world origin to the image with optimized params
    io, _ = cv2.projectPoints(to, rvec, -po, K, dist)
    io = io[0][0]
    iox = int(io[0])
    ioy = int(io[1])
    image = cv2.circle(img, (iox, ioy), radius=10, color=(0, 0, 255), thickness=-1)
    plt.imshow(image)

    plt.show()
