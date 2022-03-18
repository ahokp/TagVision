import apriltag as at
import cv2
import numpy as np
import os
from mpl_toolkits import mplot3d 
from matplotlib import pyplot as plt
from libs import LMA

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

# World position of the object
obj_t = np.array([[0.345], [0.2335], [0]])

# Orientation of the object
RWO = create_world_rotation_matrix(0, 0, np.pi/2)

# Corner points of the object
opoints = np.array([[0.06, -0.16, 0],
                    [0.06, 0.16, 0],
                    [-0.06, 0.16, 0],
                    [-0.06, -0.16, 0],
                    [0.06, -0.16, 0.11],
                    [0.06, 0.16, 0.11],
                    [-0.06, 0.16, 0.11],
                    [-0.06, -0.16, 0.11]]).astype(np.float32)

edges = np.array([0, 1,
                  1, 2,
                  2, 3,
                  3, 0,
                  0, 4,
                  1, 5,
                  2, 6,
                  3, 7,
                  4, 5,
                  5, 6,
                  6, 7,
                  7, 4]).reshape(-1, 2)

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

# Tag model points
obj_points = np.array([-1, -1, 0,
                        1, -1, 0,
                        1, 1, 0,
                        -1, 1, 0]).astype(np.float32).reshape(1, -1, 3)*tag_size/2

Root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

for n in range(15):
    imgpath = Root_dir + f'/TagVision/images/img{n+1}.png'
    img = cv2.imread(imgpath)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = (img.shape[1], img.shape[0])

    detector = at.apriltag('tag36h11')
    detections = detector.detect(img_gray)
    tag_count = len(detections)

    ipts = []
    opts = []
    ids = []

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
    #io, _ = cv2.projectPoints(to, rvec, -po, K, dist)
    io, _ = cv2.projectPoints(np.zeros((3,1)), rvec, po, K, dist)
    io = io[0][0]
    iox = int(io[0])
    ioy = int(io[1])
    cv2.circle(img, (iox, ioy), radius=10, color=(255, 0, 0), thickness=-1)


    hvec = np.array([0, 0, 0, 1])
    # Camera extrinsics in global
    EGC = np.vstack((np.hstack((Ro.T, po)), hvec))
    # Object extrinsics in global
    EGO = np.vstack((np.hstack((RWO, obj_t)), hvec))

    # Object pose in camera coordinate system
    ECO = np.matmul(EGC, EGO)
    RCO = ECO[:3, :3].astype(np.float32)
    TCO = ECO[:3, 3].astype(np.float32)
    rco_vec, _ = cv2.Rodrigues(RCO)
    
    # Draw object in image
    oic, _ = cv2.projectPoints(np.zeros((3, 1)), rco_vec, TCO, K, dist)
    oic = oic[0][0]
    oicx = int(oic[0])
    oicy = int(oic[1])
    cv2.circle(img, (oicx, oicy), radius=10, color=(0, 255, 0), thickness=-1)
    
    # Draw object xyz-axes 
    diag = np.eye(3)*0.15
    ax_pts, _ = cv2.projectPoints(diag, rco_vec, TCO, K, dist)
    ax_pts = ax_pts.astype(int)
    x_pt = ax_pts[0][0]
    y_pt = ax_pts[1][0]
    z_pt = ax_pts[2][0]
    cv2.line(img, (oicx, oicy), (x_pt[0], x_pt[1]),
             (255, 0, 0), 4, 16)
    cv2.line(img, (oicx, oicy), (y_pt[0], y_pt[1]),
             (0, 255, 0), 4, 16)
    cv2.line(img, (oicx, oicy), (z_pt[0], z_pt[1]),
             (0, 0, 255), 4, 16)

    # Drawing bounding box around the object
    ipoints, _ = cv2.projectPoints(opoints, rco_vec, TCO, K, dist)
    ipoints = ipoints.astype(int).reshape(-1, 2)
    for i, j in edges:
        start = ipoints[i]
        end = ipoints[j]
        cv2.line(img, (start[0], start[1]), (end[0], end[1]),
                 (0, 255, 0), 1, 16)
    
    plt.imshow(img)
    plt.show()

    '''
    # Save output images
    out_img_path = Root_dir + f'/TagVision/output/img{n+1}.png'
    cv2.imwrite(out_img_path, img)
    '''