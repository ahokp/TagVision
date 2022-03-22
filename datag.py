import apriltag as at
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from libs import LMA
import pyrealsense2 as rs
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
    """ Model function
    Note: This function is used as an optimization/error function in
    Levenberg-Marquardt/trf algorithm.
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
    
    #Rt, tt, og = args
    residuals = []
    for i in range(len(Rt)):
        d = np.linalg.norm(np.matmul(Rt[i], og[i]) + tt[i] + np.matmul(R_inv, t))
        residuals.append(d)
    
    return np.array(residuals)

def draw_pose(rvec, tvec, K, dist, img):
    PADDING = 200
    max_size = max(img.shape) + PADDING
    # Draw object center in image
    oic, _ = cv2.projectPoints(np.zeros((3, 1)), rvec, tvec, K, dist)
    oic = oic[0][0]
    oicx = int(oic[0])
    oicy = int(oic[1])

    # Don't draw pose if out of picture
    if oicx > max_size or oicy > max_size:
        pass
    elif oicx < -max_size or oicy < -max_size:
        pass
    else:
        cv2.circle(img, (oicx, oicy), radius=5, color=(0, 255, 0), thickness=-1)
            
        # Draw object xyz-axes 
        diag = np.eye(3)*0.1
        ax_pts, _ = cv2.projectPoints(diag, rvec, tvec, K, dist)
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

# World position of the objects
obj_ts = [np.array([[0.11], [0.342], [0]]),
          np.array([[0.455], [0.553], [0]]),
          np.array([[0.32], [1.001], [0]]),
          np.array([[0.84], [0.767], [0]])]
# Orientation of the objects with respect to the world coord system
obj_w_oris = [[0, 0, np.pi*5/4],
              [0, np.pi/2, 0],
              [0, 0, -np.pi/4],
              [0, 0, np.pi/2]]
# Create object-to-world rotations
obj_Rs = []
obj_count = len(obj_ts)
for i in range(obj_count):
    rot = obj_w_oris[i]
    obj_Rs.append(create_world_rotation_matrix(rot[0], rot[1], rot[2]))


tag_size = 0.161
# World position(xyz) of the tags
tag_ts = [np.array([[0.0805], [0.0805], [0]]),
          np.array([[0.13], [0.711], [0]]),
          np.array([[0.87], [1.068], [0]]),
          np.array([[0.611], [0.273], [0]])]

# Orientation of the tags in radian
tag_w_ori = [[0, 0, 0],
             [0, 0, -np.pi/4],
             [0, 0, 0],
             [0, 0, np.pi/4]]

tag_Rs = []
total_tags = len(tag_w_ori)
for i in range(total_tags):
    # Create tag-to-world rotations
    rot = tag_w_ori[i]
    tag_Rs.append(create_world_rotation_matrix(rot[0], rot[1], rot[2]))

# AprilTag detector
detector = at.apriltag('tag36h11')

# Camera params
K = np.array([])
dist = None

# Tag model points
obj_points = np.array([-1, -1, 0,
                        1, -1, 0,
                        1, 1, 0,
                        -1, 1, 0]).astype(np.float32).reshape(1, -1, 3)*tag_size/2

Root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


bagfile = f'{Root_dir}/TagVision/inputBag/scene.bag'

# Read .bag file saved by RealSense D435i
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, bagfile)
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
pipeline.start(config)

colorizer = rs.colorizer()

img_list = []

while True:
    
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    #print(depth_frame.get_distance(100, 100))

    #depth_color_frame = colorizer.colorize(depth_frame)
    #depth_color_image = np.asanyarray(depth_color_frame.get_data())
    
    color_image = np.asanyarray(color_frame.get_data())
    img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    img_size = (img_gray.shape[1], img_gray.shape[0])

    detections = detector.detect(img_gray)
    tag_count = len(detections)

    # No detections
    if tag_count == 0:
        cv2.imshow("Color stream", color_image)
        continue
    
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
    # Optimize camera extrinsics with Levenberg-Marquardt method
    #err, extr, _ = LMA.LM(init_extrinsic, args, model_func, kmax=100)

    # Optimize camera ectrinsics with Trust Region Reflective algorithm
    res_lsq = least_squares(model_func, init_extrinsic, args=args,
                            loss='soft_l1', f_scale=0.1, method='trf',
                            max_nfev=1)
    extr = res_lsq.x

    rvo = np.array([extr[0], extr[1], extr[2]]).astype(np.float32)
    Ro, _ = cv2.Rodrigues(rvo)
    to = np.array([[extr[3], extr[4], extr[5]]]).T.astype(np.float32)
    
    # World origin in camera coord system
    po = np.matmul(Ro.T, -to)
    
    rvec, _ = cv2.Rodrigues(Ro.T)
    # Draw world origin to the image with optimized params
    io, _ = cv2.projectPoints(np.zeros((3,1)), rvec, po, K, dist)
    io = io[0][0]
    iox = int(io[0])
    ioy = int(io[1])
    cv2.circle(color_image, (iox, ioy), radius=10, color=(255, 0, 0), thickness=-1)

    hvec = np.array([0, 0, 0, 1])
    # Camera extrinsics in global
    EGC = np.vstack((np.hstack((Ro.T, po)), hvec))

    for i in range(obj_count):
        RWO = obj_Rs[i]
        obj_t = obj_ts[i]

        # Object extrinsics in global
        EGO = np.vstack((np.hstack((RWO, obj_t)), hvec))

        # Object pose in camera coordinate system
        ECO = np.matmul(EGC, EGO)
        RCO = ECO[:3, :3].astype(np.float32)
        TCO = ECO[:3, 3].astype(np.float32)
        rco_vec, _ = cv2.Rodrigues(RCO)
        
        draw_pose(rco_vec, TCO, K, dist, color_image)

    cv2.imshow("Color stream", color_image)
    
    key = cv2.waitKey(1)
    if key == 27:
        cv2.destroyAllWindows()
        break
    
    #img = color_image.copy()
    #img_list.append(img)

# Save frames as gif
#imageio.mimsave(f'{Root_dir}/TagVision/outputGif/scene.gif', img_list, fps=20)

exit()