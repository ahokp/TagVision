# Author: Paul Ahokas
# This file contains a demonstration of how to use data returned
# by handleData.readData.

import handleData
import os
import cv2

Root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
filename = 'example'
savefile = f'{Root_dir}/TagVision/savedData/{filename}.pkl'
data_dict = handleData.readData(savefile)

rgb = data_dict['rgb']
depth = data_dict['depth']

example_obj_ids = ['luvil', 'box', 'scale', 'shoe']

frame_count = len(rgb)
# Go through each frame
for i in range(frame_count):
    rgb_frame = rgb[i]
    depth_frame = depth[i]

    for id in example_obj_ids:
        # This is a 4x4 matrix
        frame_object_pose = data_dict[id][i]

        # Get rotation matrix and translation vector
        rotation = frame_object_pose[:3, :3]
        translation = frame_object_pose[:3, 3]

        # Do stuff with rotation and translation...

    # Do stuff with rgb_frame and depth_frame...

    # Show each RGB frame for 2 seconds
    cv2.imshow('Frame', rgb_frame)
    key = cv2.waitKey(2000)
    if key == 27:
        cv2.destroyAllWindows()
        break