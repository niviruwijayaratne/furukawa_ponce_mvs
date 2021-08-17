import os
import numpy as np
import cv2

def read_camera_params(image_dir):
    files = os.listdir(image_dir)
    camera_params = None
    lat_lon_params = None
    for file in files:
        if file.endswith('par.txt'):
            camera_params = os.path.join(image_dir, file)
        if file.endswith('ang.txt'):
            lat_lon_params = os.path.join(image_dir, file)

    with open(camera_params, 'r') as f:
        camera_params = f.readlines()[1:]
    
    with open(lat_lon_params, 'r') as f:
        lat_lon_params = f.readlines()
    
    cam_matrices = []
    for i, l in enumerate(camera_params):
        cam_matrices.append({})
        line = l.split(" ")
        lat_lon = lat_lon_params[i].split(" ")
        K = np.array(line[1:10]).reshape(3, 3).astype(np.float32)
        R = np.array(line[10:19]).reshape(3, 3).astype(np.float32)
        T = np.array(line[19:]).reshape(3, 1).astype(np.float32)

        cam_matrices[i]['K'] = K
        cam_matrices[i]['R'] = R
        cam_matrices[i]['T'] = T
        cam_matrices[i]['P'] = K.dot(np.hstack([R, T]))
        cam_matrices[i]['rot_angle'] = cv2.ROTATE_90_COUNTERCLOCKWISE if float(lat_lon[0]) < 0 else cv2.ROTATE_90_CLOCKWISE
        
    return cam_matrices

def skew_symmetric(vec):
    mat = np.zeros((3, 3))
    mat[0, 1] = -vec[2]
    mat[1, 0] = vec[2]

    mat[0, 2] = vec[1]
    mat[2, 0] = -vec[1]

    mat[1, 2] = -vec[0]
    mat[2, 1] = vec[0]

    return mat

def reshape_max_responses(features):
    dims = (features.shape[0], features.shape[1]*features.shape[2])
    reshaped_features = np.zeros(dims)
    reshaped_features[:, 0::2] = features[:, 0]
    reshaped_features[:, 1::2] = features[:, 1]
    reshaped_features = reshaped_features.reshape(-1, 2).astype(np.int32)
    reshaped_features = np.flip(reshaped_features, axis=1)

    return reshaped_features

    
    
        