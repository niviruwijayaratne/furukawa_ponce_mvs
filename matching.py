import cv2
import numpy as np
import os
import helpers as hlp
import sys

from patch import Patch

IMAGE_CLASS = 'temple'
IMAGE_DIR = os.path.join("data", IMAGE_CLASS)
VIS_DIR = 'visualizations'

class Matcher:
    def __init__(self):
        self._lambda = 0.06
        self.sigma = 1
        self.beta2 = 32
        self.eta = 4
        self.camera_params = hlp.read_camera_params(IMAGE_DIR)

    def construct_gaussian(self, ksize: int, sigma: int = 1):
        '''
        Returns Gaussian filter with specified sigma, and both partial derivatives
        '''
        gaussian = lambda x, y: (1/(2*np.pi*np.square(sigma)))*np.exp(-(np.square(x) + np.square(y))/(2*np.square(sigma)))
        G = np.zeros((ksize, ksize))
        for x in range(-(ksize//2), ksize//2 + 1):
            for y in range(-(ksize//2), ksize//2 + 1):
                G[y + ksize//2, x + ksize//2] = gaussian(x, y)
        G /= np.sum(G) #normalize so that sum of gaussian = 1
        # ^^ can replace by cv2.getGaussianKernel()
        xs = ys = np.arange(ksize) - ksize//2

        xcoords = np.tile(xs, (ksize, 1))
        ycoords = np.tile(ys, (ksize, 1)).T

        Gx = np.multiply(G, xcoords/np.square(sigma))
        Gy = np.multiply(G, ycoords/np.square(sigma))

        return G, Gx, Gy
    
    def get_harris_response(self, im: np.ndarray) -> np.ndarray:
        '''
        Returns Harris Features
            Input: H x W x 3 original image
            Output: H x W x 1 Harris Features
        '''
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=self._lambda)
        harris = cv2.dilate(harris, None)
        harris[harris > 0.01*harris.max()] = 255
        harris[~(harris > 0.01*harris.max())] = 0
        return harris

    def get_dog_response(self, im: np.ndarray, sigma:int = 1) -> np.ndarray:
        '''
        Returns Difference of Gaussian Features
            Input: H x W x 3 original image
            Output: H x W x 1 DOG Features
        '''
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        ksize = 5
        G1, G1x, G1y = self.construct_gaussian(ksize, sigma=1)
        G2, G2x, G2y = self.construct_gaussian(ksize, sigma=np.sqrt(2))
        G = G1 - G2
        pad = ksize//2
    
        dog = np.zeros_like(im)
        im = np.pad(im, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

        for y in range(0, im.shape[0] - ksize):
            for x in range(0, im.shape[1] - ksize):
                window = im[y: y + ksize, x: x + ksize]
                convolved = np.sum(np.multiply(window, G))
                dog[y, x] = convolved

        return dog
        
    def filter_responses(self, im: np.ndarray, harris: np.ndarray, dog: np.ndarray, **kwargs) -> dict:
        '''
        Returns dictionary: keys = ['harris', 'dog'], values = coordinates (x, y) of respective detected features
            Input: H x W x N original image, H x W x 1 harris feature image, H x W x 1 dog feature image, 
            optional bool for visualizing detections
            Output: (H/self.beta2)*(W/self.beta2) x 2 x self.eta, dict[key] 
        '''


        max_responses = {'harris': [], 'dog': []}
        for y in range(0, im.shape[0], self.beta2):
            for x in range(0, im.shape[1], self.beta2):
                dimx = self.beta2 if x + self.beta2 <= im.shape[1] else im.shape[1] - x
                dimy = self.beta2 if y + self.beta2 <= im.shape[0] else im.shape[0] - y
                get_max = lambda x: np.argsort(x)[-self.eta:] #sort by max
                get_coords = lambda x, global_coords, dims: np.array([x//dims[0], x % dims[1]]) + global_coords
                global_coords = np.array([y, x]).reshape(-1, 1)
                dims = np.array([dimy, dimx]).reshape(-1, 1)

                if not harris is None:
                    harris_window = harris[y: y + dimy, x: x + dimx].flatten()
                    if not np.sum(harris_window) == 0:   
                        max_harris = np.apply_along_axis(get_coords, 0, get_max(harris_window), global_coords, dims)
                        max_responses['harris'].append(max_harris)

                if not dog is None:
                    dog_window = dog[y: y + dimy, x: x + dimx].flatten()
                    if not np.sum(dog_window) == 0:
                        max_dog = np.apply_along_axis(get_coords, 0, get_max(dog_window), global_coords, dims)
                        max_responses['dog'].append(max_dog)

                if 'visualize' in kwargs:
                    cv2.rectangle(im, (x, y), (x + dimx, y + dimy), (0, 255, 0), 1)
                    if not dog is None and not np.sum(dog_window) == 0:
                        for yc, xc in zip(max_dog[0], max_dog[1]):
                            cv2.circle(im, (xc, yc), 3, [255, 0, 0], -1)
                    if not harris is None and not np.sum(harris_window) == 0:
                        for yc, xc in zip(max_harris[0], max_harris[1]):
                            cv2.circle(im, (xc, yc), 3, [0, 0, 255], -1)

        if 'visualize' in kwargs:
            cv2.imshow('harris', cv2.rotate(harris, self.camera_params[kwargs['idx']]['rot_angle']))
            cv2.imshow('dog', cv2.rotate(dog, self.camera_params[kwargs['idx']]['rot_angle']))
            cv2.imshow('original', cv2.rotate(im, self.camera_params[kwargs['idx']]['rot_angle']))
            cv2.waitKey(0)

        if 'write' in kwargs:
            cv2.imwrite(os.path.join(VIS_DIR, IMAGE_CLASS, 'harris_features_' + str(kwargs['idx']) + ".png"), \
                cv2.rotate(harris, self.camera_params[kwargs['idx']]['rot_angle']))
            cv2.imwrite(os.path.join(VIS_DIR, IMAGE_CLASS, 'dog_features_' + str(kwargs['idx']) + ".png"), \
                cv2.rotate(dog, self.camera_params[kwargs['idx']]['rot_angle']))
            cv2.imwrite(os.path.join(VIS_DIR, IMAGE_CLASS, 'max_responses_' + str(kwargs['idx']) + ".png"), \
                cv2.rotate(im, self.camera_params[kwargs['idx']]['rot_angle']))

        
        if not harris is None:
            max_harris_responses = np.array(max_responses['harris'])
            max_responses['harris'] = hlp.reshape_max_responses(max_harris_responses) #x, y

        if not dog is None:
            max_dog_responses = np.array(max_responses['dog'])
            max_responses['dog'] = hlp.reshape_max_responses(max_dog_responses) #x, y

        return max_responses #x, y

    def get_fundamental_matrix(self, image_indices):
        '''
        Returns fundamental matrix between image_indices using given camera matrices
        '''

        P1 = self.camera_params[image_indices[0]]['P']
        P2 = self.camera_params[image_indices[1]]['P']

        U, S, V_t = np.linalg.svd(P1)
        c = V_t[-1]
        ep = P2.dot(c)
        F = hlp.skew_symmetric(ep).dot(P2).dot(np.linalg.pinv(P1))

        return F

    def ssd(self, feature, win_size, im1, im2, get_y):
        feature = [int(feature[0]), int(feature[1])]
        ssd_window = im1[feature[0] - win_size//2: feature[0] + win_size//2 + 1, feature[1] - win_size//2: feature[1] + win_size//2 + 1]
        min_ssd = float('inf')
        corresp_feature = None
        slope = (get_y(100) - get_y(0))/100
        if slope < 0:
            bounds = range(im1.shape[1], 1, -1)
            get_next = -1
        else:
            bounds = range(im1.shape[1])
            get_next = 1
        for x in bounds:
            y1 = int(get_y(x))
            if y1 < 0 or y1 >= im2.shape[0]:
                continue
            else:
                y2 = min(int(get_y(x + get_next)), im2.shape[0])
                for y in range(y1, y2):
                    window = im1[y - win_size//2: y + win_size//2 + 1, x - win_size//2: x + win_size//2 + 1]
                    has_neg = np.array([y - win_size//2, y + win_size//2, x - win_size//2, x + win_size//2])
                    if window.shape != ssd_window.shape or not (has_neg > 0).all():
                        continue
                    ssd = np.sum(np.square(ssd_window - window))
                    if ssd < min_ssd:
                        min_ssd = ssd
                        corresp_feature = [y, x]

        return corresp_feature

    def epipolar_correspondence(self, point, base_im_idx):
        '''
        Given a feature (either harris or dog) in image with index base_im_idx, returns dictionary 
        of features (of same type of given feature): key = image index, value = all features
        that lie within 2 pixels of corresponding epipolar line
        '''
        ims = os.listdir(IMAGE_DIR)
        ims = sorted(list(filter(lambda x: x.endswith(".png"), ims)))
        base_im = cv2.imread(os.path.join(IMAGE_DIR, ims[base_im_idx]))
        num_ims = len(ims)
        other_im_idxs = list(range(0, base_im_idx))
        other_im_idxs.extend(list(range(base_im_idx + 1, num_ims)))

        features = {}
        for idx in other_im_idxs:
            if idx == 10:
                features[idx] = []
                continue
            im = cv2.imread(os.path.join(IMAGE_DIR, ims[idx]))
            harris_responses = self.get_harris_response(im)
            dog_responses = self.get_dog_response(im)
            max_responses = self.filter_responses(im, harris_responses, dog_responses)
            harris_points = np.hstack([max_responses['harris'], np.ones_like(max_responses['harris'][:, 0]).reshape(-1, 1)]).reshape(-1, 3)
            F = self.get_fundamental_matrix([base_im_idx, idx])
            homog_point = np.squeeze(np.vstack([np.array(point).reshape(-1, 1), [1]])) 
            l = F.dot(homog_point).reshape(-1, 1)
            distances = harris_points.dot(l)/(np.linalg.norm(harris_points[:, :-1], axis=1).reshape(-1, 1))
            epipolar_consistent = harris_points[:, :-1][np.where(np.abs(distances) <= 2)[0]] #x, y
            features[idx] = np.squeeze(epipolar_consistent)
        return features # x, y

    def match(self, visualize=False):
        ims = os.listdir(IMAGE_DIR)
        ims = sorted(list(filter(lambda x: x.endswith(".png"), ims)))
        max_responses = {}
        feature_types = ['harris', 'dog']
        patches = {}
        for idx, fname in enumerate(ims): 
            patches[idx] = []
            other_im_idxs = list(range(0, idx))
            other_im_idxs.extend(list(range(idx + 1, len(ims))))

            im = cv2.imread(os.path.join(IMAGE_DIR, fname))
            harris_responses = self.get_harris_response(im)
            dog_responses = self.get_dog_response(im)
            max_responses[idx] = self.filter_responses(im, harris_responses, dog_responses)
            
            matched_features = {}
            matched_features[idx] = {}

            for i, point in enumerate(max_responses[idx]['harris']): #x, y
                epipolar_features = self.epipolar_correspondence(point, idx)
                key_list = [k for k, vals in epipolar_features.items() for v in vals]
                value_list = [list(v) for k, vals in epipolar_features.items() for v in vals]
                triangulated_distances = []
                
                f1 = np.vstack([point.reshape(-1, 1), [1]])
                P1 = self.camera_params[idx]['P']

                for im_idx, feature_list in epipolar_features.items():
                    print("HERE", im_idx)
                    for feature in feature_list:
                        f2 = np.vstack([np.array(feature).reshape(-1, 1), [1]]) #x, y
                        P2 = self.camera_params[im_idx]['P']
                        triangulated_distance = self.triangulate_points(f1, f2, P1, P2)
                        triangulated_distances.append(triangulated_distance)

                sorted_keys = [key for _, key in sorted(zip(triangulated_distances, key_list))]
                sorted_values = [val for _, val in sorted(zip(triangulated_distances, value_list))]
                sorted_triangulated_distances = sorted(triangulated_distances)

    def triangulate_points(self, p1, p2, P1, P2):
        A = np.zeros((4, 4))
        A[0, :] = p1[0]*P1[-1, :] - P1[0, :]
        A[1, :] = p1[1]*P1[-1, :] - P1[1, :]
        A[2, :] = p2[0]*P2[-1, :] - P2[0, :]
        A[3, :] = p2[1]*P2[-1, :] - P2[1, :]

        U, S, V_t = np.linalg.svd(A)
        P = V_t[-1]
        
        U, S, V_t = np.linalg.svd(P1)
        O = V_t[-1]

        return np.linalg.norm(P - O)

