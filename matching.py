import cv2
import numpy as np
import os

class Matcher:
    def __init__(self):
        self._lambda = 0.06
        self.sigma = 1
        self.beta2 = 32
        self.eta = 4

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
        
    def filter_responses(self, im: np.ndarray, harris: np.ndarray, dog: np.ndarray, visualize:bool = False) -> dict:
        '''
        Returns dictionary: keys = ['harris', 'dog'], values = coordinates of respective detected features
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

                harris_window = harris[y: y + dimy, x: x + dimx].flatten()
                max_harris = np.apply_along_axis(get_coords, 0, get_max(harris_window), global_coords, dims)
                dog_window = dog[y: y + dimy, x: x + dimx].flatten()
                max_dog = np.apply_along_axis(get_coords, 0, get_max(dog_window), global_coords, dims)
                
                max_responses['harris'].append(max_harris)
                max_responses['dog'].append(max_dog)
                if visualize:
                    cv2.rectangle(im, (x, y), (x + dimx, y + dimy), (0, 255, 0), 1)
                    for yc, xc in zip(max_dog[0], max_dog[1]):
                        cv2.circle(im, (xc, yc), 2, [255, 0, 0], -1)
                    
                    for yc, xc in zip(max_harris[0], max_harris[1]):
                        cv2.circle(im, (xc, yc), 2, [0, 0, 255], -1)
        if visualize:
            cv2.imshow('h', harris)
            cv2.imshow('d', dog)
            cv2.imshow('i', im)
            cv2.waitKey(0)
        
        max_responses['harris'] = np.array(max_responses['harris'])
        max_responses['dog'] = np.array(max_responses['dog'])

        return max_responses
