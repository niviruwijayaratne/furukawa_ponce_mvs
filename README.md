# MVS

Implementation of "Accurate, Dense, and Robust Multi-View Stereopsis" from CVPR 2007.

### Matching Algorithm
**Currently Implemented** <br>
- Harris (Left) and DOG (Middle) Feature Detection. After detecting Harris and DOG features in image, 32px x 32px grid overlayed onto image and top 4 highest responses of each feature type are kept (Right), Harris = Red, DOG = Blue. <br><br>
<img src=visualizations/temple/harris_features_0.png height=400>&nbsp;&nbsp;&nbsp;
<img src=visualizations/temple/dog_features_0.png height=400>&nbsp;&nbsp;&nbsp;
<img src=visualizations/temple/max_responses_0.png height=400>&nbsp;&nbsp;&nbsp;

- Epipolar Correspondences. Fundamental matrix calculated using given camera matrices. For each feature in image I, for all images J != I, features of the same type that lie within 2 pixels of corresponding epipolar line are kept. Alternatively, can search window along epipolar line and use SSD to match. <br><br>
  <img src=visualizations/temple/epipolar_points.gif height=400>&nbsp;&nbsp;&nbsp;
  <img src=visualizations/temple/epipolar_matching.gif height=400>&nbsp;&nbsp;&nbsp;