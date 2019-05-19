# Description
This script rotates images automatically based on image content.
Therefor it currently uses head rotations to calculate the image rotation.

# Dependencies
To run this script you need to have opencv, imutils, numpy and dlib installed.

Install them using pip:
```bash
$ pip install opencv-python imutils numpy dlib
```

This script uses the [68 point face landmarks predictor model](https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) to find face positions.

Download this model by running the following command from the directory from which you call the script:
``` bash
$ wget https://github.com/AKSHAYUBHAT/TensorFace/raw/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat
```

# How To
After installing all dependencies and downloading the face landmarks model you can simply run `python image_rotator.py -i path/to/input/image.png -o path/to/output/image.jpg` to calculate the input images rotation and save the (hopefully) correctly rotated output image.
