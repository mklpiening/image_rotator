from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
    help="path to input image")
ap.add_argument("-o", "--output", required=True,
    help="path to output image")
args = vars(ap.parse_args())


def draw_facial_landmarks(coords):
    # chib
    for i in range(0, 16):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
 
    # left eyebrow
    for i in range(17, 21):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
 
    # right eyebrow
    for i in range(22, 26):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
 
    # nose
    for i in range(27, 30):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
 
    for i in range(31, 35):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
 
    # left eye
    for i in range(36, 41):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[36][0], coords[36][1]), (coords[41][0], coords[41][1]), (0, 0, 255), 2)

    # right eye
    for i in range(42, 47):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[42][0], coords[42][1]), (coords[47][0], coords[47][1]), (0, 0, 255), 2)

    # upper lip
    for i in range(48, 54):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    for i in range(60, 64):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[48][0], coords[48][1]), (coords[60][0], coords[60][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[54][0], coords[54][1]), (coords[64][0], coords[64][1]), (0, 0, 255), 2)

    # bottom lip
    for i in range(54, 59):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    for i in range(64, 67):
        cv2.line(image, (coords[i][0], coords[i][1]), (coords[i + 1][0], coords[i + 1][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[48][0], coords[48][1]), (coords[59][0], coords[59][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[54][0], coords[54][1]), (coords[64][0], coords[64][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[48][0], coords[48][1]), (coords[60][0], coords[60][1]), (0, 0, 255), 2)
    cv2.line(image, (coords[60][0], coords[60][1]), (coords[67][0], coords[67][1]), (0, 0, 255), 2)

def get_eye_positions(coords):
    left = np.zeros(2, dtype=int)
    right = np.zeros(2, dtype=int)

    for (x, y) in coords[36:42]:
        left[0] += x
        left[1] += y
    
    for (x, y) in coords[42:48]:
        right[0] += x
        right[1] += y
    
    left = (left / 6).astype(int)
    right = (right / 6).astype(int)

    return (left[0], left[1]), (right[0], right[1])

def get_nose_position(coords):
    nose = np.zeros(2, dtype=int)

    for (x, y) in coords[27:36]:
        nose[0] += x
        nose[1] += y
    
    nose = (nose / 9).astype(int)

    return (nose[0], nose[1])

def get_mouth_position(coords):
    mouth = np.zeros(2, dtype=int)

    for (x, y) in coords[48:68]:
        mouth[0] += x
        mouth[1] += y
    
    mouth = (mouth / 20).astype(int)

    return (mouth[0], mouth[1])

def get_labial_angle(coords):
    return (coords[48][0], coords[48][1]), (coords[54][0], coords[54][1])
    

def angle(u, v):
    u = np.array(u)
    u = u / np.linalg.norm(u)
    v = np.array(v)
    v = v / np.linalg.norm(v)

    ang = math.acos(u.dot(v) / (np.sqrt((u * u).sum() * np.sqrt((v * v).sum()))))

    if v[1] * u[0] - u[1] * v[0] < 0:
        ang = -ang

    return ang

def rotate(v, angle):
    cs = math.cos(angle)
    sn = math.sin(angle)

    px = v[0] * cs - v[1] * sn; 
    py = v[0] * sn + v[1] * cs;

    return (px, py)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

original = cv2.imread(args['input'])
original = imutils.resize(original, width=500)

faces_angle = 0
amount_faces = 0

for rotation in np.arange(0, 2 * math.pi, math.pi / 4):

    image = imutils.rotate_bound(original, rotation * 180 / math.pi)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=int)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        left_eye, right_eye = get_eye_positions(coords)
        nose = get_nose_position(coords)
        mouth = get_mouth_position(coords)
        left_labial_angle, right_labial_angle = get_labial_angle(coords)

        ang = (angle(rotate((1, 0), rotation), np.subtract(np.array(right_eye), np.array(left_eye)))
                + angle(rotate((1, 0), rotation), np.subtract(np.array(right_labial_angle), np.array(left_labial_angle)))
                + angle(rotate((0, 1), rotation), np.subtract(np.array(mouth), np.array(nose)))) / 3

        if amount_faces == 0 or abs(faces_angle / amount_faces - ang) < math.pi / 4:
            faces_angle += ang
            amount_faces += 1

image_angle = 0
if amount_faces > 0:
    image_angle = faces_angle / amount_faces
if image_angle < 0:
    image_angle += 2 * math.pi

image = cv2.imread(args['input'])

if image_angle > math.pi / 4 and image_angle <= 3 * math.pi / 4:
    print('rotate 270°')
    image = imutils.rotate_bound(image, 270)
elif image_angle > 3 * math.pi / 4 and image_angle <= 5 * math.pi / 4:
    print('rotate 180°')
    image = imutils.rotate_bound(image, 180)
elif image_angle > 5 * math.pi / 4 and image_angle <= 7 * math.pi / 4:
    print('rotate 90°')
    image = imutils.rotate_bound(image, 90)
else:
    print('correct orientation')

cv2.imwrite(args['output'], image)