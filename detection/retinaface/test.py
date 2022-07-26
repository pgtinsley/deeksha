import cv2
import sys
import numpy as np
import datetime
import os
import glob
from retinaface import RetinaFace

import argparse

thresh = 0.8
scales = [1024, 1980]

count = 1

gpuid = 0
detector = RetinaFace('./R50', 0, gpuid, 'net3')

# fnames = glob.glob('./images/*.png')
# fnames = glob.glob('/afs/crc.nd.edu/user/p/ptinsley/magface/inference/MPIE/**/**/*.png')

fnames = sorted(glob.glob('/scratch365/ptinsley/results-truncation-sg3/**/*.png'))

print('Found {} images...'.format(len(fnames)))

# num_found = 0

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--i', type=int,
                    help='an integer')
args = parser.parse_args()

j = args.i

for fname in fnames[(j-1)*30000:(j)*30000]:

    img = cv2.imread(fname)
    
    faces, _ = detector.detect(img, thresh, do_flip=False)
#     print(fname, faces.shape)

    if faces is not None:
#         num_found += 1
#         print('find', faces.shape[0], 'faces')
        for i in range(faces.shape[0]):
#             print('score', faces[i][4])
            box = faces[i].astype(np.int)
#             #color = (255,0,0)
#             color = (0, 0, 255)
#             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 2)
            chip = img[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite('/scratch365/ptinsley/results-truncation-sg3-chips/{}'.format('-'.join(fname.split('/')[-2:])), cv2.resize(chip, (112,112)))
#             print(fname, chip.shape)
#             if landmarks is not None:
#                 landmark5 = landmarks[i].astype(np.int)
#                 #print(landmark.shape)
#                 for l in range(landmark5.shape[0]):
#                     color = (0, 0, 255)
#                     if l == 0 or l == 3:
#                         color = (0, 255, 0)
#                     cv2.circle(img, (landmark5[l][0], landmark5[l][1]), 1, color,
#                            2)

#     filename = './detector_test.jpg'
#     print('writing', filename)
#     cv2.imwrite(filename, img)

# print('Done processing images. Found {} faces.'.format(num_found))
print('Done processing images.')
