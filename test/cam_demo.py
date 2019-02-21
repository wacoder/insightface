import face_model
import argparse
import numpy as np
import cv2
import sys

parser = argparse.ArgumentParser(description="face model test")
parser.add_argument("--image-size", default='112,112', help='')
parser.add_argument('--model', default='', help='path to model')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='MTCNN detection 1 means only R+O, 0 means from the beginning')
parser.add_argument('--flip', default=0, type=int, help="does the learning do flip augmentation")
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

model = face_model.FaceModel(args)