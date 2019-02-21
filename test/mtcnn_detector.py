# coding: utf-8

import os
import mxnet as mx
import numpy as np
import math
import cv2
from multiprocessing import Pool
from itertools import repeat

try:
    from itertools import izip
except ImportError:
    izip = zip

#   MTCNN detector of faces
#   paper source "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks"
#   mxnet verison
class MtcnnDetector(object):
    def __init__(self,
                 model_folder='',
                 minsize=20,
                 threshold=[0.6,0.7,0.8],
                 factor=0.709,
                 num_worker=1,
                 accurate_landmark=False,
                 ctx=mx.cpu()):
        """
            Initialize the detector

            Parameters:
            -----------
                model_folder: string
                    directory where model file saved
                minsize: float number
                    minimal face can be detected
                threshold: list of float number
                    detect threshold for three stages
                factor: float number
                    scale factor for image pyramid
                num_worker: int number
                    the processes used for the first stage detection
                accurate_landmark: boolean
                    whethre to use accurate landmark deteciton
                ctx : mx device
                    where the computation conducted
        """
        self.num_worker = num_worker
        self.accurate_landmark = accurate_landmark

        # load 4 models from folder
        models = ['det1', 'det2', 'det3','det4']
        models = [ os.path.join(model_folder, f) for f in models]
        
        self.PNets = []
        for i in range(num_worker):
            workner_net = mx.model.FeedForward.load(models[0], 1, ctx=ctx)
            self.PNets.append(workner_net)

        #self.Pool = Pool(num_worker)

        self.RNet = mx.model.FeedForward.load(models[1], 1, ctx=ctx)
        self.ONet = mx.model.FeedForward.load(models[2], 1, ctx=ctx)
        self.LNet = mx.model.FeedForward.load(models[3], 1, ctx=ctx)

        self.minsize   = float(minsize)
        self.factor    = float(factor)
        self.threshold = threshold
    
    def convert_to_square(self, bbox):
        """
            convert bbox to square

            Parameters:
                bbox : numpy array, n*5
                    input bbox
            Returns:
                squares bbox
        """
        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        max_side = np.maximum(h,w)
        square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
        square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
        square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
        square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
        return square_bbox
    
    def calibrate_box(self, bbox, reg):
        """
            calibrate bboxes

        Parameters:
        ----------
            bbox: numpy array, shape n x 5
                input bboxes
            reg:  numpy array, shape n x 4
                bboxex adjustment

        Returns:
        -------
            bboxes after refinement

        """
        w = bbox[:, 2] - bbox[:, 0] + 1
        w = np.expand_dims(w, 1)
        h = bbox[:, 3] - bbox[:, 1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h, w, h])
        aug = reg_m * reg
        bbox[:, 0:4] = bbox[:, 0:4] + aug
        return bbox

 
    def pad(self, bboxes, w, h):
        """
            pad the the bboxes, alse restrict the size of it

        Parameters:
        ----------
            bboxes: numpy array, n x 5
                input bboxes
            w: float number
                width of the input image
            h: float number
                height of the input image
        Returns :
        ------s
            dy, dx : numpy array, n x 1
                start point of the bbox in target image
            edy, edx : numpy array, n x 1
                end point of the bbox in target image
            y, x : numpy array, n x 1
                start point of the bbox in original image
            ex, ex : numpy array, n x 1
                end point of the bbox in original image
            tmph, tmpw: numpy array, n x 1
                height and width of the bbox

        """
        tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1,  bboxes[:, 3] - bboxes[:, 1] + 1
        num_box = bboxes.shape[0]

        dx , dy= np.zeros((num_box, )), np.zeros((num_box, ))
        edx, edy  = tmpw.copy()-1, tmph.copy()-1

        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w-1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h-1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return  return_list

    def slice_index(self, number):
        """
            slice the index into (n,n,m), m < n
        Parameters:
        ----------
            number: int number
                number
        """
        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i + n]
        num_list = range(number)
        return list(chunks(num_list, self.num_worker))
