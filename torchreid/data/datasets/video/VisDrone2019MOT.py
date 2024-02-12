from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import sys
import os
import os.path as osp
import numpy as np

from .. import VideoDataset
#REMADE ID'S !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#transform shouldnt be None
class VisDrone2019MOT(VideoDataset):
    #dataset_dir = 'VisDrone2019-MOT\\'#incorrect

    def __init__(self, root='',video_name='',gallery_query_ratio=6, **kwargs):
        self.root = root    #D:/Drone_object_tracking/reid-data
        self.video_name = video_name
        #self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_paths, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        train = self._prepare_train()
        query, gallery = self._prepare_validation(gallery_query_ratio)
        #query = ...
        #gallery = ...

        super(VisDrone2019MOT, self).__init__(train, query, gallery, **kwargs)
        
    def _prepare_train(self):
        train_dir = 'VisDrone2019-MOT-train\\'
        abs_train_dir = self.root
        tracklets = []
        for idx in os.listdir(abs_train_dir):
            abs_idx = osp.join(abs_train_dir,idx)
            frames = sorted(os.listdir(abs_idx))
            frames_dirs = []
            for frame in frames:
                frames_dirs.append(osp.join(abs_idx, frame))
            tracklets.append([frames_dirs, int(idx), 0])
                
        
        return tracklets
    
    
    
    def _prepare_validation(self, gallery_query_ratio):
        """"
        selects 2 random videos and creates query and gallery sets, where: len(query):len(gallery) == 1 : gallery_query_ratio
        """
        val_dir = 'VisDrone2019-MOT-val\\'
        abs_val_dir = 'D:\\Drone_object_tracking\\reid-data\\VisDrone2019-MOT-val\\'
        tracklets = []
        for val_name in random.choices(os.listdir(abs_val_dir), k=2):
            abs_val_video_dir = osp.join(abs_val_dir, val_name)
            for idx in os.listdir(abs_val_video_dir):
                abs_idx = osp.join(abs_val_video_dir,idx)
                frames = sorted(os.listdir(abs_idx))
                frames_dirs = []
                for frame in frames:
                    frames_dirs.append(osp.join(abs_idx, frame))
                #print(frames_dirs)
                tracklets.append([frames_dirs, int(idx), 0])
                    
        query_idxs = random.sample(range(0, len(tracklets)), int(len(tracklets)/6))
        query = []
        for i in query_idxs:
            query.append(tracklets[i])
            
        gallery = []
        for i in range(len(tracklets)):
            if i not in query_idxs:
                gallery.append(tracklets[i])
        
        return query, gallery