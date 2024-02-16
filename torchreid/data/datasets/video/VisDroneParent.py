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
TRACKLET_LENGTH = 15
class VisDroneParent(VideoDataset):
    #dataset_dir = 'VisDrone2019-MOT\\'#incorrect

    def __init__(self, root='',gallery_query_ratio=6, **kwargs):
        self.root = root    #D:/Drone_object_tracking/reid-data
        
       

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
        if self.__class__.query_idxs == None:
            self.__class__.init_idxs(len(train))
            
        query, gallery = self._prepare_validation(gallery_query_ratio)
        #query = ...
        #gallery = ...

        super(VisDroneParent, self).__init__(train, query, gallery, **kwargs)
        
    def _prepare_train(self):
        print('self.video_name',self.video_name)
        self.abs_train_dir = osp.join(self.root, self.video_name)
        tracklets = []
        for idx in os.listdir(self.abs_train_dir):
            abs_idx = osp.join(self.abs_train_dir,idx)
            frames = sorted(os.listdir(abs_idx))
            frames_dirs = []
            for frame_idx, frame in enumerate(frames):
                
                frames_dirs.append(osp.join(abs_idx, frame))
                if (( frame_idx+1) % TRACKLET_LENGTH) == 0:  
                    
                    tracklets.append([frames_dirs, int(idx), 0])
                    frames_dirs = []
            
        #print('tracklets')
        #print(tracklets)
        return tracklets
    
    
    
    def _prepare_validation(self, gallery_query_ratio):
        #validation on same which train
        """"
        selects 2 random videos and creates query and gallery sets, where: len(query):len(gallery) == 1 : gallery_query_ratio
        """
        #val_dir = 'VisDrone2019-MOT-val\\'
        #abs_val_dir = 'D:\\Drone_object_tracking\\reid-data\\VisDrone2019-MOT-val\\'
        tracklets = []
        
        for idx in os.listdir(self.abs_train_dir):
            abs_idx = osp.join(self.abs_train_dir,idx)
            frames = sorted(os.listdir(abs_idx))
            frames_dirs = []
            for frame_idx, frame in enumerate(frames):
                
                frames_dirs.append(osp.join(abs_idx, frame))
                if( (frame_idx + 1) % TRACKLET_LENGTH ) == 0:        
                    tracklets.append([frames_dirs, int(idx), 0])
                    frames_dirs = []
        
       
        query_idxs = self.__class__.query_idxs
        query = []
        
        for i in query_idxs:
            
            query.append(tracklets[i])
            
            
        gallery = []
        #print('gallery')
        for i in range(len(tracklets)):
            if i not in query_idxs:
                gallery.append(tracklets[i])
                #print(i, end=' ')
                
                
        gtr = []
        for track in gallery:
            #print('track', track[0])
            gtr += track[0]
            
        qtr = []
        for track in query:
            #print('track', track[0])
            qtr += track[0]

        """print('len(gtr)',len(gtr))
        print('len(set(gtr))',len(set(gtr)))
        print('len(qtr)',len(qtr))
        print('len(set(qtr))',len(set(qtr)))

        print('VisDrone Parent join',set(qtr).intersection(set(gtr)))"""
        
        random.shuffle(gallery)
        return query, gallery#no inersection
    
    
"""
gtr = []
for track in gallery:
    #print('track', track[0])
    gtr += track[0]
    
qtr = []
for track in query:
    #print('track', track[0])
    qtr += track[0]

print('len(gtr)',len(gtr))
print('len(set(gtr))',len(set(gtr)))
print('len(qtr)',len(qtr))
print('len(set(qtr))',len(set(qtr)))

print('join',set(qtr).intersection(set(gtr)))"""