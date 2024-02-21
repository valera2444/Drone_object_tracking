# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:01:28 2024

@author: Asus
"""


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import random
import sys
import os
import os.path as osp
import numpy as np

from ..VisDroneParent import VisDroneParent
#REMADE ID'S !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#transform shouldnt be None
TRACKLET_LENGTH = 10
class VisDroneChild_uav0000339_00001_v(VisDroneParent):
    #dataset_dir = 'VisDrone2019-MOT\\'#incorrect
    query_idxs = None
    @staticmethod
    def init_idxs(size):
        
        VisDroneChild_uav0000339_00001_v.query_idxs = random.sample(range(0, size),k= size//6)
        
    def __init__(self, root='',gallery_query_ratio=6, **kwargs):
        
        self.video_name='uav0000339_00001_v'

        super(VisDroneChild_uav0000339_00001_v, self).__init__(root, gallery_query_ratio, **kwargs)
        
        