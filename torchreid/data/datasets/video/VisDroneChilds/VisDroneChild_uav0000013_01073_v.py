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
class VisDroneChild_uav0000013_01073_v(VisDroneParent):
    #dataset_dir = 'VisDrone2019-MOT\\'#incorrect

    def __init__(self, root='',gallery_query_ratio=6, **kwargs):
        
        self.video_name='uav0000013_01073_v'

        super(VisDroneChild_uav0000013_01073_v, self).__init__(root, gallery_query_ratio, **kwargs)
        
    