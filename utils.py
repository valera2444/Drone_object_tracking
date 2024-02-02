import cv2
import pandas as pd
import numpy as np
import torch
import motmetrics as mm
from PIL import Image
def write_time_to_df(video_name=None, resized_to=None, source_width=None, source_height=None, avg_reading_time=None,
                     avg_resizing_time=None,avg_yolo_time=None, avg_predictor_time=None,avg_drawing_time=None,avg_fps_time=None):
    """
    util function for writing time charateristcs in Datafarme
    params:

    return:
    dataframe with time characterics
    """
    columns=['video_name','resized_to', 'source_width','source_height','avg_reading_time','avg_resizing_time',
             'avg_yolo_time','avg_predictor_time','avg_drawing_time','avg_fps_time']
    dataframe = pd.DataFrame(columns=columns)


    single_note = [video_name,resized_to, source_width,source_height,avg_reading_time,avg_resizing_time,
             avg_yolo_time,avg_predictor_time,avg_drawing_time,avg_fps_time]
    s = pd.Series(single_note, index=columns)
    dataframe = pd.concat([s.to_frame().T, dataframe], ignore_index=True, axis=0)

    return dataframe

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """
    resizes image with saving aspect ratio
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def replace_cuda_with_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, dict):
        return {key: replace_cuda_with_cpu(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [replace_cuda_with_cpu(item) for item in data]
    else:
        return data

def make_all_numpy(obj):
    if isinstance(obj, torch.Tensor):
        return obj.numpy()
    elif isinstance(obj, (list, tuple)):
        return [make_all_numpy(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: make_all_numpy(value) for key, value in obj.items()}
    else:
        return obj
    
def video_annot_txt_to_dataframe(annot_path):
    """
    util function for transforming txt file ti dataframe
    input:
    annot_path - str path to annotations.txt which are in MOT16 format('D:\\Drone_object_tracking\\VisDrone2019-MOT-val\\annotations\\uav0000268_05773_v.txt') as example
    
    return:
    dataframe in MOT16 format
    """
    columns=['frame_index','target_id', 'left_top_x','left_top_y','width','height','score','category','truncation','occlusion']
    dataframe = pd.DataFrame(columns=columns)
    
    with open(annot_path) as f:
        for line in f.readlines():
    
            line = line[:-1]
            single_pred = [ int(s) for s in line.split(',') ]
            s = pd.Series(single_pred, index=columns)
            dataframe = pd.concat([s.to_frame().T, dataframe], ignore_index=True, axis=0)
    
    return dataframe

def write_metrics_to_df(num_frames=None, idf1=None, idp=None, idr=None, recall=None, precision=None, num_objects=None,
                        mostly_tracked=None,partially_tracked=None, mostly_lost=None, num_false_positives=None,
                        num_misses=None, num_switches=None,num_fragmentations=None, mota=None, motp=None ):

    """
    util function for writing metric charateristcs in Datafarme
    params:

    return:
    dataframe with metric characterics
    """
    columns=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ]
    dataframe = pd.DataFrame(columns=columns)
    single_note = [num_frames, idf1, idp, idr, recall, precision, num_objects, mostly_tracked,
                        partially_tracked, mostly_lost, num_false_positives, num_misses, num_switches,
                        num_fragmentations, mota, motp ]
    s = pd.Series(single_note, index=columns)
    dataframe = pd.concat([s.to_frame().T, dataframe], ignore_index=True, axis=0)

    return dataframe


def motMetricsEnhancedCalculator(gtSource, tSource):
    '''
    input:
    gtSource - numpy array of ground truth detections in MOT16 format
    tSource - numpy array of prediction detections in MOT16 format

    return:
    strsummary - string with results
    df - dataframe with results
    '''

  # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    gt = gtSource
    t = tSource
    #print('is number of frames in ground truth equal to predictions?:', len(np.unique(gt[:,0])) == len(np.unique(t[:,0])))
    #print(len(np.unique(gt[:,0])) , len(np.unique(t[:,0])))
  # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
        t_dets = t[t[:,0]==frame,1:6] # select all detections in t

        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                  t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                     'recall', 'precision', 'num_objects', \
                                     'mostly_tracked', 'partially_tracked', \
                                     'mostly_lost', 'num_false_positives', \
                                     'num_misses', 'num_switches', \
                                     'num_fragmentations', 'mota', 'motp' \
                                    ], \
                      name='acc')
    #print(type(summary))
    #print(summary.columns)
    #print(summary.iloc[0].tolist())

    strsummary = mm.io.render_summary(
      summary,
      #formatters={'mota' : '{:.2%}'.format},
      namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
               'precision': 'Prcn', 'num_objects': 'GT', \
               'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
               'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
               'num_misses': 'FN', 'num_switches' : 'IDsw', \
               'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
              }
    )
    #print(strsummary)

    df  = write_metrics_to_df(*summary.iloc[0].tolist())

    return strsummary, df
def convert_array(array):

    res = np.empty((0,3))
    for row in range(array.shape[0]):
      arr = np.array([[array[row, :4], array[row, 4],array[row,5] ]], dtype=object)
      res = np.vstack((res,arr))
      
    return res

def usingPIL(f):
    im = Image.open(f)
    return np.asarray(im)

