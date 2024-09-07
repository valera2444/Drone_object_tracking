import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
import os

from utils import write_metrics_to_df, write_time_to_df, replace_cuda_with_cpu, make_all_numpy,\
 video_annot_txt_to_dataframe, motMetricsEnhancedCalculator, convert_array, usingPIL




from os import walk

import datetime
import cv2
from time import time
from time import sleep
import os.path as osp


from sort import Sort    # crashes kernel
#from deepsort.tracker import DeepSortTracker
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

import argparse


def make_predictions(model=None, tracker=None, dir_path_sequences=None, dest_path=None, img_size=None, NMS_iou=1,device=None):
    '''
    also writes video in dest
    input:
    model - model for predictions. MUST return predictions in formst like YOLOv8
    tracker - MOT technic. SORT or DeepSORT
    dir_path_sequences - string path to video sequence('D:\\Drone_object_tracking\\VisDrone2019-MOT-val\\sequences\\uav0000268_05773_v') as example
    dest_path - name of a file to write video in
    img_size - image size for YOLO inference
    NMS_iou - parameter for YOLO NMS
    return:
    preds - dataframe of predictions in MOT16 format
    str_res - string with description of time characteristics of alghorithm
    times - dataframe of time wasted for different parts of function
    '''
    # print(next(walk(dir_path_sequences)))
    # print(dir_path_sequences+'/'+next(walk(dir_path_sequences)))
    # print(dir_path_sequences)
    # print(dir_path_sequences+'\\'+next(walk(dir_path_sequences))[2][0])
    print('make predictions begin', dir_path_sequences)
    # print(list(walk(dir_path_sequences)))
    height, width = cv2.imread(dir_path_sequences + '/' + next(walk(dir_path_sequences))[2][0]).shape[:2]
    # height, width = cv2.imread(dir_path_sequences).shape[:2]
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

    try:
        os.remove("video.avi")
    except:
        pass

    video = cv2.VideoWriter(dest_path, fourcc, 20, (width, height))

    mot_tracker = tracker

    # acc = mm.MOTAccumulator(auto_id=True)
    # annots = video_annot_txt_to_dataframe('D:\\Drone_object_tracking\\VisDrone2019-MOT-val\\annotations\\uav0000086_00000_v.txt')
    columns = ['frame_number', 'obj_id', 'left_top_x', 'left_top_y', 'width', 'height', 'confidence', 'category',
               'truncation',
               'occlusion']

    preds = pd.DataFrame(columns=columns)
    # print(len(filenames))

    avg_yolo_time = 0.
    avg_predictor_time = 0.
    avg_frame_time_arr = []
    avg_yolo_time_arr = []
    avg_predictor_time = []
    avg_resizing_time = []
    avg_drawing_time = []
    avg_reading_time = []

    filenames = sorted(next(walk(dir_path_sequences), (None, None, []))[2])  # BUG may appear
    # print(filenames)
    # i = 0
    for idx, img_relative_path in enumerate(filenames):

        time_before_fps = time()
        path = dir_path_sequences + '/' + img_relative_path

        time_before_reading = time()
        # img = cv2.imread(path)    #Using this, time after - before is NEARLY equal for 928px, for 640 still different(at the begging of small inference time is 20 ms)
        img = usingPIL(path)
        time_after_reading = time()
        avg_reading_time.append((time_after_reading - time_before_reading) * 1000)

        time_before_resizing = time()

        time_after_resizing = time()
        avg_resizing_time.append((time_after_resizing - time_before_resizing) * 1000)

        time_before_YOLO = time()
        prediction = model(img, verbose=False, imgsz=img_size, iou=NMS_iou, agnostic_nms=True)  # INCORRECT!
        time_after_YOLO = time()
        boxes = prediction[0].boxes.xyxy.type(torch.IntTensor).to(device)
        scores = prediction[0].boxes.conf
        classes = prediction[0].boxes.cls.type(torch.IntTensor)
        scores = torch.unsqueeze(scores, 1)

        torch_detections = torch.cat((boxes, scores), dim=1)
        torch_detections_cpu = replace_cuda_with_cpu(torch_detections)
        numpy_detections = make_all_numpy(torch_detections_cpu)

        time_before_updating_tracker = time()

        if type(mot_tracker) == DeepSort:
            numpy_detections = np.concatenate((numpy_detections, np.expand_dims(classes.cpu().numpy(), axis=1)),
                                              axis=1)

            result_tracker = mot_tracker.update_tracks(convert_array(numpy_detections), frame=img)  # WARNING here
            results = []

            for res in result_tracker:
                if not res.is_confirmed():  # state != confirmed
                    # print(res.state)
                    continue

                reses = res.to_tlwh(orig=True)  # orig=True,orig_strict =True
                if reses is None:  # similar to repo. Как я понял это если этому треку нет соответсвующей ground truth bbox
                    # print('None')
                    # print(res.track_id)
                    continue
                arr = reses.tolist()  # this parameters enable to return only DETECTROS bboxs#orig=True,orig_strict = True
                arr.append(res.track_id)
                results.append(arr)



        else:
            result_tracker = mot_tracker.update(numpy_detections)
            results = result_tracker

        # results =numpy_detections
        time_after_updating_tracker = time()

        time_before_drawing_predictions = time()
        #if idx % 100 == 0:
            #print('numpy_detections', len(numpy_detections))
            #print('results', len(results))  # This less because of tentatie
        # break
        for res in results:
            x1, y1, x2, y2, obj_id = [int(a) for a in res]

            # print('make predictions drawing',img)
            img = cv2.rectangle(np.float32(img), (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            img = cv2.putText(np.float32(img), str(obj_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                              thickness=2)

            single_pred = [idx + 1, obj_id, x1, y1, x2 - x1, y2 - y1, 1, 1, 1, 1]
            s = pd.Series(single_pred, index=columns)
            preds = pd.concat([s.to_frame().T, preds], ignore_index=True, axis=0)

        time_after_drawing_predictions = time()
        video.write(np.uint8(img))  # about 30 seconds for uav0000086_00000_v
        time_after_fps = time()

        avg_drawing_time.append((time_after_drawing_predictions - time_before_drawing_predictions) * 1000)
        avg_frame_time_arr.append((time_after_fps - time_before_fps) * 1000)
        avg_yolo_time_arr.append(time_after_YOLO - time_before_YOLO)
        avg_predictor_time.append(time_after_updating_tracker - time_before_updating_tracker)
        # img = image_resize(img, width=width)

    str_res = ''
    str_res += dir_path_sequences + '\n'
    str_res += f'Resized to {img_size}\n'
    str_res += f'image width:{width}, image height:{height}\n'
    str_res += f'Image reading time ms: {sum(avg_reading_time) / len(avg_reading_time)}\n'
    str_res += f'Image resizing time ms: {sum(avg_resizing_time) / len(avg_resizing_time)}\n'
    str_res += f'Avarage Yolo time ms:{sum(avg_yolo_time_arr) / len(avg_yolo_time_arr) * 1000}\n'
    str_res += f'Avarage predictor time ms:{sum(avg_predictor_time) / len(avg_predictor_time) * 1000}\n'
    str_res += f'Drawing predictions time ms: {sum(avg_drawing_time) / len(avg_drawing_time)}\n'

    # print(len(preds))
    times = write_time_to_df(video_name=dir_path_sequences,
                             resized_to=img_size,
                             source_width=width,
                             source_height=height,
                             avg_reading_time=sum(avg_reading_time) / len(avg_reading_time),
                             avg_resizing_time=sum(avg_resizing_time) / len(avg_resizing_time),
                             avg_yolo_time=sum(avg_yolo_time_arr) / len(avg_yolo_time_arr) * 1000,
                             avg_predictor_time=sum(avg_predictor_time) / len(avg_predictor_time) * 1000,
                             avg_drawing_time=sum(avg_drawing_time) / len(avg_drawing_time),
                             avg_fps_time=sum(avg_frame_time_arr) / len(avg_frame_time_arr))

    return preds, str_res, times

def proccess_single_video(model=None, tracker=None, dir_path_source_sequences=None, dir_path_source_annotations=None,dest_path=None,
                          video_name=None, img_sz=None,NMS_iou=1, device=None):
    """
    proccesses one video: find time and accuracy metrics on given video

    input:
    dir_path_source_sequences - '/content/drive/MyDrive/VisDrone2019-MOT-val/sequences/'
    dir_path_source_annotations - '/content/drive/MyDrive/VisDrone2019-MOT-val/annotations/'
    video_name - name of video for both annots and sequences

    return:
    time_df - time results dataframe
    metrics_df - metrics results dataframe
    """
    print(video_name,img_sz)
    preds, res_str, time_df = make_predictions(model,tracker, dir_path_source_sequences+'/'+video_name,dest_path,img_sz,NMS_iou,device=device)

    annots = video_annot_txt_to_dataframe(f'{dir_path_source_annotations}/{video_name}.txt')

    MOT, metrics_df = motMetricsEnhancedCalculator(annots.to_numpy(), preds.to_numpy() )

    return time_df, metrics_df




def argument_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("src_seqs_path",
                        help="folder that contains sequences")

    parser.add_argument("src_annot_path",
                        help="folder that contains sequences")

    parser.add_argument("video_name",
                        help="video name")

    parser.add_argument("dest_path",
                        help="folder where output video and tracking metrics will be stored")

    parser.add_argument('device',
                        help="use cpu or gpu for both yolo and reid",
                        choices=['cpu', 'gpu'])

    parser.add_argument('model_type',
                        help="choose between ['DeepSORT', SORT]",
                        choices = ['SORT','DeepSORT'])

    parser.add_argument('img_size',
                        help="the image will be resized to img_size*img_size for YOLO")

    parser.add_argument('NMS_iou',
                        help="non-max suppression argument for YOLO")




    return parser.parse_args()




def main():
    args = argument_parsing()
    model = YOLO("YOLOv8x3n.pt")

    device = 'cuda' if args.device == 'gpu' else 'cpu'

    model.to(device)

    if args.model_type == 'SORT':
        tracker = Sort()

    else:
        tracker = DeepSort(max_age=50,
                       embedder='torchreid',
                       embedder_model_name='osnet_ain_x0_25',
                       #embedder_wts = 'log\\osnet_ain_x0_25-triplet-visdrone\\model\\model.pth.tar-135',
                       gating_only_position = True,
                       embedder_gpu=True if device == 'cuda' else False
                       )

    video_name = args.video_name
    dir_path_source_sequences_val = args.src_seqs_path
    dir_path_source_annotations_val = args.src_annot_path

    dest_path = f'{args.dest_path}/{video_name}.avi'

    time_df, metrics_df = proccess_single_video(model=model,
                                                tracker=tracker,
                                                dir_path_source_sequences=dir_path_source_sequences_val,
                                                dir_path_source_annotations=dir_path_source_annotations_val,
                                                dest_path=dest_path,
                                                video_name=video_name,
                                                img_sz=640,
                                                NMS_iou=0.3,
                                                device=device)  # этот параметр убирает только то что в одном классе, чтобы убирать в разных надо nms_agnostic. Он для YOLO

    time_df['clear_fps'] = time_df['avg_reading_time'] + time_df['avg_yolo_time'] + time_df['avg_predictor_time']

    result = pd.concat((time_df, metrics_df), axis=1, ignore_index=False)
    result = pd.concat((time_df, metrics_df), axis=1, ignore_index=False)
    result.to_csv(f'{args.dest_path}/results.csv', sep=';', index=False)
    print(result)

"""
DeeepSort predicts bounding boxes which yolo doesnt
при дипсорте много перескакиъивает на соседей -
плохой ReID может(потому что выбирается только из тех что подходят по sort, а это соседи)
Наверное сопровождает то что утеряно на протяжении max age
Параметры (orig=True,orig_strict =True) позволяют не выдавать то что предсказано калманом
"""

if __name__ == '__main__':
    #print('SLEEP for 5 MINUTES')
    #sleep(300)
    print('cuda availability:',torch.cuda.is_available())
    main()