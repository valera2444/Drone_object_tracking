from __future__ import print_function, absolute_import
import numpy as np
import shutil
import os.path as osp
import cv2

from .tools import mkdir_if_missing

__all__ = ['visualize_ranked_results']

GRID_SPACING = 10
QUERY_EXTRA_SPACING = 90
BW = 5 # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(    #WORKS ONLY FOR MY VISDRONE
    distmat, dataset, data_type, width=128, height=256, save_dir='', topk=10
):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid, dsetid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    mkdir_if_missing(save_dir)

    print('# query: {}\n# gallery {}'.format(num_q, num_g))
    print('Visualizing top-{} ranks ...'.format(topk))

    query, gallery = dataset
    assert num_q == len(query)
    assert num_g == len(gallery)

    indices = np.argsort(distmat, axis=1)
    
    def get_last_two_dirs(path):
        # Split the path into directories
        dirs = path.split(osp.sep)
    
        # Extract the last two directories
        last_two_dirs = dirs[-2]+'_'+osp.splitext(dirs[-1])[0]
        
        return last_two_dirs

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == 'gallery':
                suffix = 'TRUE' if matched else 'FALSE'
                dst = osp.join(
                    dst, prefix + '_top' + str(rank).zfill(3)
                ) + '_' + suffix
            else:
                dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                print('img_path',img_path)
                #dst = dst + get_last_two_dirs(img_path)+'.jpg'
                f_name = osp.join(dst , get_last_two_dirs(img_path)+'.jpg')
                print('f_name',f_name)
                shutil.copy(img_path, f_name)
        else:
            dst = osp.join(
                dst, prefix + '_top' + str(rank).zfill(3) + '_name_' +
                osp.basename(src)
            )
            #file_name = 
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg_path, qpid, qcamid = query[q_idx][:3]
        qimg_path_name = qimg_path[0] if isinstance(
            qimg_path, (tuple, list)
        ) else qimg_path

        if data_type == 'image':
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols*width + topk*GRID_SPACING + QUERY_EXTRA_SPACING, 3
                ),
                dtype=np.uint8
            )
            grid_img[:, :width, :] = qimg
        else:
            #print('qimg_path_name',qimg_path_name)
        
            
            qdir = osp.join(
                save_dir, get_last_two_dirs(qimg_path_name)#osp.basename(osp.splitext(qimg_path_name)[0])
            )
            
            #print('get_last_two_dirs(qimg_path_name)',get_last_two_dirs(qimg_path_name))
            mkdir_if_missing(qdir)
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')

        rank_idx = 1
        #print('q_idx',q_idx)
        #print('indices[q_idx, :]',indices[q_idx, :]) # в indicies по индексу q_idx лежат инлексы галереи в порядке возрастания расстояний?
        for g_idx in indices[q_idx, :]:
            
            gimg_path, gpid, gcamid = gallery[g_idx][:3]
            print(f'rank_idx {rank_idx},{gpid},{qpid}')
            
            #invalid = (qpid == gpid) & (qcamid == gcamid)
            invalid = False
            if not invalid:
                matched = gpid == qpid
                if data_type == 'image':
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(
                        gimg,
                        BW,
                        BW,
                        BW,
                        BW,
                        cv2.BORDER_CONSTANT,
                        value=border_color
                    )
                    gimg = cv2.resize(gimg, (width, height))
                    start = rank_idx*width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    end = (
                        rank_idx+1
                    ) * width + rank_idx*GRID_SPACING + QUERY_EXTRA_SPACING
                    grid_img[:, start:end, :] = gimg
                else:
                    #print('rank_idx',rank_idx)
                    #print('gimg_path',gimg_path) #one tracklet from galery
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix='gallery',
                        matched=matched
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == 'image':
            imname = osp.basename(osp.splitext(qimg_path_name)[0])
            cv2.imwrite(osp.join(save_dir, imname + '.jpg'), grid_img)

        if (q_idx+1) % 100 == 0:
            print('- done {}/{}'.format(q_idx + 1, num_q))

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
