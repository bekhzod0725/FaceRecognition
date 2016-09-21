"""
    Load data from Lookalike Database
    
    Author: Bekhzod Umarov <bumar1@unh.newhaven.edu>
    Course: Artificial Intelligence
    Year:   2016
"""

from os import listdir, makedirs, remove
from os.path import join, exists, isdir

from sklearn.utils import deprecated

import logging
import numpy as np

from sklearn.datasets.base import get_data_home, Bunch
from sklearn.externals.joblib import Memory
from sklearn.externals.six import b

logger = logging.getLogger(__name__)
path = 'Lookalike_Final_Publish_v6.0/1_Classwise_separated'
def _get_targets():
    dirs = listdir(path)
    names = []
    fpaths = []
    for d in dirs:
        name = d[3:] #int(d[:2]), d[3:]
        for type in listdir( join(path, d) ):
            for image in listdir( join(path, d, type) ):
                image_path = join(path, d, type, image)
                names.append(name + ' (' + type + ')')
                fpaths.append(image_path)
    return np.array(names), np.array(fpaths)

def _load_imgs(file_paths, names):
    try:
        from scipy.misc import imread
        from scipy.misc import imresize
    except ImportError:
        raise ImportError("The Python Imaging Library (PIL)"
                "is required to load data from jpeg files")

    slice_ = (slice(0, 150), slice(0, 150))
    h_slice,w_slice = slice_
    h = (h_slice.stop - h_slice.start) // (h_slice.step or 1)
    w = (w_slice.stop - w_slice.start) // (w_slice.step or 1)

    resize = 0.4
    h = int(h*resize)
    w = int(w*resize)

    n_faces = len(file_paths)
    faces = np.zeros((n_faces, h, w), dtype=np.float32)
    targets = np.zeros((n_faces, 1), dtype='|S32')
    for i, file_path in enumerate(file_paths):
        img = imread(file_path)
        if img.ndim is 0:
            raise RuntimeError("Failed to read the image file %s,"
                    "Please make sure that libjpeg is installed"
                    % file_path)
        _h,_w,_c = img.shape
        imsize = _h if (_h <= _w) else _w
        resize_percentage = 150.0/imsize + 0.005
        img = imresize(img, resize_percentage)

        _h,_w,_c = img.shape
        _h_shift = _w_shift = 0
        if _w > _h:
            _w_shift = (_w - 150)//2
        elif _h > _w:
            _h_shift = (_h - 150)//2

        slice_ = (slice(2*_h_shift, 150 + 2*_h_shift), slice(_w_shift, 150 + _w_shift))

        face = np.asarray(img[slice_], dtype=np.float32)
        face /= 255.0
        face = imresize(face, resize)
        face = face.mean(axis=2)
        faces[i, ...] = face
        targets[i, ...] = names[i]
    return faces, targets

def _get_lookalike_people():
    names, file_paths = _get_targets()
    faces, targets = _load_imgs( file_paths, names)

    c_id = -1
    target_names = []
    target = []
    for i in range(len(targets)):
        if targets[i] not in target_names:
            target_names.append(targets[i])
            c_id += 1
        target.append(c_id)

    target_names = np.array(target_names)
    target = np.array(target)

    return faces, target_names, target

def get_lookalike_people():
    m = Memory(cachedir='./cache_data', compress=6, verbose=0)
    load_func = m.cache(_get_lookalike_people)

    #faces, targets, target_ids = _get_lookalike_people()
    faces, targets, target_ids = load_func()

    return Bunch( data=faces.reshape(len(faces), -1), 
                  images=faces, 
                  target=target_ids, 
                  target_names=targets,
                  DESCR="Look Alike People Dataset")
