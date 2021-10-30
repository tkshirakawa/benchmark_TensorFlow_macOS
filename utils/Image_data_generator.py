'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, csv, math, time, cv2
import numpy as np
from joblib import Parallel, delayed

from tensorflow.keras.utils import Sequence     # TF2




def Pad_and_Resize(image, size):

    if image.shape[:2] == (size, size): return image
    else:                               height, width = image.shape[:2]

    if height == width:
        sq_image = image
    elif height > width:
        left = int((height - width) / 2.0)
        sq_image = cv2.copyMakeBorder(image, 0, 0, left, height-width-left, cv2.BORDER_CONSTANT, (0,0,0))
    elif height < width:
        top = int((width - height) / 2.0)
        sq_image = cv2.copyMakeBorder(image, top, width-height-top, 0, 0, cv2.BORDER_CONSTANT, (0,0,0))

    if (height * width) > (size * size) : return cv2.resize(sq_image, dsize=(size, size), interpolation=cv2.INTER_AREA)
    else                                : return cv2.resize(sq_image, dsize=(size, size), interpolation=cv2.INTER_LINEAR)




'''
    For TensorFlow 2.0
    Image data generator should be a subclass of keras.utils.Sequence that generates batch data.

    See https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
'''

# Read a list (first row for header, two column for data) from .csv file, and load images.
# Images must be gray-scale without alpha values.

class ImageDataGenerator_CSV_with_Header(Sequence):

    def __init__(self, data_name, data_file, input_shape, batch_size=32, rescale=1.0):

        self.data_name = data_name
        self.batch_size = batch_size
        self.rescale = np.array([rescale])

        with open(data_file, 'r', newline='', encoding='utf-8_sig') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)        # !!! First row is a header, skip !!!
            paths = [row for row in reader]

        img_size = input_shape[0]
        data_count = len(paths)
        data_shape = (data_count,) + input_shape

        print('Path              : {}'.format(data_file))
        print('Data name         : {}'.format(data_name))
        print('Input shape       : {}'.format(input_shape))
        print('CSV data counts   : {}'.format(data_count))
        print('Pixel scaling     : {}'.format(self.rescale))

        assert input_shape[0] == input_shape[1], 'Input image [{}x{}] must be square'.format(input_shape[0], input_shape[1])
        assert input_shape[2] == 1, 'Channel count [{}] must be 1 for input shape'.format(input_shape[2])

        for path_row in paths:
            assert os.path.isfile(path_row[0]), 'File lost: ' + path_row[0]
            assert os.path.isfile(path_row[1]), 'File lost: ' + path_row[1]


        time_start = time.perf_counter()

        print('Loading data from CSV file...')

        ncpu = os.cpu_count()
        parallel_count = max(int(data_count / (ncpu * 1000)), 1)
        total_subprocess = ncpu * parallel_count
        ss_list = [[]] * parallel_count
        for i in range(parallel_count):
            list_temp = []
            for j in range(ncpu):
                loc = ncpu * i + j
                ss1 = int(data_count * float(loc  ) / float(total_subprocess))
                ss2 = int(data_count * float(loc+1) / float(total_subprocess))
                list_temp.append((ss1, ss2))
            ss_list[i] = list_temp

        # When not use structure palette: images must be grayscale
        def _subprocess_for_XY_load(x_out, y_out, paths, ss1, ss2):
            for i in range(ss1, ss2):
                x_out[i] = Pad_and_Resize(cv2.imread(paths[i][0], cv2.IMREAD_UNCHANGED), size=img_size).reshape(data_shape[1:])
                y_out[i] = Pad_and_Resize(cv2.imread(paths[i][1], cv2.IMREAD_UNCHANGED), size=img_size).reshape(data_shape[1:])

        # Parallel
        self.X = np.empty(data_shape, dtype=np.uint8)
        self.Y = np.empty(data_shape, dtype=np.uint8)
        for i in range(parallel_count):
            print('Loading [{}/{}] - image index = {} to {}'.format(i+1, parallel_count, ss_list[i][0][0], ss_list[i][-1][-1]-1))
            Parallel(n_jobs=ncpu, verbose=0, require='sharedmem')( [delayed(_subprocess_for_XY_load)(self.X, self.Y, paths, ss1, ss2) for ss1, ss2 in ss_list[i]] )

        self.data_length = data_count
        self.batch_count = int(math.ceil(data_count / float(self.batch_size)))

        print('Data length         : {}'.format(self.data_length))
        print('Batch size          : {}'.format(self.batch_size))
        print('Batch count         : {}'.format(self.batch_count))
        print('X shape             : {}'.format(self.X.shape))
        print('Y shape             : {}'.format(self.Y.shape))

        print('Duration time [sec] : {}'.format(time.perf_counter() - time_start))


    # Number of batch in the Sequence
    def __len__(self): return self.batch_count


    # Gets batch at position index
    def __getitem__(self, index):
        bs = self.batch_size
        if index < self.batch_count - 1 :
            return self.rescale * self.X[index*bs : (index+1)*bs],     self.rescale * self.Y[index*bs : (index+1)*bs]
        else:
            return self.rescale * self.X[index*bs : self.data_length], self.rescale * self.Y[index*bs : self.data_length]


    def datalength(self): return self.data_length


    def getdataXY_in(self, i1, i2):
        return self.rescale * self.X[i1:i2], self.rescale * self.Y[i1:i2]


    def callback_on_train_begin(self, logs=None, params=None):
        pass


    def callback_on_epoch_begin(self, epoch, logs=None, params=None):     # epoch = 0, 1, 2, ...
        print('[Image Data Generator] {} : CSV dataset'.format(self.data_name))


    def callback_on_epoch_end(self, epoch, logs=None, params=None):     # epoch = 0, 1, 2, ...
        pass



