'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os, platform, datetime
import importlib.machinery as imm


# To supress/disable logging output from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Select for Automatic Mixed Precision (AMP) - See https://www.tensorflow.org/guide/mixed_precision
_use_AMP = False #True

precision = 'mixed_float16' if _use_AMP else 'float32'
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1' if _use_AMP else '0'


if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    assert not _use_AMP, 'AMP is enabled, which causes slow computation speed in macOS.'


_num_threads = os.cpu_count()


_timezone_JST = datetime.timezone(datetime.timedelta(hours=+9), 'JST')      # Japan Standard Time, Change it for your time zone


# The absolute path to this file and directory
train_file_path = os.path.abspath(__file__)
train_dir_path = os.path.dirname(train_file_path)




#############################################################################################
# Define loss and metrics.

loss_metrics_path = os.path.join(train_dir_path, 'utils', 'Loss_and_metrics.py')
_LM = imm.SourceFileLoader('Loss_and_metrics', loss_metrics_path).load_module()

# Loss function for training
loss_func = _LM.MSE_loss_w_iou_score
# loss_func = _LM.MSE_loss_w_dice_coef
# loss_func = _LM.MSE_loss_w_iou_score_rot
# loss_func = _LM.MSE_loss_w_dice_coef_rot

def get_loss(): return {loss_func.__name__: loss_func}


# Metrics
# NOTE: The metrics defined here MUST include the keys 'iou_score' and 'dice_coef' for monitoring the best performance model.
# NOTE: The metrics defined in Loss_and_metrics.py are estimated / assumed values for each batch, NOT final results.
def get_metrics(): return {'iou_score': _LM.iou_score, 'dice_coef': _LM.dice_coef}

# Define loss and metrics.
#############################################################################################




#############################################################################################
# Main training sequence

def Train(neural_network_py, training_data, validation_data, output_dir_path, update_val_metrics_for_epoch):

    import glob, shutil, time, math
    import numpy as np

    import tensorflow as tf

    from utils.Training_callbacks import BestMetricsMonitor, AutoLRManager, ImageDataGeneratorCallback
    from utils.Image_data_generator import ImageDataGenerator_CSV_with_Header


    # NOTE: Data must be (samples, height, width, channels)
    assert tf.keras.backend.image_data_format() == 'channels_last', 'image_data_format must be \'channels_last\''


    # For mixed precision computing - See https://www.tensorflow.org/guide/mixed_precision
    #   mixed_precision.set_global_policy(policy)
    #   policy : A Policy, or a string that will be converted to a Policy. Can also be None,
    #   in which case the global policy will be constructed from tf.keras.backend.floatx()
    if _use_AMP: tf.keras.mixed_precision.set_global_policy('mixed_float16')


    print('__________________________________________________________________________________________________')

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0 : print('Available GPUs: {}'.format(gpus))
    else             : print('!!! No GPUs found !!!')

    try:    numpy_blas = np.__config__.blas_opt_info['libraries']
    except: numpy_blas = 'Not found'
    try:    numpy_lapack = np.__config__.lapack_opt_info['libraries']
    except: numpy_lapack = 'Not found'
    print('Numpy BLAS libraries:   {}'.format(numpy_blas))
    print('Numpy LAPACK libraries: {}'.format(numpy_lapack))


    # Make an unique model ID and directory names from the time now
    while True:
        starttime = time.time()
        startdate = datetime.datetime.now(_timezone_JST)
        analysis_id = startdate.strftime("%Y%m%d%H%M%S")

        work_dir_path = os.path.join(output_dir_path, 'run' + analysis_id + '*')

        if len(glob.glob(work_dir_path)) == 0 :
            temp_work_dir_path = work_dir_path[:-1]
            os.makedirs(temp_work_dir_path)
            break
        else:
            time.sleep(2)


    ############################################
    # Training parameters

    # Load loss and metrics
    loss = get_loss()
    metrics = get_metrics()
    loss_name = list(loss.keys())[0]

    # Neural network model
    NN = imm.SourceFileLoader(os.path.splitext(os.path.basename(neural_network_py))[0], neural_network_py).load_module()
    NN_notification = []

    try:    NN_model_name    = NN.Model_Name()
    except: NN_model_name, _ = NN.__name__, NN_notification.append('ALERT: Define a model name in the neural network model file.')

    try:    NN_model_descript    = NN.Model_Description()
    except: NN_model_descript, _ = 'Empty description.', NN_notification.append('ALERT: Define description for the model in the neural network model file.')

    try:    NN_batch_size    = NN.Batch_Size()
    except: NN_batch_size, _ = 32, NN_notification.append('NOTE: The batch size is not defined in the neural network model file, automatically set to 32.')

    try:    NN_input_shape = NN.Input_Shape()
    except: NN_input_shape, _ = (512,512,1), NN_notification.append('ALERT: The input shape is not defined in the neural network model file, automatically set to (512,512,1).')

    try:    NN_num_classes = NN.Class_Number()
    except: NN_num_classes, _ = 1, NN_notification.append('ALERT: The class number is not defined in the neural network model file, automatically set to 1.')


    # Initial parameters
    LR_params = {'formula'          : [None, 0.0, 0],               # Learning rate formula calculates LR at points of epochs - ['poly', base_lr, number_of_epochs] is available
                 'graph'            : [[0,4e-3], [100,2e-3]],       # Learning rate graph defines LR at points of epochs - [[epoch_1, LR_1], [epoch_2, LR_2], ... [epoch_last, LR_last]]
                 'step'             : [0.1, 2.5],                   # Multiplying values to LR - will be applied when mIoU is [NOT improved, improved]
                 'limit'            : [0.001, 1.0] ,                # Limitation of LR multiplier - when [NOT improved, improved]
                 'patience'         : [1000, 1000],                 # Patience counts before applying step for LR - when [NOT improved, improved]
                 'stop_count'       : 50 }                          # Define a count number before early stopping

    LR_params_set = True
    try:    LR_params = NN.Learning_Rate_Parameters()
    except: LR_params_set, _ = False, NN_notification.append('ALERT: The list of learning rate parameters is not defined in the neural network model file, trying to set separately.')

    if not LR_params_set:
        try:    LR_params['formula']    = NN.Learning_Rate_Formula()
        except: NN_notification.append('ALERT: The LR formula is not defined in the neural network model file, automatically deactivated.')

        try:    LR_params['graph']    = NN.Learning_Rate_Lsit()
        except: NN_notification.append('ALERT: The LR graph is not defined in the neural network model file, automatically set.')

        try:    LR_params['patience']    = NN.Count_before_LR_Step()
        except: NN_notification.append('ALERT: The patience count for LR step is not defined in the neural network model file, automatically set.')

        try:    LR_params['stop_count']    = NN.Count_before_Stop()
        except: NN_notification.append('ALERT: The count for early stopping is not defined in the neural network model file, automatically set.')

    if LR_params['formula'][0] is None: number_of_epochs = LR_params['graph'][-1][0]
    else:                               number_of_epochs = LR_params['formula'][2]


    # Paths and directories
    training_data_dirname = os.path.basename(os.path.dirname(training_data))
    work_dir_path   = os.path.join(output_dir_path, 'run' + analysis_id + ', ' + NN_model_name + ', ' + training_data_dirname)

    if os.path.exists(temp_work_dir_path) : os.rename(temp_work_dir_path, work_dir_path)
    else                                  : os.makedirs(work_dir_path)

    model_base_path = os.path.join(work_dir_path, 'model' + analysis_id)


    # Image data generator
    '''
        For TensorFlow 2.0
        Model.fit_generator IS DEPRECATED.
        To use Model.fit, generator classes, ImageDataGenerator_XXX(), were updated as subclasses of keras.utils.Sequence.

        See:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
    '''
    print('__________________________________________________________________________________________________')
    print('- Loading images for training...')
    ext = os.path.splitext(training_data)[1]
    if   ext == '.csv':
        training_images = ImageDataGenerator_CSV_with_Header('Training data from CSV', training_data, input_shape=NN_input_shape, batch_size=NN_batch_size, rescale=1.0/255.0)
    else:
        sys.exit()

    print('\n- Loading images for validation...')
    ext = os.path.splitext(validation_data)[1]
    if   ext == '.csv':
        validation_images = ImageDataGenerator_CSV_with_Header('Validation data from CSV', validation_data, input_shape=NN_input_shape, batch_size=NN_batch_size, rescale=1.0/255.0)
    else:
        sys.exit()


    # Construct model object
    if LR_params['formula'][0] is not None: base_lr = LR_params['formula'][1]
    elif LR_params['graph'] is not None:    base_lr = LR_params['graph'][0][1]
    else:
        print('Invalid learning rate.')
        sys.exit()

    model = NN.Build_Model()


    # SGD(lr=base_lr, momentum=0.95, nesterov=True)                               # base_lr=0.1
    # Adam(lr=base_lr, clipnorm=2.0, beta_1=0.9, beta_2=0.999, amsgrad=False)     # base_lr=0.001
    # Nadam(lr=base_lr, clipnorm=2.0, beta_1=0.9, beta_2=0.999)                   # base_lr=0.001
    optimizer = tf.keras.optimizers.SGD(learning_rate=base_lr, momentum=0.95, nesterov=True)

    optimizer_name = optimizer.__class__.__name__
    if _use_AMP: optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)


    model.compile(optimizer          = optimizer,
                  loss               = loss[loss_name],
                  metrics            = list(metrics.values()),
                  loss_weights       = None,
                  sample_weight_mode = None,
                  weighted_metrics   = None,
                  target_tensors     = None )


    # Descriptions
    model.summary()
    print('Date                    : {0}'.format(startdate))
    print('TensorFlow version      : {0}'.format(tf.version.VERSION))
    print('OS-version              : {0}'.format(platform.platform()))
    print('Processor               : {0}'.format(platform.processor()))
    print('GPUs                    : {0}'.format(gpus))
    print('Numpy multi-threading   : {0}'.format('YES, count='+str(_num_threads) if _num_threads > 1 else 'NO'))
    print('Numpy BLAS libraries    : {0}'.format(numpy_blas))
    print('Numpy LAPACK libraries  : {0}'.format(numpy_lapack))
    print('__________________________________________________________________________________________________')
    print('Model name              : {0}'.format(NN_model_name))
    print('Model description       : {0}'.format(NN_model_descript))
    print('Model input shape       : {0}'.format(NN_input_shape))
    print('Number of classes       : {0}'.format(NN_num_classes))
    print('Loaded model path       : {0}'.format(neural_network_py))
    print('Working directory       : {0}'.format(work_dir_path))
    print('__________________________________________________________________________________________________')
    print('Optimizer               : {0}'.format(optimizer_name))
    print('Precision mode          : {0}'.format('Mixed precision FP16 (dynamic loss scale)' if _use_AMP else 'Single precision FP32'))
    print('Loss                    : {0}'.format(loss_name))
    print('Metrics                 : {0}'.format(list(metrics.keys()) ))
    print('Batch size              : {0}'.format(NN_batch_size))
    print('Epochs                  : {0}'.format(number_of_epochs))
    print('Metrics recalculation   : {0} for each epoch end'.format('ON' if update_val_metrics_for_epoch else 'OFF'))
    print('Learning rate formula   : {0}'.format(LR_params['formula']))
    print('Learning rate graph     : {0}'.format(LR_params['graph']))
    print('LR step                 : {0}'.format(LR_params['step']))
    print('LR limit                : {0}'.format(LR_params['limit']))
    print('Patience for LR step    : {0}'.format(LR_params['patience']))
    print('Patience for early stop : {0}'.format(LR_params['stop_count']))
    print('__________________________________________________________________________________________________')
    if len(NN_notification) > 0:
        for n in NN_notification: print(n)
        print('__________________________________________________________________________________________________')


    # Save descriptions, network figure and parameters
    with open(os.path.join(work_dir_path,'training_parameters.txt'), mode='w') as f:
        f.write('Date                    : {0}\n'.format(startdate))
        f.write('TensorFlow version      : {0}\n'.format(tf.version.VERSION))
        f.write('OS-version              : {0}\n'.format(platform.platform()))
        f.write('Processor               : {0}\n'.format(platform.processor()))
        f.write('GPUs                    : {0}\n'.format(gpus))
        f.write('Numpy multi-threading   : {0}\n'.format('YES, count='+str(_num_threads) if _num_threads > 1 else 'NO'))
        f.write('Numpy BLAS libraries    : {0}\n'.format(numpy_blas))
        f.write('Numpy LAPACK libraries  : {0}\n\n'.format(numpy_lapack))
        f.write('Model name              : {0}\n'.format(NN_model_name))
        f.write('Model description       : {0}\n'.format(NN_model_descript))
        f.write('Model input shape       : {0}\n'.format(NN_input_shape))
        f.write('Number of classes       : {0}\n'.format(NN_num_classes))
        f.write('Loaded model path       : {0}\n'.format(neural_network_py))
        f.write('Working directory       : {0}\n\n'.format(work_dir_path))
        f.write('Training images         : {0} sets in {1}\n'.format(training_images.datalength(), training_data))
        f.write('Validation images       : {0} sets in {1}\n\n'.format(validation_images.datalength(), validation_data))
        f.write('Optimizer               : {0}\n'.format(optimizer_name))
        f.write('Precision mode          : {0}\n'.format('Mixed precision FP16 (dynamic loss scale)' if _use_AMP else 'Single precision FP32'))
        f.write('Loss                    : {0}\n'.format(loss_name))
        f.write('Metrics                 : {0}\n'.format(list(metrics.keys()) ))
        f.write('Batch size              : {0}\n'.format(NN_batch_size))
        f.write('Epochs                  : {0}\n'.format(number_of_epochs))
        f.write('Metrics recalculation   : {0} for each epoch end\n'.format('ON' if update_val_metrics_for_epoch else 'OFF'))
        f.write('Learning rate formula   : {0}\n'.format(LR_params['formula']))
        f.write('Learning rate graph     : {0}\n'.format(LR_params['graph']))
        f.write('LR step                 : {0}\n'.format(LR_params['step']))
        f.write('LR limit                : {0}\n'.format(LR_params['limit']))
        f.write('Patience for LR step    : {0}\n\n'.format(LR_params['patience']))
        f.write('Patience for early stop : {0}\n'.format(LR_params['stop_count']))
        if len(NN_notification) > 0:
            for n in NN_notification: f.write('{}\n'.format(n))
        f.write('\n')
        model.summary(print_fn=lambda x: f.write(x + '\n'))


    # Allocate callbacks
    print('\n- Defining callbacks...')
    bm_monitor = BestMetricsMonitor(training_data=training_images, validation_data=validation_images, model_base_path=model_base_path, nn_name=NN_model_name, batch_size=NN_batch_size,
                                        patience=LR_params['stop_count'], update_val_metrics=update_val_metrics_for_epoch)
    data_generator_callback = ImageDataGeneratorCallback(training_data_generator=training_images, validation_data_generator=validation_images)
    lr_manager = AutoLRManager(param=LR_params, bm_monitor=bm_monitor)

    def ScheduleLR(epoch, lr):
        print('[Learning Rate Scheduler]')
        raw_lr = lr
        if LR_params['formula'][0] == 'poly':
            print('Polynomial decay : base_lr = {}, power = 0.9'.format(LR_params['formula'][1]))   # See https://arxiv.org/pdf/1506.04579.pdf
            raw_lr = LR_params['formula'][1] * math.pow(1 - epoch / number_of_epochs, 0.9)
        # elif LR_params['formula'][0] == 'XXX':
        #     print('Learning rate by XXX: ...
        #     raw_lr = LR_params['formula'][1] ...
        elif LR_params['graph'] is not None:
            print('Predefined graph : [epoch, LR] = {}'.format(LR_params['graph']))
            def LR_at_epoch(epoch, pt1, pt2): return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) * (epoch - pt1[0]) + pt1[1]
            for i in range(len(LR_params['graph'])-1):
                if LR_params['graph'][i][0] <= epoch and epoch < LR_params['graph'][i+1][0]:
                    raw_lr = LR_at_epoch(epoch, LR_params['graph'][i], LR_params['graph'][i+1])
                    break
        m = lr_manager.get_LR_multiplier()
        new_LR = m * raw_lr
        print('Learning rate = {} (raw LR = {}, LR multiplier = {})'.format(new_LR, raw_lr, m))
        return new_LR

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(ScheduleLR, verbose=0)
    csv_logger   = tf.keras.callbacks.CSVLogger(os.path.join(work_dir_path,'training_log.csv'), separator=',', append=False)


    # Train the model
    '''
        For TensorFlow 2.0
        fit_generator -> fit

        Warning: Model.fit_generator IS DEPRECATED. It will be removed in a future version.
        Instructions for updating: Please use Model.fit, which supports generators.

        See:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
    '''
    print('\n- Starting model training...')

    model.fit(
        x                       = training_images,
        epochs                  = number_of_epochs,
        verbose                 = 0,
        callbacks               = [bm_monitor, data_generator_callback, lr_manager, lr_scheduler, csv_logger],
        validation_data         = validation_images,
        shuffle                 = False,
        initial_epoch           = 0,
        max_queue_size          = 10,
        workers                 = 1,
        use_multiprocessing     = False )


    print('\n==================================================================================================')
    print('Computation time        : {0}'.format(datetime.timedelta(seconds=time.time()-starttime)))
    print('From the date           : {0}\n'.format(startdate))
    print('==================================================================================================')




# Main
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Training manager for deep learning using Keras/TensorFlow. Copyright (c) 2019-2021, Takashi Shirakawa. All rights reserved.', add_help=True)
    parser.add_argument('-n', '--neural_network_py', help='Path to a neural network file of a Keras model (eg CV_net_Synapse.py)', required=True)
    parser.add_argument('-t', '--training_data', help='Path to a CSV/.h5 file for training images', required=True)
    parser.add_argument('-v', '--validation_data', help='Path to a CSV/.h5 file for validation images', required=True)
    parser.add_argument('-o', '--output', help='Path to a directory to save results in it', required=True)
    parser.add_argument('--update_val_metrics_for_epoch', help='Update IoU and Dice metrics of validation data at the end of each epoch', action='store_true')
    args = parser.parse_args()

    Train(args.neural_network_py, args.training_data, args.validation_data, args.output, True)



