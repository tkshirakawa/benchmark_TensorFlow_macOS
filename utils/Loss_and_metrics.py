'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# Axis for image
# IMPORTANT to complete calculation in each image plane in several batches or samples 
# NOTE: K.image_data_format() must be 'channels_last'! = (samples, height, width, channels)
# NOTE: In the following metrics and losses, the tensor 'y_XXXX' has the dimension of (batch size, height, width, channels)
# _HW = (-3,-2,-1)
_HW = (-3,-2)       # (batch size, height, width, channels) -> (batch size, channels)




'''
Metrics
'''

# IoU: 0.0 [bad] - 1.0 [good]
def iou_score_base(y_true, y_pred, weight='pixel'):
    '''
    Return
    1) IoU score for each element in y_true and y_pred.
    2) tf.reduce_max(t, axis=_HW) represents existence of non-black pixels in each image in y_ture.
    NOTE: tf.math.divide_no_nan computes an unsafe divide which returns 0 if the y is zero.
    '''
    t = tf.clip_by_value(y_true, 0, 1)            # (batch size, height, width, channels)
    p = tf.clip_by_value(y_pred, 0, 1)
    u = tf.clip_by_value(t + p, 0, 1)
    truth = tf.reduce_sum(t, axis=_HW)          # (batch size, channels)
    predc = tf.reduce_sum(p, axis=_HW)
    union = tf.reduce_sum(u, axis=_HW)
    if weight == 'pixel'  : return tf.math.divide_no_nan(truth + predc - union, union), tf.reduce_max(t, axis=_HW)      # (batch size, channels)
    elif weight == 'area' : return tf.math.divide_no_nan(truth + predc - union, union), union                   # (batch size, channels)
    else                  : return tf.math.divide_no_nan(truth + predc - union, union)

def iou_score(y_true, y_pred):
    v, w = iou_score_base(y_true, y_pred, weight='pixel')
    return tf.math.divide_no_nan(tf.reduce_sum(v * w, axis=-1), tf.reduce_sum(w, axis=-1))      # (batch size)


# Dice: 0.0 [bad] - 1.0 [good]
def dice_coef_base(y_true, y_pred, weight='pixel'):
    t = tf.clip_by_value(y_true, 0, 1)            # (batch size, height, width, channels)
    p = tf.clip_by_value(y_pred, 0, 1)
    u = tf.clip_by_value(t + p, 0, 1)
    truth = tf.reduce_sum(t, axis=_HW)          # (batch size, channels)
    predc = tf.reduce_sum(p, axis=_HW)
    union = tf.reduce_sum(u, axis=_HW)
    tr_pd = truth + predc
    if weight == 'pixel'  : return tf.math.divide_no_nan(2.0 * (tr_pd - union), tr_pd), tf.reduce_max(t, axis=_HW)      # (batch size, channels)
    elif weight == 'area' : return tf.math.divide_no_nan(2.0 * (tr_pd - union), tr_pd), tr_pd                   # (batch size, channels)
    else                  : return tf.math.divide_no_nan(2.0 * (tr_pd - union), tr_pd)

def dice_coef(y_true, y_pred):
    v, w = dice_coef_base(y_true, y_pred, weight='pixel')
    return tf.math.divide_no_nan(tf.reduce_sum(v * w, axis=-1), tf.reduce_sum(w, axis=-1))      # (batch size)




'''
Losses
'''

def combine_losses(funcA, funcB, y_true, y_pred):
    a = funcA(y_true, y_pred)           # (batch size, channels)
    b, e = funcB(y_true, y_pred)        # (batch size, channels)
    return tf.math.divide_no_nan(tf.reduce_sum(e * (a + b), axis=-1), tf.reduce_sum(e, axis=-1))


def iou_score_loss_base(y_true, y_pred):
    v, w = iou_score_base(y_true, y_pred, weight='pixel')
    return 1.0 - tf.square(v), w

def dice_coef_loss_base(y_true, y_pred):
    v, w = dice_coef_base(y_true, y_pred, weight='pixel')
    return 1.0 - tf.square(v), w


def MSE_loss_base(y_true, y_pred):
    t = tf.clip_by_value(y_true, 0, 1)        # (batch size, height, width, channels)
    p = tf.clip_by_value(y_pred, 0, 1)
    u = tf.clip_by_value(t + p, 0, 1)
    return tf.math.divide_no_nan(tf.reduce_sum(tf.math.squared_difference(p, t), axis=_HW), tf.reduce_sum(u, axis=_HW))   # (batch size, channels)

def MSE_loss_w_iou_score(y_true, y_pred):   return combine_losses(MSE_loss_base, iou_score_loss_base, y_true, y_pred)
def MSE_loss_w_dice_coef(y_true, y_pred):   return combine_losses(MSE_loss_base, dice_coef_loss_base, y_true, y_pred)



