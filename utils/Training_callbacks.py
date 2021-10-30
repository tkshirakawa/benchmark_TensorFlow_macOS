'''
    Copyright (c) 2021, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import numpy as np


from tensorflow.keras.callbacks import Callback     # TF2


# NOTE: Call this BestMetricsMonitor prior to the other callbacks to update mIoU metrics with the key 'val_iou_score' and 'val_dice_coef'.
class BestMetricsMonitor(Callback):

    def __init__(self, training_data, validation_data, model_base_path, nn_name, batch_size, patience, update_val_metrics):
        super(BestMetricsMonitor, self).__init__()

        self.trn_len    = training_data.datalength()
        self.val_gen = validation_data
        self.val_len = validation_data.datalength()

        self.model_base_path = model_base_path
        self.nn_name = nn_name

        self.patience = patience
        self.update_val_metrics = update_val_metrics

        self.ss_list = []
        for i in range(self.val_len):
            ss1 = i * batch_size
            ss2 = ss1 + batch_size
            if ss2 < self.val_len:
                self.ss_list.append((ss1, ss2, i))
                continue
            else:
                self.ss_list.append((ss1, self.val_len, i))
                break

        print('BestMetricsMonitor - monitors mean IoU and Dice coef to save the best model on each epoch end.')


    def on_train_begin(self, logs=None):
        self.best_miou = -np.Inf
        self.best_dice = -np.Inf
        self.wait = 0
        self.metrics_updated = False
        self.stopped_epoch = 0


    def on_epoch_begin(self, epoch, logs=None):
        self.trn_loss = 0.0
        self.trn_miou = 0.0
        self.trn_dice = 0.0

        self.epoch_time = time.perf_counter()
        print('---------------- Epoch {}/{} ----------------'.format(epoch+1, self.params['epochs']))


    def on_batch_end(self, batch, logs=None):
        count = float(batch + 1)
        self.trn_loss += logs['loss']
        self.trn_miou += logs['iou_score']
        self.trn_dice += logs['dice_coef']

        steps = self.params['steps']
        bar = '|' * int(40.0 * count / steps)
        dur_time = time.perf_counter()-self.epoch_time
        print('\r{}/{} [{: <40}] - {:.1f} s, {:.0f} ms/batch : Loss {:.4f}, mIoU {:.4f}, Dice {:.4f}'.format(int(count), steps, bar, dur_time, 1000*dur_time/count, self.trn_loss/count, self.trn_miou/count, self.trn_dice/count), end='', flush=True)
        if batch == steps - 1: print(' - done')


    def on_epoch_end(self, epoch, logs=None):
        '''
        The following 'mious' and 'dices' store IoU and Dice scores for each structure class (channel) averaged for the validation images (axis=0). Their shapes are (channels, ).
        On the other hand, IoU and Dice scores calculated in 'Loss_and_metrics.py' are the averaged values for the channels (axis=-1). So, their shapes are (batch size, ).
        This means that the scores in 'Loss_and_metrics.py' are NOT independent among structure classes,
        and the results here and in 'Loss_and_metrics.py' will NOT be the same if calculation will be performed with multi structure classes (channel>1).
        (The results will be completely same if calculation is for single class (channel=1).)
        '''
        print('\nMetrics for validation data in this epoch:')

        if self.update_val_metrics:
            time_start = time.perf_counter()
            mious  = 0.0
            dices  = 0.0
            counts = 0.0

            ss_len = len(self.ss_list)
            for ss1, ss2, i in self.ss_list:
                bar = '|' * int(30.0 * (i + 1) / ss_len)
                print('\rRecalculating {}/{} [{: <30}]'.format(i+1, ss_len, bar), end='', flush=True)

                x, t = self.val_gen.getdataXY_in(ss1, ss2)          # (ss2-ss1, height, width, 1), (ss2-ss1, height, width, 1)
                p = self.model.predict_on_batch(x).clip(0.0, 1.0)   # (ss2-ss1, height, width, 1)
                u = (t + p).clip(0.0, 1.0)                          # (ss2-ss1, height, width, 1)
                exist  = t.max(axis=(1,2))                          # (ss2-ss1, 1)
                union  = u.sum(axis=(1,2))                          # (ss2-ss1, 1)
                tr_pd  = t.sum(axis=(1,2)) + p.sum(axis=(1,2))      # (ss2-ss1, 1)
                intsec = np.maximum(tr_pd - union, 0.0)             # (ss2-ss1, 1)
                mious  += np.sum(exist * intsec / (union + 1e-5))
                dices  += np.sum(exist * 2.0 * intsec / (tr_pd + 1e-5))
                counts += np.sum(exist)

            mious /= np.maximum(counts, 1.0)
            dices /= np.maximum(counts, 1.0)
            print(' {:.2f} sec - done'.format(time.perf_counter() - time_start))

            miou = logs['val_iou_score'] = mious
            dice = logs['val_dice_coef'] = dices

        else:
            miou = logs['val_iou_score']
            dice = logs['val_dice_coef']

        print('[Averaged   ]  mIoU {:.6f}, Dice {:.6f} : data length {:<7}/ {:<7}\n'.format(miou, dice, self.trn_len, self.val_len))

        self.metrics_updated = False
        # def create_model_path(metrics_name, metrics_val):
        #     return '{0}, {1}={2:.4f}, {3}.h5'.format(self.model_base_path, metrics_name, metrics_val, self.nn_name)

        # Save model with the best mean IoU
        if miou > self.best_miou:
            print('Val. mean IoU  = {0:.6f} (updated from the value = {1:.6f})'.format(miou, self.best_miou))
            # path = create_model_path('mIoU', self.best_miou)
            # if os.path.exists(path): os.remove(path)
            # self.model.save(filepath=create_model_path('mIoU', miou), overwrite=True, include_optimizer=True)
            self.best_miou = miou
            self.metrics_updated = True
        else:
            print('Val. mean IoU  = {0:.6f} (not updated, the current best = {1:.6f})'.format(miou, self.best_miou))

        # Save model with the best Dice coef
        if dice > self.best_dice:
            print('Val. Dice coef = {0:.6f} (updated from the value = {1:.6f})'.format(dice, self.best_dice))
            # path = create_model_path('Dice', self.best_dice)
            # if os.path.exists(path): os.remove(path)
            # self.model.save(filepath=create_model_path('Dice', dice), overwrite=True, include_optimizer=True)
            self.best_dice = dice
            self.metrics_updated = True
        else:
            print('Val. Dice coef = {0:.6f} (not updated, the current best = {1:.6f})'.format(dice, self.best_dice))

        # Early stopping
        if self.metrics_updated:
            self.wait = 0
        else:
            self.wait += 1
            print('Patience count = {0} (early stop at {1} patience)'.format(self.wait, self.patience))
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print('Epoch {0}: early stopping...'.format(self.stopped_epoch + 1))
        print(' ')


    def metrics_updated_in_Monitor(self):
        return self.metrics_updated




class AutoLRManager(Callback):

    def __init__(self, param, bm_monitor):
        super(AutoLRManager, self).__init__()
        self.p = param
        self.bm_monitor = bm_monitor
        print('AutoLRManager - updates learning rate by monitoring the best metrics.')
        assert self.p['step'][0] <= 1.0 and self.p['step'][1] >= 1.0, 'Learning rate steps in LR_params must be <= 1.0 for decay and >=1.0 for recovery'


    def on_train_begin(self, logs=None):
        self.lr_multiplier = 1.0
        self.n_good = 0
        self.n_bad = 0


    def on_epoch_end(self, epoch, logs=None):
        metrics_updated = self.bm_monitor.metrics_updated_in_Monitor()
        print('Monitoring metrics for learning rate management... : {}'.format('updated' if metrics_updated else 'Not updated'))

        if metrics_updated:
            self.n_good += 1
            self.n_bad = 0
            step = self.p['step'][1]    # >= 1.0
            print('Count of update [{}] - LR will rise/recover at [{}] updates by step = {}'.format(self.n_good, self.p['patience'][1], step))
            if self.n_good >= self.p['patience'][1]:
                upper_limit = 1.0 if self.p['limit'][1] is None else self.p['limit'][1]
                self.lr_multiplier = min(upper_limit, self.lr_multiplier * step)
                print('LR multiplier is set to {} for the next epoch (upper limit = {})'.format(self.lr_multiplier, upper_limit))
                self.n_good = 0
        else:
            self.n_good = 0
            self.n_bad += 1
            step = self.p['step'][0]    # <= 1.0
            print('Count of not-update [{}] - LR will decay at [{}] patiences by step = {}'.format(self.n_bad, self.p['patience'][0], step))
            if self.n_bad >= self.p['patience'][0]:
                lower_limit = self.p['limit'][0]
                self.lr_multiplier = max(lower_limit, self.lr_multiplier * step)
                print('LR multiplier is set to {} for the next epoch (lower limit = {})'.format(self.lr_multiplier, lower_limit))
                self.n_bad = 0

        logs['lr_multiplier'] = self.lr_multiplier
        print(' ')


    def get_LR_multiplier(self):
        return self.lr_multiplier




class ImageDataGeneratorCallback(Callback):

    def __init__(self, training_data_generator, validation_data_generator):
        super(ImageDataGeneratorCallback, self).__init__()
        self.trn_dg = training_data_generator
        self.val_dg = validation_data_generator
        print('ImageDataGeneratorCallback - calls ImageDataGenerator.')

    def on_train_begin(self, logs=None):
        self.trn_dg.callback_on_train_begin(logs, self.params)
        self.val_dg.callback_on_train_begin(logs, self.params)

    def on_epoch_begin(self, epoch, logs=None):
        self.trn_dg.callback_on_epoch_begin(epoch, logs, self.params)
        self.val_dg.callback_on_epoch_begin(epoch, logs, self.params)

    def on_epoch_end(self, epoch, logs=None):
        self.trn_dg.callback_on_epoch_end(epoch, logs, self.params)
        self.val_dg.callback_on_epoch_end(epoch, logs, self.params)



