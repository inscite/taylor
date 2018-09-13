# Copyright (c) 2018 S H
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 2nd Library
import numpy as np
import tensorflow as tf


def mean_sqrt_error_l2(labels=None, predictions=None, joint_return=False, value_only=False, name_ext=''):

    namespace_ext = '' if name_ext == '' else '_{:s}'.format(name_ext)
    mse = tf.reduce_mean(tf.square(tf.subtract(labels, predictions)), name="mse{:s}".format(namespace_ext))

    if value_only:
        return mse
    else:
        pass

    with tf.device('/cpu:0'):
        mse_summary = tf.summary.scalar(name='loss/mse{:s}'.format(namespace_ext), tensor=mse)

    if joint_return:
        return mse, mse_summary
    else:
        return mse


def root_mean_sqrt_error(labels=None, predictions=None, joint_return=False, name_ext=''):
    namespace_ext = '' if name_ext == '' else '_{:s}'.format(name_ext)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predictions))), name="rmse{:s}".format(namespace_ext))

    with tf.device('/cpu:0'):
        rmse_summary = tf.summary.scalar(name='loss/rmse{:s}'.format(namespace_ext), tensor=rmse)

    if joint_return:
        return rmse, rmse_summary
    else:
        return rmse


def mean_abs_error_l1(labels=None, predictions=None, joint_return=False, name_ext=''):

    namespace_ext = '' if name_ext == '' else '_{:s}'.format(name_ext)
    mae = tf.reduce_mean(tf.abs(tf.subtract(labels, predictions)), name="mae{:s}".format(namespace_ext))

    with tf.device('/cpu:0'):
        mae_summary = tf.summary.scalar(name='loss/mae{:s}'.format(namespace_ext), tensor=mae)

    if joint_return:
        return mae, mae_summary
    else:
        return mae


def tf_cross_entropy_error(labels=None, predictions=None):

    entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=predictions)

    with tf.device('/cpu:0'):
        tf.summary.scalar(name='loss/cross_entropy', tensor=entropy)

    return entropy


def binary_cross_entropy_error(labels=None, predictions=None, joint_return=False):

    entropy = -tf.reduce_mean(labels * tf.log(predictions) + (1 - labels) * tf.log(1 - predictions))

    with tf.device('/cpu:0'):
        entropy_summary = tf.summary.scalar(name='loss/binary_cross_entropy', tensor=entropy)

    if joint_return:
        return entropy, entropy_summary
    else:
        return entropy


def peak_signal_to_noise_ratio(labels=None, predictions=None, mse=None, dtype=np.float32, peak=1, joint_return=False,
                               name_ext=''):

    peak_float = dtype(peak)

    namespace_ext = '' if name_ext == '' else '_{:s}'.format(name_ext)

    _mse = mse if mse is not None else mean_sqrt_error_l2(labels=labels, predictions=predictions, value_only=True)
    psnr = (20 * tf_log_b_x(x=peak_float, base=10)) - (10 * tf_log_b_x(x=_mse, base=10))

    with tf.device('/cpu:0'):
        psnr_summary = tf.summary.scalar(name='loss/psnr{:s}'.format(namespace_ext), tensor=psnr)

    if joint_return:
        return psnr, psnr_summary
    else:
        return psnr


def tf_log_b_x(x=None, base=10):

    # explicit log base declaration

    num = tf.log(x=x)
    denom = tf.log(tf.constant(value=base, dtype=num.dtype))

    # fin
    return num / denom


def get_loss_for_summary(prediction=None, label=None):

    mae = mean_abs_error_l1(labels=label, predictions=prediction)
    mse = mean_sqrt_error_l2(labels=label, predictions=prediction)
    rmse = root_mean_sqrt_error(labels=label, predictions=prediction)
    psnr = peak_signal_to_noise_ratio(labels=label, predictions=prediction)

    # fin
    return
