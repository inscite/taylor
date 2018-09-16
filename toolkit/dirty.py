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

import os
import random

import numpy as np
import tensorflow as tf


def absrootpath(path_dest):
    return os.path.join(os.environ['PYTHONPATH'].split(':')[0], path_dest)


def read_list(path=None, sort=True, as_ndarray=False):
    resolve = list()
    with open(path) as f:
        for line in f:
            resolve.append(line.strip())

    if sort:
        resolve.sort()
    else:
        pass

    if as_ndarray:
        resolve = np.array(resolve)
    else:
        pass

    return resolve


def write_list(path=None, filemode='w', item_list=None):
    with open(path, filemode) as f:
        for item in item_list:
            f.write(item + '\n')


def summary_filters(namespace, tensor, trigger=True, mode='img', channel_order='nhwc', num_of_filters=1):
    with tf.device('/cpu:0'):
        # get tensor shape
        tensor_shape = tensor.get_shape()
        tensor_shape_val = tensor_shape.as_list()

        if mode == 'img' and trigger and num_of_filters > 0:
            try:
                tensor_pick = tensor[:1, :, :, :]
                if channel_order.lower() == 'nhwc':
                    channel_index = -1
                    tensor_punct = tensor_pick
                elif channel_order.lower() == 'nchw':
                    channel_index = 1
                    tensor_punct = tf.split(value=tensor_pick[0],
                                            num_or_size_splits=tensor_shape_val[channel_index], axis=0)
                    tensor_punct = tf.stack(values=tensor_punct, axis=-1)
                else:
                    channel_index = None
                    tensor_punct = None

                candidates = random.sample(range(tensor_shape_val[channel_index]), num_of_filters)

                for idx in candidates:
                    tf.summary.image(namespace + '/' + str(idx), tensor_punct[:1, :, :, idx:idx+1], max_outputs=1)
            except ValueError:
                print('failed to summary %d filter(s) from tensor %s' % (num_of_filters, namespace))
                pass
        elif mode == 'grads' and trigger and num_of_filters > 0:
            try:
                candid_second = random.sample(range(tensor_shape[-2].value), 1)
                candidates = random.sample(range(tensor_shape[-1].value), num_of_filters)
                for idx in candidates:
                    # print('CANDIDA:', candid_second[0], idx)
                    # print('CANDID:', tensor[:, :, candid_second[0]:candid_second[0]+1, idx:idx+1])
                    getcha = tf.reshape(tensor=tensor[:, :, candid_second[0]:candid_second[0]+1, idx:idx+1],
                                        shape=(1, tensor_shape[0], tensor_shape[1], 1))
                    tf.summary.image(namespace+'/'+str(idx), getcha, max_outputs=1)
            except ValueError:
                print('failed to summary %d filter(s) from tensor %s' % (num_of_filters, namespace))
                pass
        else:
            pass
