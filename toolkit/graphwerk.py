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

from os import path
from pprint import PrettyPrinter
from sys import exit

import tensorflow as tf


def get_graph_col_dict_filtered(collection=None, f_list=None, verbose=False):

    # filter session variables by keywords STARTSWITH or ENDSWITH

    _collection = collection if collection is not None else tf.get_collection(key=tf.GraphKeys.GLOBAL_VARIABLES)

    g_col_ops = dict()
    g_col_f_ops = dict()
    for gx in _collection:
        gx_name = str(gx.name).replace(':0', '')
        for f_key in f_list:
            if str(gx.name).startswith(f_key) or str(gx.name).endswith(f_key):
                g_col_f_ops.update({gx_name: gx})
                break
            else:
                continue
        else:
            g_col_ops.update({gx_name: gx})

    if verbose:
        pp = PrettyPrinter()
        pp.pprint('[Unfiltered]')
        pp.pprint(g_col_ops)
        pp.pprint('[Filtered]')
        pp.pprint(g_col_f_ops)
    else:
        pass

    # fin
    return g_col_ops, g_col_f_ops


def restore_ckpt(sess=None, saver=None, ckpt_dir=None, model_name=None,
                 saver_ops_dict=None, saver_max_to_keep=None, exit_when_failed=True):

    _saver = None
    latest_ckpt_step = 0

    if saver is None:
        _saver = tf.train.Saver(var_list=saver_ops_dict, max_to_keep=saver_max_to_keep)
    else:
        _saver = saver

    # load checkpoint iff there exists correct checkpoint
    if tf.train.checkpoint_exists(checkpoint_prefix=ckpt_dir):
        # in-line checkpoint restoration
        latest_path = tf.train.latest_checkpoint(checkpoint_dir=ckpt_dir)
        latest_ckpt_name = path.basename(latest_path)
        latest_ckpt_step = int(latest_ckpt_name.replace(model_name + '-', '')) + 1

        print('[D] Started restoring Model {:s} from {:s}'.format(model_name, latest_ckpt_name))
        _saver.restore(sess=sess, save_path=latest_path)

        print('[D] Model {:s} restored from {:s} successfully!'.format(model_name, latest_ckpt_name))
    else:
        print('[D] Failed to load pre-trained model')
        if exit_when_failed:
            exit(-1)
        else:
            pass

    # fin
    return _saver, latest_ckpt_step
