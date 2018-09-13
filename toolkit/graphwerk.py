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

from pprint import PrettyPrinter

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
