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

import tensorflow as tf


def restore_kernel_bias(mode='conv', restore_name=None, name=None, kernel_size=(1, 1),
                        depth_input=None, depth_output=None, is_trainable=False, collection_from=None, verbose=False):

    if mode == 'conv':
        _depth_input = depth_input
        _depth_output = depth_output
        _depth_bias = _depth_output
    elif mode == 'deconv':
        _depth_input = depth_output
        _depth_output = depth_input
        _depth_bias = _depth_input
    else:
        _depth_input = -1
        _depth_output = -1
        _depth_bias = -1

    with tf.variable_scope("", reuse=tf.AUTO_REUSE):

        kernel = tf.get_variable(
            name='{:s}/kernel'.format(restore_name),
            shape=(kernel_size[0], kernel_size[1], _depth_input, _depth_output),
            trainable=is_trainable,
            collections=collection_from
        )

        kernel = tf.Variable(initial_value=kernel, trainable=is_trainable, name='{:s}/kernel'.format(name))

        bias = tf.get_variable(
            name='{:s}/bias'.format(restore_name),
            shape=(_depth_bias,),
            trainable=is_trainable,
            collections=collection_from
        )

        bias = tf.Variable(initial_value=bias, trainable=is_trainable, name='{:s}/bias'.format(name))

        if verbose:
            print('kernel:', kernel)
            print('bias:', bias)
        else:
            pass

    return kernel, bias


def conv_layer_restore(inputs=None, strides=None, padding='VALID', channel_order=None, name=None,
                       restore_name=None, kernel_size=None, depth_input=None, depth_output=None,
                       is_trainable=False, collection_from=None, verbose=False):

    _strides = [1, strides[0], strides[1], 1] if strides is not None else [1, 1, 1, 1]

    kernel, bias = restore_kernel_bias(
        restore_name=restore_name,
        name=name,
        kernel_size=kernel_size, depth_input=depth_input, depth_output=depth_output,
        is_trainable=is_trainable, collection_from=collection_from, verbose=verbose
    )

    net = tf.nn.conv2d(input=inputs, filter=kernel, strides=_strides, padding=padding,
                       data_format=channel_order, name=name) + bias

    if verbose:
        print('<RESTORE>', name, ':', net, '=>', tf.get_variable_scope())
    else:
        pass

    # fin
    return net


def conv_T_layer_restore(inputs=None, strides=None, padding='VALID', channel_order=None, name=None,
                         restore_name=None, kernel_size=None, depth_input=None, depth_output=None,
                         is_trainable=False, collection_from=None, verbose=False):

    _strides = [1, strides[0], strides[1], 1] if strides is not None else [1, 1, 1, 1]

    kernel, bias = restore_kernel_bias(
        mode='deconv', restore_name=restore_name, name=name,
        kernel_size=kernel_size, depth_input=depth_input, depth_output=depth_output,
        is_trainable=is_trainable, collection_from=collection_from, verbose=verbose
    )

    input_shape = inputs.get_shape().as_list()
    kernel_shape = kernel.get_shape().as_list()
    # kernel = tf.reshape(tensor=kernel,
    #                     shape=[kernel_shape[0], kernel_shape[1], kernel_shape[3], kernel_shape[2]])

    if channel_order.lower() == 'nhwc':
        in_h = input_shape[1]
        in_w = input_shape[2]
    elif channel_order.lower() == 'nchw':
        in_h = input_shape[2]
        in_w = input_shape[3]
    else:
        in_h = None
        in_w = None
        pass

    # reference: https://github.com/tensorflow/tensorflow/issues/2118#issuecomment-215488127
    if padding == 'VALID':
        out_h = (in_h - kernel_shape[0] + _strides[1]) * _strides[1]
        out_w = (in_w - kernel_shape[1] + _strides[2]) * _strides[2]
    elif padding == 'SAME':
        out_h = in_h * _strides[1]
        out_w = in_w * _strides[2]
    else:
        out_h = None
        out_w = None

    # reference: https://github.com/tensorflow/tensorflow/issues/8972#issue-219501031
    if channel_order.lower() == 'nhwc':
        output_shape = [tf.shape(input=inputs)[0], out_h, out_w, depth_output]
    elif channel_order.lower() == 'nchw':
        output_shape = [tf.shape(input=inputs)[0], depth_output, out_h, out_w]
    else:
        output_shape = None
    # print('output_shape:', output_shape)

    # filter: A 4-D Tensor with the same type as value,
    # and shape [height, width, output_channels, in_channels].
    # filter's in_channels dimension must match that of value.
    net = tf.nn.conv2d_transpose(value=inputs, filter=kernel, output_shape=output_shape,
                                 strides=_strides, padding=padding, data_format=channel_order, name=name) + bias

    if verbose:
        print('conv_T_layer_restore:', depth_input, depth_output)
        print('kernel:', kernel)
        print('bias:', bias)
        print('<RESTORE>', name, ':', net)
    else:
        pass

    # fin
    return net


def conv_T_layer_restore_v2(inputs=None, strides=None, padding='VALID', channel_order=None,
                            name=None, kernel_size=None, depth_output=None,
                            is_trainable=False, collection_from=None, verbose=False):

    if channel_order.lower() == 'nhwc':
        _data_format = 'channels_last'
        channel_index = -1
    elif channel_order.lower() == 'nchw':
        _data_format = 'channels_first'
        channel_index = 1
    else:
        _data_format = None
        channel_index = None

    depth_input = inputs.get_shape().as_list()[channel_index]

    return conv_T_layer_restore(inputs=inputs, strides=strides, padding=padding, channel_order=channel_order,
                                name=name, restore_name=name, kernel_size=kernel_size,
                                depth_input=depth_input, depth_output=depth_output,
                                is_trainable=is_trainable, collection_from=collection_from, verbose=verbose)


def get_or_restore_convolution(type='conv', restore=False,
                               inputs=None, strides=None, padding='VALID', channel_order=None, name=None,
                               use_same_name_restore=True, restore_name=None, kernel_size=None,
                               depth_output=None, is_trainable=False, collection_from=None,
                               kernel_init=None, verbose=False):

    if channel_order.lower() == 'nhwc':
        _data_format = 'channels_last'
        channel_index = -1
    elif channel_order.lower() == 'nchw':
        _data_format = 'channels_first'
        channel_index = 1
    else:
        _data_format = None
        channel_index = None

    _restore_name = name if use_same_name_restore else restore_name
    depth_input = inputs.get_shape().as_list()[channel_index]

    if type == 'conv':
        if restore:
            net = conv_layer_restore(inputs=inputs, strides=strides, padding=padding, channel_order=channel_order,
                                     name=name, restore_name=_restore_name, kernel_size=kernel_size,
                                     depth_input=depth_input,  depth_output=depth_output,
                                     is_trainable=is_trainable, collection_from=collection_from, verbose=verbose)
        else:
            net = tf.layers.conv2d(inputs=inputs, filters=depth_output,
                                   kernel_size=kernel_size, strides=strides, padding=padding,
                                   data_format=_data_format, name=name, kernel_initializer=kernel_init)
    elif type == 'deconv':
        if restore:
            net = conv_T_layer_restore(inputs=inputs, strides=strides, padding=padding, channel_order=channel_order,
                                       name=name, restore_name=_restore_name, kernel_size=kernel_size,
                                       depth_input=depth_input, depth_output=depth_output,
                                       is_trainable=is_trainable, collection_from=collection_from, verbose=verbose)
        else:
            net = tf.layers.conv2d_transpose(inputs=inputs, filters=depth_output,
                                             kernel_size=kernel_size, strides=strides, padding=padding,
                                             data_format=_data_format, name=name, kernel_initializer=kernel_init)
    else:
        net = None

    return net
