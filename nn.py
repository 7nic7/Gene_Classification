import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Dense


def conv2d(inputs=None,
           filters=None,
           name=None,
           kernel_size=(2, 2),
           padding='valid',
           input_shape=None,
           activation=None,
           is_train=True,
           vis=False):
    if vis:
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(1, 1),
                      padding=padding,
                      input_shape=input_shape,
                      activation=activation,
                      name=name)
    return tf.layers.conv2d(inputs,
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=(1, 1),
                            padding=padding,
                            activation=activation,
                            trainable=is_train,
                            name=name)


def max_pool(inputs=None,
             name=None,
             pool_size=(2, 2),
             padding='valid',
             vis=False):
    if vis:
        return MaxPool2D(pool_size=pool_size,
                         strides=(2, 2),
                         padding=padding,
                         name=name)
    return tf.layers.max_pooling2d(inputs,
                                   pool_size=pool_size,
                                   strides=(2, 2),
                                   padding=padding,
                                   name=name)


def dense(inputs=None,
          units=None,
          name=None,
          activation=tf.nn.tanh,
          is_train=True,
          vis=False):
    if vis:
        return Dense(units=units,
                     activation=tf.nn.tanh,
                     name=name)
    return tf.layers.dense(inputs,
                           units=units,
                           activation=activation,
                           trainable=is_train,
                           name=name)
