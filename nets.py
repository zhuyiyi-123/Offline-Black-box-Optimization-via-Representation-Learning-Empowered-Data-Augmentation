import torch.nn as nn
import torch.nn.functional as F
import tensorflow.keras.layers as tfkl
import tensorflow as tf
from tensorflow_probability import distributions as tfpd
import numpy as np


class TanhMultiplier(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super(TanhMultiplier, self).__init__(**kwargs)
        w_init = tf.constant_initializer(1.0)
        self.multiplier = tf.Variable(initial_value=w_init(
            shape=(1,), dtype="float32"), trainable=True)

    def call(self, inputs, **kwargs):
        exp_multiplier = tf.math.exp(self.multiplier)
        return tf.math.tanh(inputs / exp_multiplier) * exp_multiplier



def EncoderModel(input_shape, activations=('relu', 'relu'), hidden=2048, final_tanh=False, dropout_rate=0.4,
                 use_batchnorm=False):
    activations = [tf.keras.layers.LeakyReLU() if act == 'leaky_relu' else
                   tf.keras.layers.Activation(tf.math.cos) if act == 'cos' else
                   act for act in activations]

    layers = [tfkl.Flatten(input_shape=input_shape)]
    for act in activations:
        layers.extend([tfkl.Dense(hidden), act])
        if use_batchnorm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.extend([tfkl.Dense(1)])
    if final_tanh:
        layers.extend([TanhMultiplier()])
    return tf.keras.Sequential(layers)



class ContinuousDecoderModel(tf.keras.Model):

    def __init__(self, design_shape, latent_size, hidden=50, dropout_rate=0.5):
        super(ContinuousDecoderModel, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size
        self.dropout_rate = dropout_rate

        self.embed_0 = tf.keras.layers.Dense(hidden)
        self.embed_0.build((None, 1))

        self.dense_0 = tf.keras.layers.Dense(hidden)
        self.dense_0.build((None, latent_size + hidden))
        self.dropout_0 = tf.keras.layers.Dropout(dropout_rate)
        self.ln_0 = tf.keras.layers.LayerNormalization()
        self.ln_0.build((None, hidden))

        self.dense_1 = tf.keras.layers.Dense(hidden)
        self.dense_1.build((None, hidden + hidden))
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.ln_1 = tf.keras.layers.LayerNormalization()
        self.ln_1.build((None, hidden))

        self.dense_2 = tf.keras.layers.Dense(hidden)
        self.dense_2.build((None, hidden + hidden))
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.ln_2 = tf.keras.layers.LayerNormalization()
        self.ln_2.build((None, hidden))

        self.dense_3 = tf.keras.layers.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden + hidden))

    def sample(self, y, **kwargs):
        kwargs.pop("temp", 1.0)
        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)
        y_embed = self.embed_0(y, **kwargs)

        x = self.dense_0(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)

        x = self.dense_2(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(tf.concat([x, y_embed], 1), **kwargs)
        return tf.reshape(x, [tf.shape(y)[0], *self.design_shape])


class DiscreteDecoderModel(tf.keras.Model):

    def __init__(self, design_shape, latent_size, hidden=50):
        super(DiscreteDecoderModel, self).__init__()
        self.design_shape = design_shape
        self.latent_size = latent_size
        self.embed_0 = tfkl.Dense(hidden)
        self.embed_0.build((None, 1))

        # define a layer of the neural net with two pathways
        self.dense_0 = tfkl.Dense(hidden)
        self.dense_0.build((None, latent_size + hidden))
        self.ln_0 = tfkl.LayerNormalization()
        self.ln_0.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_1 = tfkl.Dense(hidden)
        self.dense_1.build((None, hidden + hidden))
        self.ln_1 = tfkl.LayerNormalization()
        self.ln_1.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_2 = tfkl.Dense(hidden)
        self.dense_2.build((None, hidden + hidden))
        self.ln_2 = tfkl.LayerNormalization()
        self.ln_2.build((None, hidden))

        # define a layer of the neural net with two pathways
        self.dense_3 = tfkl.Dense(np.prod(design_shape))
        self.dense_3.build((None, hidden + hidden))

    def sample(self, y, **kwargs):
        temp = kwargs.pop("temp", 1.0)
        z = tf.random.normal([tf.shape(y)[0], self.latent_size])
        x = tf.cast(z, tf.float32)
        y = tf.cast(y, tf.float32)

        y_embed = self.embed_0(y, **kwargs)
        x = self.dense_0(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_0(x), alpha=0.2)
        x = self.dense_1(tf.concat([x, y_embed], 1), **kwargs)

        x = tf.nn.leaky_relu(self.ln_1(x), alpha=0.2)
        x = self.dense_2(tf.concat([x, y_embed], 1), **kwargs)
        x = tf.nn.leaky_relu(self.ln_2(x), alpha=0.2)
        x = self.dense_3(tf.concat([x, y_embed], 1), **kwargs)

        # logits = tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
        # return tfpd.RelaxedOneHotCategorical(
        #     temp, logits=tf.math.log_softmax(logits)).sample()
        return tf.reshape(x, [tf.shape(y)[0], *self.design_shape])
