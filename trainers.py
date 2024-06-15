from collections import defaultdict
import torch
import torch.nn as nn
from torch import autograd
from noise import disc_noise, cont_noise
import tensorflow as tf
import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Autoencoder(tf.Module):
    def __init__(self, encoder, decoder, optimizer=tf.keras.optimizers.Adam, learning_lr=1e-3, is_discrete=False,
                 keep=0.99, noise_std=0.0, input_size=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=learning_lr)
        self.optimizer_decoder = tf.keras.optimizers.Adam(
            learning_rate=learning_lr,
            beta_1=0.0,
            beta_2=0.9)
        self.is_discrete = is_discrete
        self.keep = keep
        self.noise_std = noise_std
        self.temp = tf.Variable(0.0, dtype=tf.float32)
        self.input_size = input_size

    def mean_nll(self, y_gen, y):
        return nn.functional.binary_cross_entropy_with_logits(y_gen, y)

    def penalty(self, y_gen, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = self.mean_nll(y_gen * scale, y)
        grad = autograd.grad(loss, [scale], create_graph=True, retain_graph=True)[0]
        return torch.sum(grad ** 2)

    def train_step(self, x, y, w):
        statistics = dict()
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x = disc_noise(x, keep=self.keep, temp=self.temp)
        else:
            x = cont_noise(x, self.noise_std)

        with tf.GradientTape(persistent=True) as tape:
            y_gen = self.encoder(x, training=True)
            loss_erm = (y_gen - y) * (y_gen - y)
            x_gen = self.decoder.sample(y_gen, temp=self.temp, training=False)
            loss_recons =(x - x_gen) * (x - x_gen)
            statistics[f'loss_erm'] = tf.convert_to_tensor(loss_erm)
            statistics[f'loss_recons'] = tf.convert_to_tensor(loss_recons)

            # build the total loss
            weight_1 = tf.constant(0.01)
            # weight_2 = tf.constant(0.05)
            total_loss = tf.reduce_mean(w * (loss_erm))
            statistics[f"total_loss/0.01"] = tf.convert_to_tensor([total_loss])
            dis_loss = tf.reduce_mean(w * tf.reduce_mean(loss_recons))
            enc_list = self.encoder.trainable_variables
            var_list = self.decoder.trainable_variables
            self.optimizer_encoder.apply_gradients(zip(tape.gradient(total_loss, enc_list), enc_list))
            self.optimizer_decoder.apply_gradients(zip(tape.gradient(dis_loss, var_list), var_list))

        return statistics

    def validate_step(self, x, y):
        statistics = dict()
        batch_dim = tf.shape(y)[0]

        # corrupt the inputs with noise
        if self.is_discrete:
            x_real = disc_noise(x, keep=self.keep, temp=self.temp)
        else:
            x_real = cont_noise(x, self.noise_std)

        # evaluate the discriminator on generated samples
        y_gen = self.encoder(x, training=False)
        loss_erm = (y_gen - y) * (y_gen - y)
        x_gen = self.decoder.sample(y_gen, temp=self.temp, training=False)
        loss_recons = (x - x_gen) * (x - x_gen)
        loss_lips_x = self.penalty(torch.from_numpy(y_gen.cpu().numpy()).to(device),
                                   torch.from_numpy(y.cpu().numpy()).to(device))
        # loss_lips_y = self.penalty(x_gen, x)

        statistics[f'loss_erm'] = loss_erm.cpu().numpy()
        statistics[f'loss_recons'] = loss_recons
        statistics[f'loss_lips_x'] = [loss_lips_x.cpu().detach().numpy()]
        # statistics[f'loss_lips_y'] = loss_lips_y

        return statistics

    def train(self, dataset):
        statistics = defaultdict(list)
        for x, y, w in dataset:
            for name, tensor in self.train_step(x, y, w).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def validate(self, dataset):
        statistics = defaultdict(list)
        for x, y in dataset:
            for name, tensor in self.validate_step(x, y).items():
                statistics[name].append(tensor)
        for name in statistics.keys():
            statistics[name] = tf.concat(statistics[name], axis=0)
        return statistics

    def launch(self, train_data, val_data, logger, epochs=200, start_epoch=0):
        for e in range(start_epoch, start_epoch + epochs):
            for name, loss in self.train(train_data).items():
                logger.record(name, loss, start_epoch + e)
            for name, loss in self.validate(val_data).items():
                logger.record(name, loss, start_epoch + e)
