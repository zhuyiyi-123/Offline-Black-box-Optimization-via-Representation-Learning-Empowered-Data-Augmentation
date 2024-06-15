from logger import Logger
from data import StaticGraphTask, build_pipeline
from nets import EncoderModel
import numpy as np
import torch
import os
import tensorflow as tf
from nets import ContinuousDecoderModel, DiscreteDecoderModel
from trainers import Autoencoder
from utils import get_weights, TaskDataset
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--task', choices=['TFBind8-Exact-v0', 'Superconductor-RandomForest-v0', 'UTR-ResNet-v0', 'HopperController-Exact-v0', 'DKittyMorphology-Exact-v0'], type=str, 
                                       default='TFBind8-Exact-v0')
parser.add_argument('--task_kwargs', default={"relabel": False}, type=str)
parser.add_argument('--normalize_ys', default=True, type=bool)
parser.add_argument('--normalize_xs', default=True, type=bool)
parser.add_argument('--sample', default=1, type=int)
parser.add_argument('--ratio', default=0.8, type=float)

args = parser.parse_args()


def algor(args):
    # create the training task and logger
    logger = Logger("data")
    task = StaticGraphTask(args.task, **args.task_kwargs)

    if args.normalize_ys:
        task.map_normalize_y()
    if task.is_discrete and not False:
        task.map_to_logits()
    if args.normalize_xs:
        task.map_normalize_x()

    X = task.x
    y = task.y
    N = len(X)
    ratio = args.ratio
    if args.sample==1:
        Y = np.empty([N])
        for i in range(N):
            Y[i] = -y[i]
        index = Y.argsort()
        new_index = index[int(N * ratio):]
        np.random.shuffle(new_index)
        x = torch.Tensor(X[new_index].astype(np.float32)).cuda()
        y = torch.Tensor(-np.expand_dims(Y[new_index], 1).astype(np.float32)).cuda()
    elif args.sample==2:
        Y = np.empty([N])
        for i in range(N):
            Y[i] = -y[i]
        index = Y.argsort()
        new_index = index[int(N * ratio):]
        block_count = int(ratio * len(new_index))
        block_len = len(new_index) // block_count
        new_index = np.delete(new_index, [block_len * i - 1 for i in range(1, block_count + 1)])
        np.random.shuffle(new_index)
        x = torch.Tensor(X[new_index]).cuda()
        y = torch.Tensor(-np.expand_dims(Y[new_index], 1)).cuda()
    elif args.sample==3:
        Y = np.empty([N])
        for i in range(N):
            Y[i] = -y[i]
        index = Y.argsort()
        new_index = index[int(N * ratio):]
        new_index = np.random.choice(new_index, size=int(N * 0.2), replace=False)
        np.random.shuffle(new_index)
        x = torch.Tensor(X[new_index]).cuda()
        y = torch.Tensor(-np.expand_dims(Y[new_index], 1)).cuda()
    
    input_shape = x.shape[1:]

    # make several encoder and decoder neural networks with two hidden layers
    encoder = EncoderModel(
        input_shape,
        activations=['leaky_relu', 'leaky_relu'],
        hidden=2048)
    # encoder = EncoderModel(np.prod(x.shape[1:]), 2048, np.prod(y.shape[1:]))
    if task.is_discrete:
        decoder = DiscreteDecoderModel(input_shape, latent_size=32, hidden=2048)
    else:
        decoder = ContinuousDecoderModel(input_shape, latent_size=32, hidden=2048)

    # create a trainer for encoder and decoder
    trainers = Autoencoder(encoder, decoder, optimizer=torch.optim.Adam, learning_lr=0.001, input_size=input_shape)

    # build a weighted data set using collected samples
    x_cpu = x.cpu()
    y_cpu = y.cpu()
    train_data, val_data = build_pipeline(x=x_cpu.numpy().astype(np.float32),
                                          y=y_cpu.numpy().astype(np.float32),
                                          w=get_weights(y_cpu.numpy()),
                                          batch_size=128, val_size=500, buffer=1)

    # train the model for several epochs
    initial_epochs = 200
    trainers.launch(train_data, val_data, logger, initial_epochs)

    a_x = []
    a_y = []
    max_y = max(y)
    a = np.where(y.cpu().numpy() == max_y.cpu().numpy())[0]
    max_x = tf.convert_to_tensor(x[a].cpu())
    for i in range(200):
        with tf.GradientTape() as tape:
            tape.watch(max_x)
            model = encoder(max_x)
        grads = tape.gradient(model, max_x)
        learning_rate = 0.01
        max_x = max_x + learning_rate * grads
        max_y = encoder(max_x)
        if i == 0:
            max_xxx = max_x
            max_yy = max_y
        max_x = tf.cast(max_x, dtype=tf.float32)
        max_y = tf.cast(max_y, dtype=tf.float32)
        max_xxx = tf.concat([max_xxx, max_x], axis=0)
        max_yy = tf.concat([max_yy, max_y], axis=0)
        if grads.cpu().numpy().all() <= 0.01:
            break
    max_y = encoder(max_x)
    max_xx = decoder.sample(max_y)
    recon_err = np.zeros((len(max_xx)))
    au_x = np.zeros((20, len(x[0])))
    au_y = np.zeros((20, 1))
    a = 0
    for i in range(len(max_xx)):
        max_x_float = torch.tensor(max_xx[i].cpu().numpy().astype(np.float64)).cuda()
        loss = torch.tensor(max_x[i].cpu().numpy()).cuda() - max_x_float
        recon_err[i] = tf.reduce_mean(tf.convert_to_tensor((loss * loss).cpu().numpy()))
        if recon_err[i] <= 1000:
            a = a + 1
        steps = 200
        w = 0
        q = 0
        p = 0
        r = 0
        model = True
    while a < 20:
        print(a, steps)
        steps = steps - 10
        if steps <= 0:
            steps = 200
            w = w + a
            m = 0
            p = p + 1
            if w <= 20:
                sorted_indices = np.argsort(recon_err)
                min_20_indices = sorted_indices[:a]
                for j in range(a):
                    au_x[q + j] = max_x[min_20_indices[j]]
                    au_y[q + j] = max_y[min_20_indices[j]]
                q = q + a
            elif m == 0:
                a = 20 - q
                sorted_indices = np.argsort(recon_err)
                min_20_indices = sorted_indices[:a]
                for j in range(a):
                    au_x[q + j] = max_x[min_20_indices[j]]
                    au_y[q + j] = max_y[min_20_indices[j]]
                model = False
                break
            arr = np.array(max_yy.numpy())
            sorted_arr = np.sort(arr)[::-1]
            r = 1
            max_y = sorted_arr[r]
            indices = np.where(arr == max_y)[0]
            if len(indices) != 1:
                max_x = tf.convert_to_tensor(max_xxx[indices[0]].cpu().numpy())
            else:
                max_x = tf.convert_to_tensor(max_xxx[indices[0]].cpu().numpy())
            r = r + 1
        for i in range(steps):
            with tf.GradientTape() as tape:
                tape.watch(max_x)
                if len(max_x) == len(x[0]):
                    model = encoder(tf.reshape(max_x, shape=(1,len(x[0]))))
                else:
                    model = encoder(tf.reshape(max_x, shape=(len(max_x),len(x[0]))))
            grads = tape.gradient(model, max_x)
            learning_rate = 0.01
            max_x = max_x + learning_rate * grads
            # if max(grads.cpu().numpy().all()) <= 0.01:
            #     break
        if len(max_x) == len(x[0]):
            max_y = encoder(tf.reshape(max_x, shape=(1,len(x[0]))))
        else:
            max_y = encoder(tf.reshape(max_x, shape=(len(max_x),len(x[0]))))
        max_xx = decoder.sample(max_y)
        recon_err = np.zeros((len(max_xx)))
        a = 0
        for i in range(len(max_xx)):
            max_x_float = torch.tensor(max_xx[i].cpu().numpy().astype(np.float64)).cuda()
            loss = torch.tensor(max_x[i].cpu().numpy()).cuda() - max_x_float
            recon_err[i] = tf.reduce_mean(tf.convert_to_tensor((loss * loss).cpu().numpy()))
            if recon_err[i] <= 1000:
                a = a + 1
        logger.record("score", max_y, steps + 200 * p, percentile=True)

    if model:
        sorted_indices = np.argsort(recon_err)
        min_20_indices = sorted_indices[:20]
        for j in range(20):
            au_x[j] = max_x[min_20_indices[j]]
            au_y[j] = max_y[min_20_indices[j]]
        # if task.is_normalized_y:
        #     if task.is_normalized_y:
        #         au_y[j] = task.denormalize_y(au_y[j].astype(np.float32))
        #     if task.is_normalized_x:
        #         au_x[j] = task.denormalize_y(au_x[j].astype(np.float32))
    save_name = args.task + '-' + args.sample + '-' +args.ratio
    df_x = pd.DataFrame(au_x, columns=['x_' + str(i) for i in range(len(x[0]))])
    df_y = pd.DataFrame(au_y, columns=['y'])
    df_xy = pd.concat([df_x, df_y], axis=1)   
    df_xy.to_csv(os.path.join(save_name + '.csv'))

if __name__=="__main__":
    algor(args)
