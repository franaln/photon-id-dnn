import os
import pickle
import argparse
import json
import pandas
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers
from tensorflow.keras import losses
from tensorflow.python.keras.utils import losses_utils
import tensorflow.keras.backend as K

from disco import distance_corr

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('--df_train', help='Input train file')
parser.add_argument('--df_val', help='Input val file')
parser.add_argument('-c', '--conf', help='Configuration')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('-v', '--verbose', type=int, default=2)
parser.add_argument('-g', '--generator', action='store_true')
parser.add_argument('-s', '--selection')

args = parser.parse_args()


#--------
# Config
#--------
df_train_path = args.df_train
df_val_path   = args.df_val

with open(args.conf, "r") as read_file:
    conf_dict = json.load(read_file)

nlayers = int(conf_dict['layers'])

if isinstance(conf_dict['neurons'], str) and ',' in conf_dic['neurons']:
    neurons = [ int(n) for n in conf_dict['neurons'].split(',') ] 
else:
    neurons = [ int(conf_dict['neurons']) for i in range(nlayers) ]

activation = conf_dict['activation']
optimizer  = conf_dict['optimizer']
lr         = float(conf_dict['lr'])
dropout    = float(conf_dict['dropout'])
batch_size = int(conf_dict['batch_size'])
epochs     = int(conf_dict['epochs'])

disco_alpha = int(conf_dict['disco_alpha'])
disco_var = conf_dict['disco_var']

if args.output_dir:
    output_dir = args.output_dir
    os.system(f'mkdir -p {output_dir}')
else:
    output_dir = '.'

output_name = f'{args.output_name}_' if args.output_name else ''

use_generator = args.generator and args.selection is None


os.system(f'cp {args.conf} {output_dir}/{args.conf}')

# ---------------
# Data generator
# ---------------
input_vars = [
    'n_pt',
    'n_eta',
    'n_rhad_mixed',
    'n_reta',
    'n_rphi',
    'n_weta2',
    'n_eratio',
    'n_deltae',
    'n_weta1',
    'n_wtots1',
    'n_fracs1',
]

cols_to_read = ['truth_label', 'rw', 'is_conv'] + input_vars + [disco_var,]

def get_steps(df_path, batch_size):
    df = pandas.read_hdf(df_path, columns=['event'])
    print(len(df))
    return int(np.ceil(len(df) / batch_size))

def data_generator(df_path, batch_size, steps):
    i = 0
    while True:
        # start from the begginning when reach number of steps
        if i >= steps:
            i = 0

        start_row = i*batch_size
        stop_row  = (i+1)*batch_size
        i += 1

        df = pandas.read_hdf(df_path, columns=cols_to_read, start=start_row, stop=stop_row)

        yield (df[input_vars].to_numpy(), df[['truth_label', disco_var, 'rw']].to_numpy())


if use_generator:
    steps_train = get_steps(df_train_path, batch_size)
    steps_val   = get_steps(df_val_path,   batch_size)

    data_train = data_generator(df_train_path, batch_size, steps_train)
    data_val   = data_generator(df_val_path,   batch_size, steps_val)

else:
    df_train = pandas.read_hdf(df_train_path, columns=cols_to_read)
    df_val   = pandas.read_hdf(df_val_path, columns=cols_to_read)

    if args.selection is not None:
        df_train = df_train.query(args.selection)
        df_val   = df_val.query(args.selection)
    
    x_train = df_train[input_vars].to_numpy()
    x_val = df_val[input_vars].to_numpy()
    
    y_train = df_train[['truth_label', disco_var, 'rw']].to_numpy()
    y_val   = df_val[['truth_label', disco_var, 'rw']].to_numpy()


# -----
# DisCo
# -----
def DisCo(X, Y, w=None):

    if w is None:
        w = K.ones_like(X)

    LX = K.shape(X)[0]
    LY = K.shape(Y)[0]

    X = K.reshape(X, shape=(LX,1))
    Y = K.reshape(Y, shape=(LY,1))

    ajk = K.abs(K.reshape(K.repeat(X,LX), shape=(LX,LX)) - K.transpose(X))
    bjk = K.abs(K.reshape(K.repeat(Y,LY), shape=(LY,LY)) - K.transpose(Y))

    ajk_mean = K.mean(ajk*w, axis=1)
    bjk_mean = K.mean(bjk*w, axis=1)

    ajk_1 = K.reshape(K.tile(ajk_mean, [LX]), shape=(LX,LX))
    ajk_2 = K.transpose(ajk_1)

    bjk_1 = K.reshape(K.tile(bjk_mean, [LY]), shape=(LY,LY))
    bjk_2 = K.transpose(bjk_1)

    Ajk = ajk - ajk_1 - ajk_2 + K.mean(ajk_mean*w)
    Bjk = bjk - bjk_1 - bjk_2 + K.mean(bjk_mean*w)

    AB_mean = K.mean(Ajk*Bjk*w, axis=1)
    AA_mean = K.mean(Ajk*Ajk*w, axis=1)
    BB_mean = K.mean(Bjk*Bjk*w, axis=1)

    dcor = K.mean(AB_mean*w) / K.sqrt(K.mean(AA_mean*w)*K.mean(BB_mean*w))

    if np.isnan(dcor):
        dcor = 0

    return dcor


def loss_disco(alpha):
    def loss(y_extended, y_pred):

        y_true = tf.reshape(y_extended[:,0], (-1,1))
        x      = tf.reshape(y_extended[:,1], (-1,1))
        w      = tf.reshape(y_extended[:,2], (-1,1))

        mask = tf.where(y_true<1, K.ones_like(x), K.zeros_like(x))

        X = tf.boolean_mask(x,      mask)
        Y = tf.boolean_mask(y_pred, mask)
        # W = tf.boolean_mask(w,      mask)

        N = K.shape(Y)[0]
        # f = float(N) / K.sum(W)
        # W = tf.scalar_mul(f, W)

        bce = losses_utils.compute_weighted_loss(
            losses.binary_crossentropy(y_true, y_pred),
            sample_weight=w
        )

        if alpha == 0:
            return bce

        dcor = DisCo(X, Y)

        return bce + alpha * dcor

    return loss


# custom metrics (needed to use custom labels)
def metric_auc():
    m = tf.keras.metrics.AUC()
    def auc(y_extended, y_pred):
        y_true  = tf.reshape(y_extended[:,0], (-1,1))
        weights = tf.reshape(y_extended[:,2], (-1,1))
        m.update_state(y_true, y_pred, sample_weight=weights)
        return m.result()
    return auc

def metric_disco_b():
    def disco_b(y_extended, y_pred):

        y_true = tf.reshape(y_extended[:,0], (-1,1))
        x      = tf.reshape(y_extended[:,1], (-1,1))
        w      = tf.reshape(y_extended[:,2], (-1,1))

        mask = tf.where(y_true<1, K.ones_like(x), K.zeros_like(x))

        X = tf.boolean_mask(x,      mask)
        Y = tf.boolean_mask(y_pred, mask)

        # if disco_use_weights:
        #     W = tf.boolean_mask(w,      mask)

        #     f = float(K.shape(Y)[0]) / K.sum(W)
        #     W = tf.scalar_mul(f, W)
        #     dcor = DisCo(X, Y, W)
        # else:
        dcor = DisCo(X, Y)

        return dcor

    return disco_b

def metric_disco_s():
    def disco_s(y_extended, y_pred):

        y_true = tf.reshape(y_extended[:,0], (-1,1))
        x      = tf.reshape(y_extended[:,1], (-1,1))
        w      = tf.reshape(y_extended[:,2], (-1,1))

        mask = tf.where(y_true>0, K.ones_like(x), K.zeros_like(x))

        X = tf.boolean_mask(x,      mask)[:100]
        Y = tf.boolean_mask(y_pred, mask)[:100]

        dcor = DisCo(X, Y)

        return dcor

    return disco_s

def metric_bce():
    m = tf.keras.metrics.BinaryCrossentropy()
    def bce(labels, y_pred):
        y_true  = tf.reshape(labels[:,0], (-1,1))
        weights = tf.reshape(labels[:,2], (-1,1))
        m.update_state(y_true, y_pred, sample_weight=weights)
        return m.result()
    return bce


#-------
# Model
#-------

model = models.Sequential()

for i in range(nlayers):
    if activation == 'relu':
        act = 'relu'
    elif activation == 'leakyrelu':
        act = layers.LeakyReLU()

    if i == 0:
        model.add(layers.Dense(neurons[i], activation=act, input_shape=(len(input_vars),)))
    else:
        model.add(layers.Dense(neurons[i], activation=act))

    if dropout > 0:
        model.add(layers.Dropout(dropout))
       
model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))

if optimizer == 'adam':
    optimizer = optimizers.Adam(lr=lr)
elif optimizer == 'rmsprop':
    optimizer = optimizers.RMSprop(lr=lr)

metrics_to_monitor = [
    metric_auc(),
    metric_bce(),
    metric_disco_b(),
    metric_disco_s(),
]

model.compile(
    optimizer=optimizer, 
    loss=loss_disco(disco_alpha),
    metrics=metrics_to_monitor,
    run_eagerly=True
)

model.summary()


#-----------
# Callbacks
#-----------
cb_checkpoint = callbacks.ModelCheckpoint(f'{output_dir}/{output_name}best_model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto', period=1)

cb_earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10) 


#----------
# Training
#----------
if use_generator:
    history = model.fit(data_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        steps_per_epoch=steps_train,
                        validation_data=data_val,
                        validation_steps=steps_val,
                        verbose=args.verbose,
                        callbacks=[cb_checkpoint, cb_earlystopping])
else:
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        verbose=args.verbose,
                        callbacks=[cb_checkpoint, cb_earlystopping])


with open(f'{output_dir}/{output_name}history.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)


model.save(f'{output_dir}/{output_name}model.h5')
