import os
import pickle
import argparse
import json
import pandas
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import regularizers

parser = argparse.ArgumentParser(description='train.py')

parser.add_argument('--df_train', help='Input train file')
parser.add_argument('--df_val', help='Input val file')
parser.add_argument('-c', '--conf', help='Configuration')
parser.add_argument('--output_dir', type=str)
parser.add_argument('--output_name', type=str)
parser.add_argument('-v', '--verbose', type=int, default=2)

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
batch_norm = int(conf_dict['batch_norm'])


if args.output_dir:
    output_dir = args.output_dir
    os.system(f'mkdir -p {output_dir}')
else:
    output_dir = '.'

output_name = f'{args.output_name}_' if args.output_name else ''


# ---------------
# Data generator
# ---------------
input_vars = [
    'pt',
    'eta',
    'rhad_mixed',
    'reta',
    'rphi',
    'weta2',
    'eratio',
    'deltae',
    'weta1',
    'wtots1',
    'fracs1',
]

cols_to_read = ['truth_label', 'rw',] + input_vars

def get_steps(df_path, batch_size):
    df = pandas.read_hdf(df_path, columns=['event'])
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

        yield (df[input_vars].to_numpy(), df['truth_label'].to_numpy(), df['rw'].to_numpy())


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

    if batch_norm == 1:
        model.add(layers.BatchNormalization())

    if dropout > 0:
        model.add(layers.Dropout(dropout))
       
model.add(layers.Dense(1, activation='sigmoid', dtype='float32'))

if optimizer == 'adam':
    optimizer = optimizers.Adam(lr=lr)
elif optimizer == 'rmsprop':
    optimizer = optimizers.RMSprop(lr=lr)

metrics_to_monitor = [
    # metrics.Recall(name='recall'),
    metrics.AUC(name='auc'),        
]

metrics_to_monitor_w = [
    # metrics.Recall(name='recall'),
    metrics.AUC(name='auc'),        
]

model.compile(
    optimizer=optimizer, 
    loss='binary_crossentropy', 
    metrics=metrics_to_monitor,
    weighted_metrics=metrics_to_monitor_w,
)

model.summary()

#-----------
# Callbacks
#-----------
cb_checkpoint = callbacks.ModelCheckpoint(f'{output_dir}/{output_name}best_model.h5', monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='auto', period=1)

cb_earlystopping = callbacks.EarlyStopping(monitor='val_loss', patience=10) 


#------
# Data
#------
steps_train = get_steps(df_train_path, batch_size)
steps_val   = get_steps(df_val_path,   batch_size)

data_train = data_generator(df_train_path, batch_size, steps_train)
data_val   = data_generator(df_val_path,   batch_size, steps_val)


#----------
# Training
#----------
history = model.fit(data_train,
                    epochs=epochs,
                    batch_size=batch_size,
                    steps_per_epoch=steps_train,
                    validation_data=data_val,
                    validation_steps=steps_val,
                    verbose=args.verbose,
                    callbacks=[cb_checkpoint, cb_earlystopping])

with open(f'{output_dir}/{output_name}history.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

model.save(f'{output_dir}/{output_name}model.h5')
