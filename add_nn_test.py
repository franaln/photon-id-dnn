import sys
import pandas
import numpy as np
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.python.keras.utils import losses_utils
import ROOT
import json

input_path  = 'data/df_test.h5'
scale_path  = 'data/scale_conf.json'
output_path = 'output_jun18/df_test_output.h5'

models_dict = {
    'nn_baseline': 'trainings/train_may02_baseline/best_model.h5',

    'nn_disco50':  'trainings/train_may02_DisCo_alpha50/best_model.h5',
    'nn_disco25':  'trainings/train_may02_DisCo_alpha25/best_model.h5',
    'nn_disco15':  'trainings/train_may02_DisCo_alpha15/best_model.h5',
    'nn_disco10':  'trainings/train_may02_DisCo_alpha10/best_model.h5',
    'nn_disco10':  'trainings/train_may02_DisCo_alpha10/best_model.h5',
    'nn_disco10_calo': 'trainings/train_may02_DisCo_alpha10_iso_calo/best_model.h5',
    'nn_disco25_calo': 'trainings/train_jun11_DisCo_alpha25_iso_calo20/best_model.h5',
}

df = pandas.read_hdf(input_path, 'df')

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

x = df[input_vars].to_numpy()

with open(scale_path) as f:
    scale_dict = json.load(f)


# x[:,0] /= 500.
# x[:,1] = np.abs(x[:,1]) / 2.37

x[:,0] /= 250.
x[:,1] /= 2.37

for iss, ss in enumerate(input_vars):
    if ss not in scale_dict:
        continue
    if ss in ('pt', 'eta'):
        continue
    mean, std = scale_dict[ss]
    x[:,iss] -= mean
    x[:,iss] /= std


def loss_disco(alpha):
    def loss(y_ext, y_pred):
        return 0.
    return loss

for model_name, model_path in models_dict.items():
    
    if 'DisCo' in model_path:
        model = models.load_model(model_path, 
                                  compile=False, custom_objects={'loss_disco': loss_disco(0)})
    else:
        model = models.load_model(model_path)

    df[model_name] = model.predict(x, batch_size=4096)



# output w/o sigmoid activation
for model_name, model_path in models_dict.items():
    
    if 'DisCo' in model_path:
        model = models.load_model(model_path, 
                                  compile=False, custom_objects={'loss_disco': loss_disco(0)})
    else:
        model = models.load_model(model_path)

    model.layers[-1].activation = None

    new_path = model_path.replace('.h5', '_wosigmoid.h5')
    model.save(new_path)
    new_model = models.load_model(new_path)

    df[f'{model_name}_wosigmoid'] = model.predict(x, batch_size=4096)


df.to_hdf(output_path, 'df', format='table')


