import sys
import pandas
import numpy as np
from tensorflow.keras import models
import ROOT
import json

mini_dir = 'mini/'

mini_files = [
    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
]

scale_path  = 'data/feb02/scale_conf.json'
model_path  = 'train_feb15/best_model.h5'
output_path = 'df_test__train_feb15_2NN.h5'

columns_to_read = [
    'event',
    'truth_label',
    'pt',
    'eta',
    'mu',
    'is_tight',
    'is_conv',
    'weight',
    'rhad_mixed',
    'reta', 
    'rphi', 
    'weta2', 
    'eratio', 
    'deltae', 
    'weta1', 
    'wtots1', 
    'fracs1', 
    'is_looseprime4',
    'is_isoloose',
    'is_isotight',
    'iso_calo',
    'iso_track',
]

int_cols = ('event', 'is_conv', 'truth_label', 'is_loose', 'is_tight', 'is_looseprime4', 'is_isoloose', 'is_isotight')

type_dict = {}
for col in columns_to_read:
    if col == 'event':
        type_dict[col] = 'int64'
    elif col in int_cols:
        type_dict[col] = 'int32'
    else:
        type_dict[col] = 'float32'


# Read and convert ntuples to pandas dataframe
dfs = []
for i, path in enumerate(mini_files):

    print(f'Adding mini ntuple: {path}')
    
    f = ROOT.TFile.Open(mini_dir+path)
    
    tree = f.Get('SinglePhoton')
    
    data, columns = tree.AsMatrix(return_labels=True, columns=columns_to_read)
    
    df = pandas.DataFrame(data=data, columns=columns)
    df = df.astype(type_dict)
    df = df.loc[df['event'] % 4 == 3]

    dfs.append(df)
    
    f.Close()


df = pandas.concat(dfs, ignore_index=True)

del dfs


# 
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

shower_shapes = [
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

with open(scale_path) as f:
    scale_dict = json.load(f)


df['n_pt'] = df['pt'] / 500.
df['n_eta'] = df['eta'].abs() / 2.37

for ss in shower_shapes:
    mean, std = scale_dict[ss]
    df[f'n_{ss}'] = df[ss] - mean
    df[f'n_{ss}'] /= std


x = df[input_vars].to_numpy()

model = models.load_model(model_path)
#model.summary()

df['output'] = model.predict(x)


# # output w/o sigmoid activation
# model.layers[-1].activation = None

# new_path = model_path.replace('.h5', '_wosigmoid.h5')
# model.save(new_path)
# new_model = models.load_model(new_path)


input_vars_1 = [
    'n_pt',
    'n_eta',
    'n_rhad_mixed',
    'n_reta',
    'n_rphi',
    'n_weta2',
    'n_wtots1',
]

input_vars_2 = [
    'n_pt',
    'n_eta',
    'n_eratio',
    'n_deltae',
    'n_weta1',
    'n_fracs1',
]

x1 = df[input_vars_1].to_numpy()
x2 = df[input_vars_2].to_numpy()

model_2NN_1 = models.load_model('train_feb15_2NN/best_model_1.h5')
model_2NN_2 = models.load_model('train_feb15_2NN/best_model_2.h5')

df['output_2NN_1'] = model_2NN_1.predict(x1)
df['output_2NN_2'] = model_2NN_2.predict(x2)


df.to_hdf(output_path, 'df', format='table')
