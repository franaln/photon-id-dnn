import ROOT
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# -------------
# Configuration
# -------------

mini_dir = '/mnt/R5/ATLAS/PhotonID/SP_mini/'
output_dir = 'data/'
scale_path = 'data/scale_conf.json'

shower_shapes = [
    'rhad_mixed', 'reta', 'rphi', 'weta2',
    'eratio', 'deltae', 'weta1', 'wtots1', 'fracs1',
]

variables_train = [ 'event', 'truth_label', 'weight', 'pt', 'eta', 'is_conv', ] + shower_shapes
variables_test  = variables_train + [ 'is_looseprime4', 'is_isoloose', 'is_isotight', 'iso_calo', 'iso_track', 'mu', 'is_tight' ]

int_cols = ('event', 'is_conv', 'truth_label', 'is_loose', 'is_tight', 'is_looseprime4', 'is_isoloose', 'is_isotight')

mini_files_train = [
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
]

mini_files_test = [
    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
]
        
# -------------

for sample in ('train', 'val', 'test'):
    
    if sample in ('train', 'val'):
        columns_to_read = variables_train
        mini_files = mini_files_train
    else:
        columns_to_read = variables_test
        mini_files = mini_files_test

    type_dict = {}
    for col in columns_to_read:
        if col == 'event':
            type_dict[col] = 'int64'
        elif col in int_cols:
            type_dict[col] = 'int32'
        else:
            type_dict[col] = 'float32'

         
    print('Creating df for %s ...' % (sample))

    # Read and convert ntuples to pd dataframe
    dfs = []
    for i, path in enumerate(mini_files):

        print(f'Adding mini ntuple: {path}')

        f = ROOT.TFile.Open(mini_dir + path)

        tree = f.Get('SinglePhoton')

        data, columns = tree.AsMatrix(return_labels=True, columns=columns_to_read)

        df = pd.DataFrame(data=data, columns=columns)
        df = df.astype(type_dict)

        if sample == 'train':
            df = df.loc[(df['event'] % 4 < 2) & (df['pt'] < 500)]
        elif sample == 'val':
            df = df.loc[df['event'] % 4 == 2]
        elif sample == 'test':
            df = df.loc[df['event'] % 4 == 3]

        dfs.append(df)

        f.Close()

    df = pd.concat(dfs, ignore_index=True)
    del dfs


    # remove shower shapes outliers (FIX?)
    if sample == 'train':
        for ss in shower_shapes:
            df = df.drop(df.loc[df[ss] < -10].index)
            df = df.drop(df.loc[df[ss] >  10].index)

    #
    # Re-weighting: weight fakes to match signal in pt/eta bins for train/val sample
    #
    if sample == 'train' or sample == 'val':
        if sample == 'train':
            pt_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
        else:
            pt_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 10000]

        eta_bins = [0, 0.6, 0.8, 1.2, 1.37, 1.52, 1.81, 2.01, 2.37]
    
        h_pt_eta_s = np.histogram2d(df.loc[df['truth_label'] == 1,'pt'], np.abs(df.loc[df['truth_label'] == 1,'eta']), bins=(pt_bins, eta_bins), weights=df.loc[df['truth_label'] == 1,'weight'])[0]
        h_pt_eta_b = np.histogram2d(df.loc[df['truth_label'] == 0,'pt'], np.abs(df.loc[df['truth_label'] == 0,'eta']), bins=(pt_bins, eta_bins), weights=df.loc[df['truth_label'] == 0,'weight'])[0]

        weights_pt_eta_b = h_pt_eta_s / h_pt_eta_b

        df['pt_bin']   = pd.cut(df['pt'], bins=pt_bins, labels=False).astype('int')
        df['eta_bin']  = pd.cut(np.abs(df['eta']), bins=eta_bins, labels=False).astype('int')
  
        df['rw'] = np.where(df['truth_label'] == 1, df['weight'], df['weight'] * weights_pt_eta_b[df['pt_bin'],df['eta_bin']])

        # normalize weights to have mean=1
        df['rw'] /= np.mean(df['rw'])
    
        df = df.drop(columns=['pt_bin', 'eta_bin'])


        # Add scaled columns
        df['n_pt'] = df['pt'] / 500.
        df['n_eta'] = df['eta'] / 2.37

        if sample == 'train':

            w = df['rw'].to_numpy()
            scale_dict = dict()
            for iss, ss in enumerate(shower_shapes):
                
                X = df[ss].to_numpy().reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(X, None, w)

                t_mean, t_std = scaler.mean_[0], scaler.scale_[0]
                scale_dict[ss] = (t_mean, t_std)

            with open(scale_path, 'w') as fp:
                json.dump(scale_dict, fp)

                
        with open(scale_path) as f:
            scale_dict = json.load(f)

        for ss in shower_shapes:
            mean, std = scale_dict[ss]
            print(f'Scaling var = {ss} with mean = {mean}, std = {std}')
            df[f'n_{ss}'] = (df[ss] - mean) / std


    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save
    print('Saving df to df_%s.h5' % sample) 
    df.to_hdf(f'{output_dir}/df_%s.h5' % sample, 'df', format='table')



