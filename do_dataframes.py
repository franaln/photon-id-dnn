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

samples = ('train', 'val', 'test')

shower_shapes = [
    'rhad_mixed', 'reta', 'rphi', 'weta2',
    'eratio', 'deltae', 'weta1', 'wtots1', 'fracs1',
]

variables = [ 
    'event', 'mcid', 'truth_label',  'weight', 
    'pt', 'eta', 'mu', 'is_conv', 'conv_type',
    'iso_calo20', 'iso_calo40', 'iso_track',  'is_isoloose', 'is_isotight', 'is_isotightcaloonly',
    'is_tight', 'is_looseprime4',
    'f1',
    ##'conv_radius', 'E1E2', 'maxEcell_E', 'maxEcell_time',  'e277',
] + shower_shapes

int_cols = ('event', 'is_conv', 'truth_label', 'is_loose', 'is_tight', 'is_looseprime4', 'is_isoloose', 'is_isotight', 'is_isotightcaloonly', 'conv_type')

mini_files_train = [
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
]

mini_files_test = [
    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
]

selection = {
    'train': '(event % 4 < 2)  & (pt < 250) & (f1>0.005)',
    'val':   '(event % 4 == 2) & (pt < 250) & (f1>0.005)',
    'test':  '(event % 4 == 3) & (f1>0.005)',
}

# -------------

for sample in samples:

    is_train_val = sample != 'test'
    
    if is_train_val:
        mini_files = mini_files_train
    else:
        mini_files = mini_files_test

    type_dict = {}
    for col in variables:
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

        data, columns = tree.AsMatrix(return_labels=True, columns=variables)

        df = pd.DataFrame(data=data, columns=columns)
        df = df.astype(type_dict)

        df = df.query(selection[sample])

        dfs.append(df)

        f.Close()

    df = pd.concat(dfs, ignore_index=True)
    del dfs


    # remove shower shapes outliers (FIX?)
    if is_train_val:
        for ss in shower_shapes:
            n = len(df.loc[(df[ss] < -10)].index) + len(df.loc[(df[ss] >  10)].index)
            print(f'Removing {n} outliers events for {ss}') 
            df = df.drop(df.loc[df[ss] < -10].index)
            df = df.drop(df.loc[df[ss] >  10].index)

    #
    # Re-weighting: weight fakes to match signal in pt/eta bins for train/val sample
    #
    if is_train_val:
        pt_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250,] ## 275, 300, 350, 400,] ## 450, 500]
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
        df['n_pt'] = df['pt'] / 250.
        df['n_eta'] = df['eta'] / 2.37

        if sample == 'train':

            w = df['rw'].to_numpy()
            scale_dict = dict()
            for iss, ss in enumerate(shower_shapes+['n_pt', 'n_eta']):
                
                X = df[ss].to_numpy().reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(X, None, w)

                t_mean, t_std = round(scaler.mean_[0], 4), round(scaler.scale_[0], 4)
                scale_dict[ss] = (t_mean, t_std)

            with open(scale_path, 'w') as fp:
                json.dump(scale_dict, fp)

                
        with open(scale_path) as f:
            scale_dict = json.load(f)

        for ss in shower_shapes+['n_pt', 'n_eta']:
            mean, std = scale_dict[ss]
            print(f'Scaling var {ss} with mean = {mean:.4f}, std = {std:.4f}')
            df[f'n_{ss}'] = (df[ss] - mean) / std


    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Save
    print(f'Saving df to {output_dir}/df_{sample}.h5') 
    df.to_hdf(f'{output_dir}/df_{sample}.h5', 'df', format='table')



