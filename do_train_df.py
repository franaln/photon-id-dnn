import ROOT
import pandas 
import json
import numpy as np
from sklearn.preprocessing import StandardScaler

input_files = [
    # '/mnt/BIG/PhotonID/mini/PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini.root',
    # '/mnt/BIG/PhotonID/mini/Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    'mini/PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    'mini/Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    # '/mnt/BIG/PhotonID/mini/PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini.root',
    # '/mnt/BIG/PhotonID/mini/Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
]

output_dir = 'data/feb16'

do_scaling = True

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

columns_to_read = [ 'event', 'truth_label', 'weight', 'pt', 'eta' ] + shower_shapes


int_cols = ('event', 'is_conv', 'truth_label')
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
for i, path in enumerate(input_files):

    print(f'Adding mini ntuple: {path}')
    
    f = ROOT.TFile.Open(path)

    tree = f.Get('SinglePhoton')
        
    data, columns = tree.AsMatrix(return_labels=True, columns=columns_to_read)
        
    dfs.append(pandas.DataFrame(data=data, columns=columns))
        
    f.Close()

df_all = pandas.concat(dfs, ignore_index=True)
df_all = df_all.astype(type_dict)

del dfs


# Split in train/val (cut at pt<500 because the low stat of fakes at high pt)
# The other events(event%4==3) are left for test
df_samples = {
    'train': df_all.loc[(df_all['event'] % 4 < 2) & (df_all['pt'] < 500)],
    'val':   df_all.loc[df_all['event'] % 4 == 2],
}

del df_all


scale_dict = {}

for sample, df in df_samples.items():

    print(f'Creating df for {sample}')

    # remove shower shapes outliers (FIX?)
    for ss in shower_shapes:
        df = df.drop(df.loc[df[ss] < -10].index)
        df = df.drop(df.loc[df[ss] >  10].index)


    # Split signal/fakes
    df_s = df.loc[df['truth_label'] == 1]
    df_b = df.loc[df['truth_label'] == 0]

    del df

        
    #
    # Re-weighting: weight fakes to match signal in pt/eta bins for train/val sample 
    #
    if sample == 'train':
        pt_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
    else:
        pt_bins = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500, 10000]
        
    eta_bins = [0, 0.6, 0.8, 1.2, 1.37, 1.52, 1.81, 2.01, 2.37]
        
    h_pt_eta_s = np.histogram2d(df_s['pt'], np.abs(df_s['eta']), bins=(pt_bins, eta_bins), weights=df_s['weight'])[0]
    h_pt_eta_b = np.histogram2d(df_b['pt'], np.abs(df_b['eta']), bins=(pt_bins, eta_bins), weights=df_b['weight'])[0]

    weights_pt_eta_b = h_pt_eta_s / h_pt_eta_b
    
    df_s['pt_bin']  = pandas.cut(df_s['pt'], bins=pt_bins, labels=False).astype('int')
    df_b['pt_bin']  = pandas.cut(df_b['pt'], bins=pt_bins, labels=False).astype('int')
        
    df_s['eta_bin']  = pandas.cut(np.abs(df_s['eta']), bins=eta_bins, labels=False).astype('int')
    df_b['eta_bin']  = pandas.cut(np.abs(df_b['eta']), bins=eta_bins, labels=False).astype('int')
    
    df_s['rw'] = df_s['weight']
    df_b['rw'] = df_b['weight'] * weights_pt_eta_b[df_b['pt_bin'],df_b['eta_bin']] 
    
    # merge s/b
    df = pandas.concat([df_s, df_b], ignore_index=True)
    
    del df_s
    del df_b

    # normalize weights to have mean=1
    df['rw'] /= np.mean(df['rw'])

    # Re-scaling
    if do_scaling:

        df['pt'] /= 500.
        df['eta'] = df['eta'].abs() / 2.37

        if sample == 'train':
            w = df['rw'].to_numpy()
            for iss, ss in enumerate(shower_shapes):
                
                X = df[ss].to_numpy().reshape(-1, 1)
                
                scaler = StandardScaler()
                scaler.fit(X, None, w)

                t_mean, t_std = scaler.mean_[0], scaler.scale_[0]
                scale_dict[ss] = (t_mean, t_std)

            with open(f'{output_dir}/scale_conf.json', 'w') as fp:
                json.dump(scale_dict, fp)

        for ss in shower_shapes:
            mean, std = scale_dict[ss]
            print(f'Scaling var = {ss} with mean = {mean}, std = {std}')
            df[ss] -= mean
            df[ss] /= std


    # shuffle before saving 
    df = df.sample(frac=1).reset_index(drop=True)

    # Save dataframe
    outname = f'{output_dir}/df_{sample}.h5' if do_scaling else f'{output_dir}/df_{sample}_raw.h5'
    print(f'Saving df to {outname}')
    df.to_hdf(outname, 'df', format='table')

