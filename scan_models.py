import os
import json
import random
import itertools

# Parameters to scan
s_layers        = [2, 4, 6, 8]
s_neurons       = [64, 128, 256]
s_activation    = ['relu']
s_optimizer     = ['adam']
s_lr            = [0.001, 0.0001]
s_batch_size    = [4096]
s_epochs        = [100]
s_dropout       = [-1, 0.2, 0.3]
s_batch_norm    = [0]

df_train_path = 'data/feb02/df_train.h5'
df_val_path   = 'data/feb02/df_val.h5'

output_dir = 'model_scan_feb02'

nrnd = 10

scan = itertools.product(s_layers,
                         s_neurons,
                         s_activation,
                         s_optimizer,
                         s_lr,
                         s_batch_size,
                         s_epochs,
                         s_dropout,
                         s_batch_norm)

l_scan = list(scan)

l_rnd_scan = random.choices(l_scan, k=nrnd)

print(f'Total configurations: {len(l_scan)}, and I will evaluate {len(l_rnd_scan)}')

cols = [
    'layers',
    'neurons',
    'activation',
    'optimizer',
    'lr',
    'batch_size',
    'epochs',
    'dropout',
    'batch_norm',
]


for pars in l_rnd_scan:
    
    s_dict = { c: s for c, s in zip(cols, pars) }

    pars_str = '_'.join([ str(x) for x in pars])

    name = f'scan_{pars_str}'

    print(f'Running {name} ...')

    if os.path.isfile(f'{output_dir}/{name}_conf.json'):
        print('already done. skip!')
        continue

    with open(f'{output_dir}/{name}_conf.json', 'w') as fp:
        json.dump(s_dict, fp)

    cmd = f'python train.py --df_train {df_train_path} --df_val {df_val_path} -c {output_dir}/{name}_conf.json -v 2 --output_name {name} --output_dir {output_dir}'

    os.system(cmd)

