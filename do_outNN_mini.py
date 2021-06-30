import sys
import numpy as np
import utils
import uproot
from array import array
import ROOT

files = [
    '/mnt/R5/ATLAS/PhotonID/SP_mini/PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/data15_276262_284484_GRL_p3930_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/data16_297730_311481_GRL_p3930_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/data17_325713_340453_GRL_p3930_Rel21_AB21.2.94_v0_mini.root',
    '/mnt/R5/ATLAS/PhotonID/SP_mini/data18_348885_364292_GRL_p3930_Rel21_AB21.2.94_v0_mini.root',
]

output_files = [
    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0_mini_NN.root',
    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0_mini_NN.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0_mini_NN.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0_mini_NN.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0_mini_NN.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0_mini_NN.root',
    'data15_276262_284484_GRL_p3930_Rel21_AB21.2.94_v0_mini_NN.root',
    'data16_297730_311481_GRL_p3930_Rel21_AB21.2.94_v0_mini_NN.root',
    'data17_325713_340453_GRL_p3930_Rel21_AB21.2.94_v0_mini_NN.root',
    'data18_348885_364292_GRL_p3930_Rel21_AB21.2.94_v0_mini_NN.root',
]

model_path = 'trainings/train_jun11_DisCo_alpha25_iso_calo20/best_model.h5'
model_name = 'nn_disco25_iso_calo'

scale_path = 'data/scale_conf.json'

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

model = utils.load_model(model_path)

for path, outpath in zip(files, output_files):

    print(f'Processing mini ntuple: {path} ...')

    input_vars = ['pt', 'eta'] + shower_shapes

    f = uproot.open(path)
    tree = f['SinglePhoton']

    nentries = tree.num_entries

    of = ROOT.TFile(outpath, 'recreate')
    ftree = ROOT.TTree("nn", "nn")

    b_output = array('f', [0.])
    ftree.Branch('nn', b_output, f'nn/F')

    counter = 0
    for df in tree.iterate(input_vars, step_size=1000000, library='pd'):

        print(f'Processing {counter}/{nentries}...')

        X = df[input_vars].to_numpy()

        utils.scale_input_variables(X, scale_path)

        n = X.shape[0]

        output = model.predict(X) #.reshape(n)

        counter += n

        for ie in range(n):
            b_output[0] = output[ie]
            ftree.Fill()

    ftree.Write("nn", ROOT.TObject.kOverwrite)



