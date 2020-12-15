import ROOT
import sys
import os

from datetime import datetime

ntuples_dir = '/mnt/BIG/PhotonID/SP_ntuples/'

files = [
    # '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/data15_276262_284484_GRL_p3930_Rel21_AB21.2.94_v0.root',
    # '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/data16_297730_311481_GRL_p3930_Rel21_AB21.2.94_v0.root',
    # '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/data17_325713_340453_GRL_p3930_Rel21_AB21.2.94_v0.root',
    # '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/data18_348885_364292_GRL_p3930_Rel21_AB21.2.94_v0.root'

    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0.root',

    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0.root',
]

cols = ROOT.std.vector('string')()

# l = [
#     'mcid',
#     'evt_runNo',
#     'evt_eventNo',

#     'intLumi', 
#     'mcTotWeightNoPU_PIDuse',
#     'puWeight',

#     'evt_mu',

#     'y_pt', 
#     'y_eta_cl_s2', 
#     'y_phi', 
#     'y_convType', 
#     'y_isTruthMatchedPhoton', 
#     'y_iso_FixedCutLoose', 
#     'y_iso_FixedCutTight', 
#     'y_iso_FixedCutTightCaloOnly', 
#     'y_IsTight', 
#     'y_IsLoose', 
#     'y_Reta', 
#     'y_Rphi', 
#     'y_weta2', 
#     'y_fracs1', 
#     'y_weta1', 
#     'y_emaxs1', 
#     'y_f1', 
#     'y_wtots1', 
#     'y_Rhad', 
#     'y_Rhad1', 
#     'y_Eratio', 
#     'y_e277', 
#     'y_deltae', 
# ]

l = [
    'event',
    'weight', 
    'mu',

    'pt', 
    'eta', 
    'phi', 
    'is_conv', 
    'truth_label',

    'is_tight', 
    'is_looseprime4',
    'is_isoloose',
    'is_isotight',
    'iso_calo',
    'iso_track',
    
    'reta', 
    'rphi', 
    'weta2', 
    'fracs1', 
    'weta1', 
    'wtots1', 
    'rhad', 
    'eratio', 
    'deltae', 
    'f1', 
]

for i in l:
    cols.push_back(i)


def reduceFile(input_file, output_file, cut):

    ROOT.ROOT.EnableImplicitMT(2)

    df = ROOT.ROOT.RDataFrame("SinglePhoton", input_file)

    df_cut = df.Filter(cut)

    df_all = df_cut.Define('event', 'evt_eventNo') \
                   .Define('mu', 'evt_mu') \
                   .Define('pt', 'y_pt') \
                   .Define('eta', 'y_eta_cl_s2') \
                   .Define('phi', 'y_phi') \
                   .Define('is_tight', 'int(y_IsTight)') \
                   .Define('is_looseprime4', 'int(y_IsLoosePrime4)') \
                   .Define('is_conv', 'int(y_convType != 0)') \
                   .Define('truth_label', 'int(y_isTruthMatchedPhoton)') \
                   .Define('weight', 'mcTotWeightNoPU_PIDuse * intLumi * puWeight') \
                   .Define('is_isoloose', 'int( (y_topoetcone20_IsoCorrTool-y_SCsubtraction-y_topoetcone20ptLogCorrection) < 0.065*y_pt && y_ptcone20_TightTTVA_pt1000/y_pt < 0.05)') \
                   .Define('is_isotight', 'int( (y_topoetcone40_IsoCorrTool-y_SCsubtraction-y_topoetcone40ptLogCorrection) < (0.022*y_pt+2.45) && y_ptcone20_TightTTVA_pt1000/y_pt < 0.05)') \
                   .Define('iso_calo', 'y_topoetcone20_IsoCorrTool-y_SCsubtraction-y_topoetcone20ptLogCorrection') \
                   .Define('iso_track', 'y_ptcone20_TightTTVA_pt1000') \
                   .Define('rhad', 'y_Rhad * (fabs(y_eta_cl_s2)>=0.8 && fabs(y_eta_cl_s2)<1.37) + y_Rhad1 * (fabs(y_eta_cl_s2)<0.8 || fabs(y_eta_cl_s2)>1.37)') \
                   .Define('reta', 'y_Reta') \
                   .Define('rphi', 'y_Rphi') \
                   .Define('weta2', 'y_weta2') \
                   .Define('eratio', 'y_Eratio') \
                   .Define('deltae', '0.001*y_deltae') \
                   .Define('weta1', 'y_weta1') \
                   .Define('wtots1', 'y_wtots1') \
                   .Define('fracs1', 'y_fracs1') \
                   .Define('f1', 'y_f1')

    print("Slimmed file will be safed at: " + output_file)
    outdf = df_all.Snapshot("SinglePhoton", output_file, cols)


if __name__ == "__main__":

    for path in files:

        file_name = path.split('/')[-1]

        output_file_name = path.split('/')[-1].replace('.root', '_mini.root')

        print("Processing " + path)

        ptCut                 = "y_pt > 25"
        etaCut                = "(fabs(y_eta_cl_s2) < 1.37 || (fabs(y_eta_cl_s2) > 1.52 && fabs(y_eta_cl_s2) < 2.37))"
        promptCut             = "(y_isTruthMatchedPhoton == 1)"
        fakeCut               = "(y_isTruthMatchedPhoton == 0)"
        triggerCut       = "HLT_g10_loose == 1 || HLT_g15_loose_L1EM7 == 1 || HLT_g20_loose_L1EM12 == 1 || HLT_g25_loose_L1EM15 == 1 || HLT_g35_loose_L1EM15 == 1 || HLT_g40_loose_L1EM15 == 1 || HLT_g45_loose_L1EM15 == 1 || HLT_g50_loose_L1EM15 == 1 || HLT_g60_loose == 1 || HLT_g70_loose == 1 || HLT_g80_loose == 1 || HLT_g100_loose == 1 || HLT_g120_loose == 1 || HLT_g140_loose== 1"

        looseCut = "y_IsLoose == 1"

        duplicate_cut = "evt_isDuplicate == 0" # removes duplicated events

        if "data" in file_name:
            reduceFile(ntuples_dir+path, output_file_name, isolationCut + " && " + etaCut + " && " + triggerCut + " && " + looseCut)
        elif "jetjet_" in file_name:
            reduceFile(ntuples_dir+path, output_file_name, ptCut + " && " + etaCut + " && " + looseCut + " && " + fakeCut + " && " + duplicate_cut)
        elif "PyPt17_inf_" in file_name:
            reduceFile(ntuples_dir+path, output_file_name,  ptCut + " && " + etaCut + " && " + looseCut + " && " + duplicate_cut)
