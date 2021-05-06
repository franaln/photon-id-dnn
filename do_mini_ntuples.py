import ROOT
import sys
import os

from datetime import datetime

##ntuples_dir = '/mnt/BIG/PhotonID/SP_ntuples/'
ntuples_dir = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/fullRun2/FinalNtuples/'

files = [
    'PyPt17_inf_mc16a_p3931_Rel21_AB21.2.94_v0.root',
    'PyPt17_inf_mc16d_p3931_Rel21_AB21.2.94_v0.root',
    'PyPt17_inf_mc16e_p3931_Rel21_AB21.2.94_v0.root',

    'Py8_jetjet_mc16a_p3929_Rel21_AB21.2.94_v0.root',
    'Py8_jetjet_mc16d_p3929_Rel21_AB21.2.94_v0.root',
    'Py8_jetjet_mc16e_p3929_Rel21_AB21.2.94_v0.root',

    'data15_276262_284484_GRL_p3930_Rel21_AB21.2.94_v0.root',
    'data16_297730_311481_GRL_p3930_Rel21_AB21.2.94_v0.root',
    'data17_325713_340453_GRL_p3930_Rel21_AB21.2.94_v0.root',
    'data18_348885_364292_GRL_p3930_Rel21_AB21.2.94_v0.root',
]


var_dict = {
    # general
    'mcid': 'mcid',
    'event': 'evt_eventNo', 
    'weight': '',
    'mu': 'evt_mu',
    
    'pt': 'y_pt',
    'eta': 'y_eta_cl_s2',
    'phi': 'y_phi',

    'pt_cl': 'y_pt_cl',

    'is_conv': 'int(y_convType != 0)',
    'truth_label': 'int(y_isTruthMatchedPhoton)',

    # shower shapes
    'rhad': 'y_Rhad',
    'rhad1': 'y_Rhad1',
    'rhad_mixed': 'y_Rhad * (fabs(y_eta_cl_s2)>=0.8 && fabs(y_eta_cl_s2)<1.37) + y_Rhad1 * (fabs(y_eta_cl_s2)<0.8 || fabs(y_eta_cl_s2)>1.37)',
    'reta': 'y_Reta',
    'rphi': 'y_Rphi',
    'weta2': 'y_weta2',
    'eratio': 'y_Eratio',
    'deltae': '0.001*y_deltae',
    'weta1': 'y_weta1',
    'wtots1': 'y_wtots1',
    'fracs1': 'y_fracs1',
    'f1': 'y_f1',
    'e277': 'y_e277',
    'emaxs1': 'y_emaxs1',

    # ID/Iso
    'is_loose': 'int(y_IsLoose)',
    'is_tight': 'int(y_IsTight)',
    'is_looseprime4': 'int(y_IsLoosePrime4)',
  
    'is_isoloose': 'int( (y_topoetcone20_IsoCorrTool-y_SCsubtraction-y_topoetcone20ptLogCorrection) < 0.065*y_pt && y_ptcone20_TightTTVA_pt1000/y_pt < 0.05 )',
    'is_isotight': 'int( (y_topoetcone40_IsoCorrTool-y_SCsubtraction-y_topoetcone40ptLogCorrection) < (0.022*y_pt+2.45) && y_ptcone20_TightTTVA_pt1000/y_pt < 0.05 )',
    'is_isotightcaloonly': 'int( (y_topoetcone40_IsoCorrTool-y_SCsubtraction-y_topoetcone40ptLogCorrection) < (0.022*y_pt+2.45) )',
    'is_isoloose_old': 'int(y_iso_FixedCutLoose)',
    'is_isotight_old': 'int(y_iso_FixedCutTight)',
    'is_isotightcaloonly_old': 'int(y_iso_FixedCutTightCaloOnly)',
    'iso_calo20': 'y_topoetcone20_IsoCorrTool-y_SCsubtraction-y_topoetcone20ptLogCorrection',
    'iso_calo40': 'y_topoetcone40_IsoCorrTool-y_SCsubtraction-y_topoetcone40ptLogCorrection',
    'iso_track': 'y_ptcone20_TightTTVA_pt1000',
    'region_MM': 'get_MM_region(y_IsTight, y_IsEMTight)',

    # extra
    'conv_type': 'y_convType',
    'conv_radius': 'y_convRadius',

    'pt_cl': 'y_pt_cl',
    'E1E2': 'y_E1E2',
    'maxEcell_E': 'y_maxEcell_E',
    'maxEcell_time': 'y_maxEcell_time',
}


cols_data = ROOT.std.vector('string')()
cols_mc   = ROOT.std.vector('string')()

for i in var_dict.keys():
    cols_mc.push_back(i)

    if i != 'truth_label':
        cols_data.push_back(i)


cxx_code = """
bool slice_filter(float fmcid, float tpt)
{
    int mcid = int(fmcid);

    if (mcid==423099) {
        if (tpt < 8 || tpt > 17) return false;
    }
    else if(mcid==423100) {
        if (tpt < 17 || tpt > 35) return false;
    } 
    else if(mcid==423101) {
        if (tpt < 35 || tpt > 50) return false;
    }
    else if(mcid==423102) {
        if (tpt < 50 || tpt > 70) return false;
    } 
    else if(mcid==423103) {
        if (tpt < 70 || tpt > 140) return false;
    }
    else if(mcid==423104) {
        if (tpt < 140 || tpt > 280) return false;
    }
    else if(mcid==423105) {
        if (tpt < 280 || tpt > 500) return false;
    }
    else if(mcid==423106) {
        if (tpt < 500 || tpt > 800) return false;
    }
    else if(mcid==423107) {
        if (tpt < 800 || tpt > 1000) return false;
    }
    else if(mcid==423108) {
        if (tpt < 1000 || tpt > 1500) return false;
    }
    else if(mcid==423109) {
        if (tpt < 1500 || tpt > 2000) return false;
    }
    else if(mcid==423110) {
        if (tpt < 2000 || tpt > 2500) return false;
    } 
    else if(mcid==423111) {
        if (tpt < 2500 || tpt > 3000) return false;
    }
    else if(mcid==423112) {
        if (tpt < 3000) return false;
    }

    return true;
}

int get_MM_region(Bool_t is_tight, UInt_t isem)
{
    if (is_tight) 
        return 1;
    else if ( !((isem)&(1<<17)) && !((isem)&(1<<19)) && !((isem)&(1<<20)) && !((isem)&(1<<21)) ) 
        return 2;
    else if ( !((isem)&(1<<10)) && !((isem)&(1<<12)) && !((isem)&(1<<13)) && !((isem)&(1<<14)) && !((isem)&(1<<18)) ) 
        return 3;
    else
        return 4;
}
"""



ROOT.gInterpreter.Declare(cxx_code)

def reduceFile(input_file, output_file, cut, data_type):

    ROOT.ROOT.EnableImplicitMT(2)

    df = ROOT.ROOT.RDataFrame("SinglePhoton", input_file)

    df = df.Filter(cut)

    for name, var_def in var_dict.items():
        if data_type == 'data' and name == 'truth_label':
            continue

        if name == 'weight':
            if data_type == 'data':
                df = df.Define(name, 'dataWeightPtBin')
            else:
                df = df.Define(name, '(mcWeight * xsecWeight * puWeight * vtxWeight * intLumi) / sumWeights')

        elif var_def != name:
            df = df.Define(name, var_def)

    print("Slimmed file will be safed at: " + output_file)
    cols = cols_data if data_type == 'data' else cols_mc
    outdf = df.Snapshot("SinglePhoton", output_file, cols)


if __name__ == "__main__":

    for path in files:

        file_name = path.split('/')[-1]

        output_file_name = path.split('/')[-1].replace('.root', '_mini.root')

        print("Processing " + path)

        ptCut         = "y_pt > 25"
        etaCut        = "(fabs(y_eta_cl_s2) < 1.37 || (fabs(y_eta_cl_s2) > 1.52 && fabs(y_eta_cl_s2) < 2.37))"
        promptCut     = "(y_isTruthMatchedPhoton == 1)"
        fakeCut       = "(y_isTruthMatchedPhoton == 0)"
        triggerCut    = "(HLT_g10_loose == 1 || HLT_g15_loose_L1EM7 == 1 || HLT_g20_loose_L1EM12 == 1 || HLT_g25_loose_L1EM15 == 1 || HLT_g35_loose_L1EM15 == 1 || HLT_g40_loose_L1EM15 == 1 || HLT_g45_loose_L1EM15 == 1 || HLT_g50_loose_L1EM15 == 1 || HLT_g60_loose == 1 || HLT_g70_loose == 1 || HLT_g80_loose == 1 || HLT_g100_loose == 1 || HLT_g120_loose == 1 || HLT_g140_loose== 1)"
        looseCut      = "y_IsLoose == 1"
        duplicate_cut = "evt_isDuplicate == 0" # removes duplicated events
        common_cut    = 'y_e277>0.1 && y_f1>0.005'

        if 'data' in file_name:
            reduceFile(ntuples_dir+path, output_file_name, f'{ptCut} && {etaCut} && {looseCut} && {triggerCut} && {common_cut}', 'data')
        elif "jetjet_" in file_name:
            reduceFile(ntuples_dir+path, output_file_name, f'{ptCut} && {etaCut} && {looseCut} && {triggerCut} && {fakeCut} && {duplicate_cut} && {common_cut}', 'fakes')
        elif "PyPt17_inf_" in file_name:
            reduceFile(ntuples_dir+path, output_file_name,  f'{ptCut} && {etaCut} && {looseCut} && {triggerCut} && {promptCut} && {duplicate_cut} && {common_cut} && slice_filter(mcid, y_truth_pt)', 'signal')
