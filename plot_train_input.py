import sys
import pickle
import ROOT
import numpy as np
import pandas
from array import array

ROOT.gROOT.SetBatch(True)

df_path = sys.argv[1]
out_dir = sys.argv[2]


def plot_s_b(name, df_t, df_v, col, weightname, bins, xtitle='', text=''):

    at_s = df_t.loc[df_t['truth_label'] == 1, col]
    at_b = df_t.loc[df_t['truth_label'] == 0, col]

    av_s = df_v.loc[df_v['truth_label'] == 1, col]
    av_b = df_v.loc[df_v['truth_label'] == 0, col]

    ht_s = ROOT.TH1F('ht_s', '', len(bins)-1, array('f', bins))
    ht_b = ROOT.TH1F('ht_b', '', len(bins)-1, array('f', bins))

    hv_s = ROOT.TH1F('hv_s', '', len(bins)-1, array('f', bins))
    hv_b = ROOT.TH1F('hv_b', '', len(bins)-1, array('f', bins))

    if weightname is None:
        for v in at_s:
            ht_s.Fill(v)
        for v in at_b:
            ht_b.Fill(v)
        for v in av_s:
            hv_s.Fill(v)
        for v in av_b:
            hv_b.Fill(v)
    else:
        wt_s = df_t.loc[df_t['truth_label'] == 1, weightname]
        wt_b = df_t.loc[df_t['truth_label'] == 0, weightname]

        wv_s = df_v.loc[df_v['truth_label'] == 1, weightname]
        wv_b = df_v.loc[df_v['truth_label'] == 0, weightname]

        for v, w in zip(at_s, wt_s):
            ht_s.Fill(v, w)
        for v, w in zip(at_b, wt_b):
            ht_b.Fill(v, w)
        for v, w in zip(av_s, wv_s):
            hv_s.Fill(v, w)
        for v, w in zip(av_b, wv_b):
            hv_b.Fill(v, w)

    # include under and overflowbins in first/last bin
    def adduo(h):
        nbins = h.GetNbinsX()

        new_val = h.GetBinContent(nbins) + h.GetBinContent(nbins+1)
        h.SetBinContent(nbins, new_val)
        h.SetBinContent(nbins+1, 0.0)

        new_val = h.GetBinContent(1) + h.GetBinContent(0)
        h.SetBinContent(1, new_val)
        h.SetBinContent(0, 0.0)

    if 'scaled' not in name:
        adduo(ht_s)
        adduo(ht_b)
        adduo(hv_s)
        adduo(hv_b)


    ht_all = ht_s.Clone()
    ht_all.Add(ht_b)

    hv_all = hv_s.Clone()
    hv_all.Add(hv_b)

    mean = ht_all.GetMean()
    std  = hv_all.GetStdDev()

    print(f'mean = {mean}, std = {std}')

    c1 = ROOT.TCanvas('', '', 600, 600)
    if 'pt' in var:
        c1.SetLogy()

    c1.SetRightMargin(0.02)
    c1.SetTopMargin(0.02)
    c1.SetTicks()

    ht_all.SetStats(0)
    ht_all.SetTitle('')
    ht_all.GetXaxis().SetTitle(xtitle)

    ht_s.SetMarkerStyle(20)
    ht_s.SetMarkerColor(ROOT.kRed-7)
    ht_s.SetLineColor(ROOT.kRed-7)
    ht_s.SetFillColorAlpha(ROOT.kRed-7, 0.4)
    ht_s.SetLineWidth(2)

    hv_s.SetLineColor(ROOT.kRed-7)
    hv_s.SetLineWidth(2)
    hv_s.SetLineStyle(2)

    ht_b.SetMarkerStyle(20)
    ht_b.SetMarkerColor(ROOT.kAzure)
    ht_b.SetLineColor(ROOT.kAzure)
    ht_b.SetFillColorAlpha(ROOT.kAzure, 0.4)
    ht_b.SetLineWidth(2)

    hv_b.SetLineColor(ROOT.kAzure)
    hv_b.SetLineWidth(2)
    hv_b.SetLineStyle(2)

    ht_all.SetLineColor(ROOT.kGray+1)
    ht_all.SetFillColorAlpha(ROOT.kGray+1, 0.4)
    ht_all.SetLineWidth(2)

    hv_all.SetLineColor(ROOT.kGray+1)
    hv_all.SetLineWidth(2)
    hv_all.SetLineStyle(2)

    ymin = ht_all.GetMinimum() ##min(ht_s.GetMinimum(), ht_b.GetMinimum(), hv_s.GetMinimum(), hv_b.GetMinimum())
    ymax = 1.2*ht_all.GetMaximum() #max(ht_s.GetMaximum(), ht_b.GetMaximum(), hv_s.GetMaximum(), hv_b.GetMaximum())

    if ymin < 1 and 'pt' in xtitle:
        ymin = 0.1
    ht_all.GetYaxis().SetRangeUser(ymin, ymax)

    ht_all.Draw('hist same')
    ht_b.Draw('hist same')
    ht_s.Draw('hist same')

    hv_all.Draw('hist same')
    hv_b.Draw('hist same')
    hv_s.Draw('hist same')


    leg = ROOT.TLegend(0.7, 0.75, 0.80, 0.85)
    leg.SetBorderSize(0)
    leg.AddEntry(ht_s, 'Signal')
    leg.AddEntry(ht_b, 'Fakes')
    leg.Draw()

    if text:
        l = ROOT.TLatex()
        l.SetNDC()
        l.SetTextSize(l.GetTextSize()*0.5)
        l.SetTextFont(42)
        l.DrawLatex(0.70, 0.88, text)

        l.DrawLatex(0.70, 0.70, f'Mean = {mean:.2f}')
        l.DrawLatex(0.70, 0.65, f'Std  = {std:.2f}')

    c1.SaveAs(name)




# Plot some kin variables: pt/eta/phi
dft_raw = pandas.read_hdf(f'{df_path}/df_train_raw.h5', 'df')
dfv_raw = pandas.read_hdf(f'{df_path}/df_val_raw.h5', 'df')

dft = pandas.read_hdf(f'{df_path}/df_train.h5', 'df')
dfv = pandas.read_hdf(f'{df_path}/df_val.h5', 'df')


input_vars = (
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
)

ranges = (
    (0, 1000),
    (-2.4, 2.4),
    (-0.03, 0.06),
    (0.80, 1.10),
    (0.50, 1.10),
    (0.006, 0.018),
    (0.00, 1),
    (0.00, 4),
    (0.40, 1.00),
    (0.00, 10),
    (0.00, 1.10),
)

for iv, var in enumerate(input_vars):

    xmin = ranges[iv][0]
    xmax = ranges[iv][1]

    bins = [ xmin+i*(xmax-xmin)/100 for i in range(100) ]

    plot_s_b(f'{out_dir}/c_train_val_{var}.pdf',    dft_raw, dfv_raw, var,     None, bins, xtitle=var, text='No weight')
    plot_s_b(f'{out_dir}/c_train_val_{var}_w.pdf',  dft_raw, dfv_raw, var, 'weight', bins, xtitle=var, text='MC weight')
    plot_s_b(f'{out_dir}/c_train_val_{var}_rw.pdf', dft_raw, dfv_raw, var,     'rw', bins, xtitle=var, text='RW')

    if var in ('pt', 'eta'):
        xmin, xmax = 0, 1.5
    else:
        xmin, xmax = -5, 5

    bins = [ xmin+i*(xmax-xmin)/100 for i in range(100) ]

    plot_s_b(f'{out_dir}/c_train_val_scaled_{var}.pdf',    dft, dfv, var, None,     bins, xtitle=f'{var} (scaled)', text='No weight')
    plot_s_b(f'{out_dir}/c_train_val_scaled_{var}_w.pdf',  dft, dfv, var, 'weight', bins, xtitle=f'{var} (scaled)', text='MC weight')
    plot_s_b(f'{out_dir}/c_train_val_scaled_{var}_rw.pdf', dft, dfv, var, 'rw',     bins, xtitle=f'{var} (scaled)', text='RW')


