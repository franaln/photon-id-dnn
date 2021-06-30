import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plot


input_file = sys.argv[1] #'output_apr25/df_test_output_apr25.h5'
output_dir = sys.argv[2] #'output_apr25'

models = [
    'nn_baseline',
    'nn_disco50',
    'nn_disco25',
    'nn_disco10_calo',
    'nn_disco25_calo',
]

colors = [
    'tab:blue',
    'tab:red',
    'tab:purple',
    'tab:orange',
    'tab:green',
]

labels = [
    'NN',
    r'NN + DisCo ($\alpha = 50$)',
    r'NN + DisCo ($\alpha = 25$)',
    r'NN + DisCo ($\alpha = 10$) calo',
    r'NN + DisCo ($\alpha = 25$) calo',
]

# Plot functions
def plot_roc(name, curves, aucs, tight=(), text=''):

    fig, ax = plt.subplots()

    ax.plot(tight[0], tight[1], 'x', color='black', label=f'Tight') ## (eff={tight[0]:.2f}, rej={tight[1]:.2f})')

    for im, m in enumerate(models):
        ax.plot(curves[m][0], curves[m][1], '-', color=colors[im], label=labels[im])

    ax.set_xlabel('Signal Efficiency', loc='right')
    ax.set_ylabel('1 - Fakes Efficiency', loc='top')

    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.25, 1.0)

    ax.legend(loc='lower left')

    ax.axvline(x=tight[0], color="black", ls='--', lw=0.5)
    ax.axhline(y=tight[1], color="black", ls='--', lw=0.5)
    ax.plot(tight[0], tight[1], 'x', color='black')

    ax.text(0.46, 0.93, text)

    fig.savefig(name, bbox_inches='tight')
    plt.close('all')


def plot_eff(t, name, bins, curves, curves_unc, xlabel, text=''):

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel, loc='right')
    if t == 'eff':
        ax.set_ylabel('Signal Efficiency', loc='top')
    else:
        ax.set_ylabel('1 - Fakes Efficiency', loc='top')

    x = np.array([ 0.5*(bins[b]+bins[b+1]) for b in range(len(bins)-1) ])
    if x[-1] >  500.:
        x[-1] = 100.

    ax.errorbar(x, curves[0], fmt='.-', color='black', label='Tight', yerr=curves_unc[0])

    for i in range(len(curves)-1):
        ax.errorbar(x, curves[i+1], fmt='.-', color=colors[i], label=labels[i], yerr=curves_unc[i+1])

    if t == 'eff':
        ax.set_ylim(0.5, 1)
    else:
        ax.set_ylim(0.0, 1.0)

    ax.legend()

    ax.text(0.1, 0.9, text, transform=ax.transAxes)

    fig.savefig(name, bbox_inches='tight')
    plt.close('all')


def plot_2d(data, title, name, zmin, zmax, annot=False):

    data[:,2] = np.nan

    fig, ax = plt.subplots()
    im = ax.imshow(data, vmin=zmin, vmax=zmax, cmap='coolwarm', origin='lower')

    ax.set_ylabel(r"$p_\mathrm{T}$ [GeV]", loc='top')
    ax.set_xlabel(r"$|\eta|$", loc='right')
    ax.set_title(title)

    ax.set_yticks([ i-0.5 for i in range(len(pt_bins)-1) ])
    ax.set_yticklabels([ '%.0f' % f for f in pt_bins[:-1]])
    ax.set_xticks([ i-0.5 for i in range(len(eta_bins)) ])
    ax.set_xticklabels([ '%.2f' % f for f in eta_bins])

    # Loop over data dimensions and create text annotations.
    if annot:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if j == 2: continue
                text = ax.text(j, i, f'{data[i, j]:.2f}',
                               ha="center", va="center", color="black")


    cax = fig.add_axes([ax.get_position().x1+0.02,ax.get_position().y0, 0.03, ax.get_position().height])
    fig.colorbar(im, cax=cax)

    fig.savefig(name, bbox_inches='tight')
    plt.close("all")




#

df = pd.read_hdf(input_file, 'df')

weight = 'weight'

df_s = df.loc[(df['truth_label'] == 1) & (df['is_isoloose']==1)]
df_b = df.loc[(df['truth_label'] == 0) & (df['is_isoloose']==1)]

s_total = df_s[weight].sum()
b_total = df_b[weight].sum()

s_pass_tight = df_s.loc[df_s['is_tight'] == 1, weight].sum()
b_pass_tight = df_b.loc[df_b['is_tight'] == 1, weight].sum()

eff_s_tight = s_pass_tight / s_total
rej_b_tight = 1 - (b_pass_tight / b_total)


nth = 200
step = 1 / nth
thresholds = [ step*i for i in range(nth+1) ]

# Inclusive
curves = {}
aucs = {}
for model in models:

    ax, ay = [], []
    auc = 0
    for ic, cut in enumerate(thresholds):

        s_pass = df_s.loc[df_s[model] > cut, weight].sum()
        b_pass = df_b.loc[df_b[model] > cut, weight].sum()

        eff_s = s_pass / s_total
        rej_b = 1 - (b_pass / b_total)

        ax.append(eff_s)
        ay.append(rej_b)

        if ic > 0:
            auc += abs(ax[ic]-ax[ic-1]) * (ay[ic] + ay[ic-1])/2

    curves[model] = (ax, ay)
    aucs[model] = auc



plot_roc(
    f'{output_dir}/c_roc.png',
    curves, aucs,
    (eff_s_tight, rej_b_tight),
    'Loose ID, FixedCutLoose isolation\nInclusive $\eta$, Inclusive $p_{\mathrm{T}}$, Converted/Unconverted'
     )

for conv_bin in (0, 1):

    df_bin_s = df_s.loc[df_s['is_conv'] == conv_bin]
    df_bin_b = df_b.loc[df_b['is_conv'] == conv_bin]

    s_total = df_bin_s[weight].sum()
    b_total = df_bin_b[weight].sum()

    s_pass_tight = df_bin_s.loc[(df_bin_s['is_tight'] == 1),weight].sum()
    b_pass_tight = df_bin_b.loc[(df_bin_b['is_tight'] == 1),weight].sum()

    if s_total < 1 or b_total < 1:
        print('No events in this bin ...')
        continue

    eff_s_tight = s_pass_tight / s_total
    rej_b_tight = 1 - (b_pass_tight / b_total)

    curves = {}
    aucs = {}
    for model in models:

        ax, ay = [], []
        auc = 0
        for ic, cut in enumerate(thresholds):

            s_pass = df_bin_s.loc[df_bin_s[model] > cut, weight].sum()
            b_pass = df_bin_b.loc[df_bin_b[model] > cut, weight].sum()

            eff_s = s_pass / s_total
            rej_b = 1 - (b_pass / b_total)

            ax.append(eff_s)
            ay.append(rej_b)

            if ic > 0:
                auc += abs(ax[ic]-ax[ic-1]) * (ay[ic] + ay[ic-1])/2

        curves[model] = (ax, ay)
        aucs[model] = auc


    if conv_bin == 1:
        cc = 'c'
        text = 'Loose ID, FixedCutLoose isolation\nInclusive $\eta$, Inclusive $p_{\mathrm{T}}$, Converted'
    else:
        cc = 'u'
        text = 'Loose ID, FixedCutLoose isolation\nInclusive $\eta$, Inclusive $p_{\mathrm{T}}$, Unconverted'

    plot_roc(
        f'{output_dir}/c_roc_{cc}.png',
        curves, aucs,
        (eff_s_tight, rej_b_tight),
        text
    )




# In bins of pt/eta/conv
# pt_bins = [25, 30, 35, 40, 50, 60, 80, 100, 10000]
# eta_bins = [0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37]
pt_bins = [25, 35, 50, 80, 10000]
eta_bins = [0, 0.6, 1.37, 1.52, 1.81, 2.37]
conv_bins = [0, 1]

n_pt_bins = len(pt_bins)-1
n_eta_bins = len(eta_bins)-1


df_s['pt_bin']  = pd.cut(df_s['pt'], bins=pt_bins, labels=False)
df_b['pt_bin']  = pd.cut(df_b['pt'], bins=pt_bins, labels=False)

df_s['eta_bin']  = pd.cut(np.abs(df_s['eta']), bins=eta_bins, labels=False)
df_b['eta_bin']  = pd.cut(np.abs(df_b['eta']), bins=eta_bins, labels=False)

df_s = df_s.astype({'pt_bin': 'int32', 'eta_bin': 'int32'})
df_b = df_b.astype({'pt_bin': 'int32', 'eta_bin': 'int32'})


effs_Tight = np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float)
rejb_Tight = np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float)

cuts_TightNNE_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
cuts_TightNNR_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }

effs_TightNNE_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
effs_TightNNR_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
rejb_TightNNE_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
rejb_TightNNR_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }

cuts_TightNNO_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
effs_TightNNO_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }
rejb_TightNNO_dict = { m: np.zeros((n_pt_bins, n_eta_bins, len(conv_bins)), dtype=float) for m in models }


for conv_bin in conv_bins:
    for eta_bin in range(len(eta_bins)-1):

        if eta_bin == 2:
            continue

        for pt_bin in range(len(pt_bins)-1):

            print(f'-- Bin: conv_bin={conv_bin}, pt_bin={pt_bin}, eta_bin={eta_bin}')
            
            df_bin_s = df_s.loc[(df_s['pt_bin'] == pt_bin) & (df_s['eta_bin'] == eta_bin) & (df_s['is_conv'] == conv_bin)]
            df_bin_b = df_b.loc[(df_b['pt_bin'] == pt_bin) & (df_b['eta_bin'] == eta_bin) & (df_b['is_conv'] == conv_bin)]

            s_total = df_bin_s[weight].sum()
            b_total = df_bin_b[weight].sum()

            s_pass_tight = df_bin_s.loc[(df_bin_s['is_tight'] == 1),weight].sum()
            b_pass_tight = df_bin_b.loc[(df_bin_b['is_tight'] == 1),weight].sum()

            if s_total < 1 or b_total < 1:
                print('No events in this bin ...')
                continue

            eff_s_tight = s_pass_tight / s_total
            rej_b_tight = 1 - (b_pass_tight / b_total)

            print(f'Tight    : eff_s = {eff_s_tight:.4f}, rej_b = {rej_b_tight:.4f}')

            effs_Tight[pt_bin, eta_bin, conv_bin] = eff_s_tight
            rejb_Tight[pt_bin, eta_bin, conv_bin] = rej_b_tight


            curves = {}
            aucs = {}
            for model in models:

                print (model)

                ax, ay = [], []
                auc = 0

                min_diff_eff, min_diff_rej = 999, 999
                wpe_cut, wpe_eff_s, wpe_rej_b = 0, 0, 0
                wpr_cut, wpr_eff_s, wpr_rej_b = 0, 0, 0

                min_dist = 999
                wpo_cut, wpo_eff_s, wpo_rej_b = 0, 0, 0

                for ic, cut in enumerate(thresholds):

                    s_pass = df_bin_s.loc[df_bin_s[model] > cut,weight].sum()
                    b_pass = df_bin_b.loc[df_bin_b[model] > cut,weight].sum()

                    eff_s = s_pass / s_total
                    rej_b = 1 - (b_pass / b_total)

                    ax.append(eff_s)
                    ay.append(rej_b)

                    if ic > 0:
                        auc += abs(ax[ic]-ax[ic-1]) * (ay[ic] + ay[ic-1])/2

                    diff = abs(eff_s - eff_s_tight)
                    if diff < min_diff_eff:
                        min_diff_eff = diff
                        wpe_cut, wpe_eff_s, wpe_rej_b = cut, eff_s, rej_b

                    diff = abs(rej_b - rej_b_tight)
                    if diff < min_diff_rej:
                        min_diff_rej = diff
                        wpr_cut, wpr_eff_s, wpr_rej_b = cut, eff_s, rej_b

                    d = (1-eff_s)*(1-eff_s) + (1-rej_b)*(1-rej_b)
                    if d < min_dist:
                        min_dist = d
                        wpo_cut, wpo_eff_s, wpo_rej_b = cut, eff_s, rej_b


                print(f'TightNNE : eff_s = {wpe_eff_s:.4f}, rej_b = {wpe_rej_b:.4f}, cut = {wpe_cut:.4f}')
                print(f'TightNNR : eff_s = {wpr_eff_s:.4f}, rej_b = {wpr_rej_b:.4f}, cut = {wpr_cut:.4f}')
                print(f'TightNNO : eff_s = {wpo_eff_s:.4f}, rej_b = {wpo_rej_b:.4f}, cut = {wpo_cut:.4f}')

                cuts_TightNNE_dict[model][pt_bin, eta_bin, conv_bin] = round(wpe_cut, 4)
                effs_TightNNE_dict[model][pt_bin, eta_bin, conv_bin] = wpe_eff_s
                rejb_TightNNE_dict[model][pt_bin, eta_bin, conv_bin] = wpe_rej_b

                cuts_TightNNR_dict[model][pt_bin, eta_bin, conv_bin] = round(wpr_cut, 4)
                effs_TightNNR_dict[model][pt_bin, eta_bin, conv_bin] = wpr_eff_s
                rejb_TightNNR_dict[model][pt_bin, eta_bin, conv_bin] = wpr_rej_b

                cuts_TightNNO_dict[model][pt_bin, eta_bin, conv_bin] = round(wpo_cut, 4)
                effs_TightNNO_dict[model][pt_bin, eta_bin, conv_bin] = wpo_eff_s
                rejb_TightNNO_dict[model][pt_bin, eta_bin, conv_bin] = wpo_rej_b

                curves[model] = (ax, ay)
                aucs[model] = auc


            bin_str = 'conv_%i_eta_%i_pt_%i' % (conv_bin, eta_bin, pt_bin)

            conv_str = 'Converted' if conv_bin == 1 else 'Unconverted'
            eta_str  = '%.2f < $|\eta|$ < %.2f' % (eta_bins[eta_bin], eta_bins[eta_bin+1]) ##'Barrel' if eta_bin == 'b' else 'End-cap'
            if pt_bin == len(pt_bins)-1:
                pt_str   = '$p_{\mathrm{T}}$ > %i GeV' % (pt_bins[pt_bin])
            else:
                pt_str   = '%i < $p_{\mathrm{T}}$ < %i GeV' % (pt_bins[pt_bin], pt_bins[pt_bin+1])

            texts = 'Loose ID, FixedCutLoose isolation'
            texts += '\n%s, %s, %s' % (conv_str, eta_str, pt_str)

            plot_roc(f'{output_dir}/c_roc_{bin_str}.png', 
                     curves, aucs,
                     (eff_s_tight, rej_b_tight),
                     text=r'%s' % texts)




# Plot efficiencies
mu_bins  = [0, 10, 20, 30, 40, 50, 60, 90]

def is_tight_bin(df, nn_name, cuts_array):
    a_conv_bin = df['is_conv']
    a_pt_bin = np.searchsorted(pt_bins, df['pt']) - 1
    a_eta_bin = np.searchsorted(eta_bins, np.abs(df['eta'])) - 1

    cond = np.array([ cuts_array[pt_bin,eta_bin,conv_bin] for conv_bin, eta_bin, pt_bin in zip(a_conv_bin, a_eta_bin, a_pt_bin) ])

    pass_tight = df[nn_name] > cond

    return pass_tight


for m in models:
    df_s[f'is_TightNNE_{m}']  = is_tight_bin(df_s, m, cuts_TightNNE_dict[m])
    df_s[f'is_TightNNR_{m}']  = is_tight_bin(df_s, m, cuts_TightNNR_dict[m])
    df_s[f'is_TightNNO_{m}']  = is_tight_bin(df_s, m, cuts_TightNNO_dict[m])

    df_b[f'is_TightNNE_{m}']  = is_tight_bin(df_b, m, cuts_TightNNE_dict[m])
    df_b[f'is_TightNNR_{m}']  = is_tight_bin(df_b, m, cuts_TightNNR_dict[m])
    df_b[f'is_TightNNO_{m}']  = is_tight_bin(df_b, m, cuts_TightNNO_dict[m])



for c in (0, 1):

    c_sel_s = df_s['is_conv'] == c
    c_sel_b = df_b['is_conv'] == c

    for var in ('pt', 'eta', 'mu'):

        if var == 'pt':
            bins = pt_bins
        elif var == 'eta':
            bins = eta_bins
        elif var == 'mu':
            bins = mu_bins

        h_s_den = np.histogram(df_s.loc[c_sel_s,var], bins=bins, weights=df_s.loc[c_sel_s,weight])[0]
        h_b_den = np.histogram(df_b.loc[c_sel_b,var], bins=bins, weights=df_b.loc[c_sel_b,weight])[0]


        for wp in ('TightNNE', 'TightNNR', 'TightNNO'):

            curves_eff, curves_eff_unc = [], []

            for sel in ['is_tight',] + [ f'is_{wp}_{m}' for m in models ]:
                h_s_num = np.histogram(df_s.loc[(c_sel_s) & (df_s[sel]), var], bins=bins, weights=df_s.loc[(c_sel_s) & (df_s[sel]), weight])[0]

                h_s_eff = h_s_num / h_s_den
                h_s_unc = np.sqrt(h_s_eff * ( np.ones_like(h_s_eff) - h_s_eff) / h_s_den)

                curves_eff.append(h_s_eff)
                curves_eff_unc.append(h_s_unc)


            curves_rej, curves_rej_unc  = [], []

            for sel in ['is_tight',] + [ f'is_{wp}_{m}' for m in models]:
                h_b_num = np.histogram(df_b.loc[(c_sel_b) & (df_b[sel]), var], bins=bins, weights=df_b.loc[(c_sel_b) & (df_b[sel]), weight])[0]

                h_b_eff = (h_b_num / h_b_den)
                h_b_rej = 1 - h_b_eff
                h_b_unc = np.sqrt(h_s_eff * ( np.ones_like(h_s_eff) - h_s_eff) / h_s_den)

                curves_rej.append(h_b_rej)
                curves_rej_unc.append(h_b_unc)


            # plot
            xtitles = {
                'pt': '$p_{\mathrm{T}}$ [GeV]',
                'eta': '$|\eta|$',
                'mu': '$\mu$',
            }

            cc = 'c' if c == 1 else 'u'

            plot_eff('eff',  f'{output_dir}/effs_{wp}_{var}_{cc}.png', bins,  curves_eff, curves_eff_unc, xtitles.get(var), 'Converted' if c == 1 else 'Unconverted')
            plot_eff('rej',  f'{output_dir}/rejb_{wp}_{var}_{cc}.png', bins,  curves_rej, curves_rej_unc, xtitles.get(var), 'Converted' if c == 1 else 'Unconverted')



            


# Plot improvements
for c in (0,1):
    cstr = 'Converted' if c == 1 else 'Unconverted'
    cc = 'c' if c == 1 else 'u'

    plot_2d(effs_Tight[:,:,c], 
            f'Efficiency for Tight, {cstr}',
            f'{output_dir}/effs_2d_Tight_{cc}.png', zmin=0, zmax=1, annot=True)

    plot_2d(rejb_Tight[:,:,c], 
            f'Rejection for Tight, {cstr}',
            f'{output_dir}/rejb_2d_Tight_{cc}.png', zmin=0, zmax=1, annot=True)

for m in models:

    for c in (0, 1):

        diff_effs_TightNNE = 100 * (effs_TightNNE_dict[m][:,:,c] - effs_Tight[:,:,c])
        diff_effs_TightNNR = 100 * (effs_TightNNR_dict[m][:,:,c] - effs_Tight[:,:,c])
        diff_effs_TightNNO = 100 * (effs_TightNNO_dict[m][:,:,c] - effs_Tight[:,:,c])
        diff_rejb_TightNNE = 100 * (rejb_TightNNE_dict[m][:,:,c] - rejb_Tight[:,:,c])
        diff_rejb_TightNNR = 100 * (rejb_TightNNR_dict[m][:,:,c] - rejb_Tight[:,:,c])
        diff_rejb_TightNNO = 100 * (rejb_TightNNO_dict[m][:,:,c] - rejb_Tight[:,:,c])

        cstr = 'Converted' if c == 1 else 'Unconverted'
        cc = 'c' if c == 1 else 'u'

        plot_2d(effs_TightNNE_dict[m][:,:,c], 
                f'Efficiency for TightNNE, {cstr}, {m}',
                f'{output_dir}/effs_2d_TightNNE_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(rejb_TightNNE_dict[m][:,:,c], 
                f'Rejection for TightNNE, {cstr}, {m}',
                f'{output_dir}/rejb_2d_TightNNE_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(effs_TightNNR_dict[m][:,:,c], 
                f'Efficiency for TightNNR, {cstr}, {m}',
                f'{output_dir}/effs_2d_TightNNR_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(rejb_TightNNR_dict[m][:,:,c], 
                f'Rejection for TightNNR, {cstr}, {m}',
                f'{output_dir}/rejb_2d_TightNNR_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(effs_TightNNO_dict[m][:,:,c], 
                f'Efficiency for TightNNO, {cstr}, {m}',
                f'{output_dir}/effs_2d_TightNNO_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(rejb_TightNNO_dict[m][:,:,c], 
                f'Rejection for TightNNO, {cstr}, {m}',
                f'{output_dir}/rejb_2d_TightNNO_{m}_{cc}.png', zmin=0, zmax=1, annot=True)


        plot_2d(diff_effs_TightNNE, 
                f'Diff in efficiency for TightNNE, {cstr}, {m}',
                f'{output_dir}/diff_effs_2d_TightNNE_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)

        plot_2d(diff_effs_TightNNR, 
                f'Diff in efficiency for TightNNR, {cstr}, {m}',
                f'{output_dir}/diff_effs_2d_TightNNR_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)

        plot_2d(diff_effs_TightNNO, 
                f'Diff in efficiency for TightNNO, {cstr}, {m}',
                f'{output_dir}/diff_effs_2d_TightNNO_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)

        plot_2d(diff_rejb_TightNNE, 
                f'Diff in rejection for TightNNE, {cstr}, {m}',
                f'{output_dir}/diff_rejb_2d_TightNNE_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)

        plot_2d(diff_rejb_TightNNR, 
                f'Diff in rejection for TightNNR, {cstr}, {m}',
                f'{output_dir}/diff_rejb_2d_TightNNR_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)

        plot_2d(diff_rejb_TightNNO, 
                f'Diff in rejection for TightNNO, {cstr}, {m}',
                f'{output_dir}/diff_rejb_2d_TightNNO_{m}_{cc}.png', zmin=-10, zmax=10, annot=True)


        plot_2d(cuts_TightNNR_dict[m][:,:,c],
                f'Cuts for TightNNR, {cstr}, {m}',
                f'{output_dir}/cuts_2d_TightNNR_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(cuts_TightNNE_dict[m][:,:,c],
                f'Cuts for TightNNE, {cstr}, {m}',
                f'{output_dir}/cuts_2d_TightNNE_{m}_{cc}.png', zmin=0, zmax=1, annot=True)

        plot_2d(cuts_TightNNO_dict[m][:,:,c],
                f'Cuts for TightNNO, {cstr}, {m}',
                f'{output_dir}/cuts_2d_TightNNO_{m}_{cc}.png', zmin=0, zmax=1, annot=True)



