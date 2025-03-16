import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import pandas as pd


def main(fnames):
    print(fnames)
    dfs = []
    for fn in fnames:
        df = pd.read_csv(fn)
        print(fn, df.shape)
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    df['L'] = df['path-02'].str.split('-').str[0].astype(int)
    df['W'] = df['path-02'].str.split('-').str[1].astype(int)
    df['mode'] = df['mode'].str.replace('\\', '')
    df = df[df['m_selection'] == 'model_best_acc']
    print(df['L'].unique())
    print(df['W'].unique())
    line_plots = []
    # ALL
    pv = df[(df['path-06'] == 'all') & (df['m_selection'] == 'model_best_acc')].pivot_table(
        index=['W', 'L', 'path-06', 'single-tf'],
        values=['adv_bacc', 'adv_auc', 'adv_aupr'], aggfunc='mean')
    for idx, pvi in pv.iterrows():
        line_plots.append({'W': idx[0], 'L': idx[1], 'name': 'all', 'metric': 'auc', 'val': pvi['adv_auc']})
        line_plots.append({'W': idx[0], 'L': idx[1], 'name': 'all', 'metric': 'aupr', 'val': pvi['adv_aupr']})
        line_plots.append({'W': idx[0], 'L': idx[1], 'name': 'all', 'metric': 'acc', 'val': pvi['adv_bacc']})
    # MD&MO
    get_pv = lambda test_mode, train_mode: df[(df['mode'] == test_mode) & (df['path-06'] == train_mode) & (
            df['m_selection'] == 'model_best_acc')].pivot_table(
        index=['W', 'L', 'path-06', 'single-tf'],
        values=['adv_bacc', 'adv_auc', 'adv_aupr'], aggfunc='mean')
    pv_mdmd = get_pv('motif_discovery', 'motif_discovery')
    pv_momo = get_pv('motif_occupancy', 'motif_occupancy')
    pv_mdmo = get_pv('motif_discovery', 'motif_occupancy')
    pv_momd = get_pv('motif_occupancy', 'motif_discovery')
    # print(pv_mdmd)
    # print(pv_momo)
    best = (pv_mdmd[['adv_bacc', 'adv_auc', 'adv_aupr']].values + pv_momo[
        ['adv_bacc', 'adv_auc', 'adv_aupr']].values) / 2
    worst = (pv_mdmo[['adv_bacc', 'adv_auc', 'adv_aupr']].values + pv_momd[
        ['adv_bacc', 'adv_auc', 'adv_aupr']].values) / 2
    # print(best)
    # print(worst)
    for idx, v_best, v_worst in zip(pv_mdmd.index, best, worst):
        print(idx, v_best, v_worst)
        base_dict = {'W': idx[0], 'L': idx[1], 'name': ('s' if idx[3] == True else 'm') + '-in'}
        line_plots.append({**base_dict, 'metric': 'acc', 'val': v_best[0]})
        line_plots.append({**base_dict, 'metric': 'auc', 'val': v_best[1]})
        line_plots.append({**base_dict, 'metric': 'aupr', 'val': v_best[2]})
        print(line_plots[-1])
        base_dict = {'W': idx[0], 'L': idx[1], 'name': ('s' if idx[3] == True else 'm') + '-out'}
        line_plots.append({**base_dict, 'metric': 'acc', 'val': v_worst[0]})
        line_plots.append({**base_dict, 'metric': 'auc', 'val': v_worst[1]})
        line_plots.append({**base_dict, 'metric': 'aupr', 'val': v_worst[2]})
        print(line_plots[-1])
    plot_df = pd.DataFrame(line_plots)
    plot_df.to_csv('plot_df.csv', index=False)
    fig, axes = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(13, 9))
    handles = None
    for k, w in enumerate(plot_df['W'].unique()):
        for i, m in enumerate(['acc', 'aupr', 'auc']):
            dfi = plot_df[(plot_df['W'] == w) & (plot_df['metric'] == m)]
            # print(w, m,dfi.shape)
            # for n in dfi['name'].unique():
            # https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
            colors = ['#648FFF', '#DC267F', '#DC267F', '#FFB000', '#FFB000']
            line_styles = ['-', '-', '--', '-', '--']
            for n, c, ls in zip(['all', 's-in', 's-out', 'm-in', 'm-out'], colors, line_styles):
                dfin = dfi[dfi['name'] == n]
                print(w, m, n, dfin.shape)
                if dfin.shape[0] == 0:
                    print('Skip')
                    continue
                axes[i, k].plot(dfin['L'].astype(str) + '-' + dfin['W'].astype(str), dfin['val'], 'x', label=n, color=c,
                                linestyle=ls)
            axes[i, k].tick_params(axis='x', labelrotation=30)
            axes[i, k].tick_params(labelleft=True if k == 0 else False, labelright=True if k == 2 else False)
            if k==2:
                axes[i, k].yaxis.set_ticks_position('both')
                # axes[i,k].yaxis.tick_right()
            if i==0:
                if k<=1:
                    axes[i,k].set_title('Training on $\mathcal{D}_{3}$')
                else:
                    axes[i,k].set_title('Training on $\mathcal{D}_{422}$')
            if k == i:
                print(dfi['L'].unique())
                xc = len(dfi['L'].unique()) - 1
                print(xc)
                lmax = dfi['L'].max()
                dfb = float(dfi[(dfi['name'] == 'm-in') & (dfi['L'] == lmax)]['val'])
                dfw = float(dfi[(dfi['name'] == 'm-out') & (dfi['L'] == lmax)]['val'])
                print('dfb', dfb)
                print('dfw', dfw)
                # sys.exit(0)
                bbox_args = None  # dict(boxstyle="round", fc="0.8")
                arrow_args = dict(arrowstyle="->")
                axes[i, k].annotate('Performance drops on OOD', xy=(xc, dfw), xycoords='data',
                                    xytext=(-25, 40), textcoords='offset points',
                                    ha="right", va="bottom",
                                    bbox=bbox_args,
                                    arrowprops=arrow_args)
                # axes[i, k].annotate('Performance drop',
                #                     xy=(lmax, dfb), xycoords='data',
                #                     xytext=(lmax, dfw), textcoords='data',
                #                     arrowprops=dict(facecolor='black', shrink=0.05)
                #                     # , horizontalalignment='right', verticalalignment='top'
                #                     )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, -1].legend(handles, labels)
    # fig.legend(handles, labels, loc='upper right')
    # axes[-1, 0].legend()
    # Set common labels
    fig.text(0.5, 0.05, 'Network size', ha='center', va='center', fontsize=12)
    for i in range(1):
        axes[0, i].set_ylabel('Accuracy')
        axes[0, i].set_ylim([.63, .92])
        axes[1, i].set_ylabel('AUPRC')
        axes[1, i].set_ylim([.71, .96])
        axes[2, i].set_ylabel('AUROC')
        axes[2, i].set_ylim([.69, .96])
        for j in range(3):
            axes[j, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            # axes[j, i].set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    # print(fig.get_dpi())
    # for item in [fig, axes]:
    # fig.patch.set_visible(False)

    # with open('test.png', 'w') as outfile:
    #     fig.canvas.print_png(outfile)
    plt.show()
    sys.exit(0)
    # df.pivot_table(index=['path-02', ])


if __name__ == '__main__':
    parser = ArgumentParser(description='App description')
    # ds params
    parser.add_argument('--fnames', type=str, required=True, nargs='+')

    FLAGS = parser.parse_args()
    plt.rcParams['axes.labelsize'] = 12
    main(**vars(FLAGS))
