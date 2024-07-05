import matplotlib
import mne
import numpy
import os
import random
import re
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

# Using Helvetica as a font
font_folder = 'fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

groups = dict()
results = dict()
rand_results = dict()

with open('bootstrap.txt') as i:
    for l in i:
        line = l.strip().split('\t')
        target = line[0]
        if target not in results.keys():
            results[target] = dict()
            rand_results[target] = dict()
        group = line[1]
        if group not in results[target].keys():
            results[target][group] = dict()
            rand_results[target][group] = dict()
        var = line[2]
        var = re.sub('^l|^L_', 'left ', var)
        var = re.sub('^r|^R_', 'right ', var)
        var = re.sub('_r', '_right ', var)
        var = re.sub('_l', '_left ', var)
        case = line[3]
        #print(var)
        vals = numpy.array(line[4:], dtype=numpy.float32)
        if case == 'sim':
            results[target][group][var] = vals
        elif case == 'rand':
            rand_results[target][group][var] = vals

all_ps = dict()

for target, t_data in results.items():
    for group, g_data in t_data.items():
        current_res = list()
        current_ks = list()
        for var, res in g_data.items():
            current_res.append((var, res))
        ### p-values
        ps = dict()
        for i in range(len(current_res)):
            '''
            ### generic p-value - not really informative
            p = scipy.stats.wilcoxon([v-t for v, t in zip(
                                               current_res[i][1],
                                               rand_results[target][group][var],
                                               )
                                      ],
                                      nan_policy='omit',
                                      ).pvalue
            all_ps[(target, group, current_res[i][0])] = p
            '''
            for j in range(len(current_res)):
                if i <= j:
                    continue
                p = scipy.stats.ttest_ind(
                                          current_res[i][1],
                                          current_res[j][1],
                                          nan_policy='omit',
                                          ).pvalue
                k = tuple(sorted([current_res[i][0], current_res[j][0]]))
                all_ps[(target, group, k)] = p
ord_ps = [(k, v) for k, v in all_ps.items()]
just_corr_ps = mne.stats.fdr_correction([v[1] for v in ord_ps])[1]
corr_ps = {k[0] : v for k, v in zip(ord_ps, just_corr_ps)}
#corr_ps = all_ps
#p_thr = 0.05/len(ord_ps)
p_thr = 0.05

all_colors = {
          'act' : [
                   'lightpink',
                   'forestgreen',
                   'steelblue',
                   'khaki',
                   'paleturquoise',
                   'mediumvioletred',
                   ],
          'conn' : [
                   'paleturquoise',
                   'mediumaquamarine',
                   'forestgreen',
                   'mediumblue',
                   'palegoldenrod',
                   'darkkhaki',
                   'palevioletred',
                   'mediumorchid',
                   'silver',
                   ],
          'les' : [
                   'goldenrod',
                   'mediumseagreen',
                   'palevioletred',
                   'royalblue',
                   ],
          }

out = 'var_by_var'
for target, t_data in results.items():
    for group, g_data in t_data.items():
        for k in ['act', 'les', 'conn']:
            if k in group:
                colors = all_colors[k]
        curr_out = os.path.join(out, target, group,)
        os.makedirs(curr_out, exist_ok=True)
        xs = sorted(g_data.keys())
        ys = [numpy.nanmean(g_data[v]) for v in xs]
        fig, ax = pyplot.subplots(
                                  constrained_layout=True,
                                  figsize=(20, 10),
                                  )
        ax.set_ylim(bottom=-(3+len(ys))*0.013, top=.3)
        ax.hlines(
                  y=[0., 0.05, 0.1, 0.15, 0.2, 0.25],
                  xmin=-.4,
                  xmax=len(xs)-.6,
                  alpha=0.2,
                  linestyles='--',
                  color='gray',
                  )

        for x in range(len(xs)):
            ax.bar(
                   x,
                   ys[x],
                   color=colors[x]
                   )
            plot_boot = g_data[xs[x]][random.sample(range(len(g_data[xs[x]])), k=int(len(g_data[xs[x]])/1))]
            ax.scatter(
                    [(x+random.choice(range(-360,360))*0.001) for _ in plot_boot],
                    plot_boot,
                    alpha=0.2,
                    edgecolors=colors[x],
                    color='white',
                    zorder=2.5,
                    )
        pyplot.xticks(
                    ticks=range(len(xs)),
                    labels=[x.replace('_', '\n') for x in xs],
                    fontsize=23,
                    fontweight='bold'
                    )

        pyplot.yticks(
                    fontsize=20,
                    )
        pyplot.ylabel(ylabel='Spearman rho',
                      fontsize=23,
                      )
        for i in range(len(xs)):
            '''
            ### simple p-value - not really relevant
            p = corr_ps[(target, group, xs[i])]
            if p < p_thr*0.01:
                corrs = [-.1, 0, .1]
            elif p < p_thr*0.1:
                corrs = [-.1, .1]
            elif p < p_thr:
                corrs = [0]
            else:
                corrs = []
            for _ in corrs:
                ax.scatter(
                       i+_,
                       0.28,
                       marker='*',
                       s=150,
                       color='black',
                       zorder=3,
                       )
            '''
            for j in range(len(xs)):
                if i <= j:
                    continue
                p = corr_ps[(target, group, tuple(sorted([xs[i], xs[j]])))]
                if p < p_thr*0.01:
                    corrs = [-.1, 0, .1]
                elif p < p_thr*0.1:
                    corrs = [-.1, .1]
                elif p < p_thr:
                    corrs = [0]
                else:
                    corrs = []
                for _ in corrs:
                    if ys[i]>ys[j]:
                        ax.scatter(
                               i+_,
                               -(3+j)*0.013,
                               marker='s',
                               s=100,
                               color=colors[j],
                               zorder=3,
                               )
                    else:
                        ax.scatter(
                               j+_,
                               -(3+i)*0.013,
                               marker='s',
                               s=100,
                               color=colors[i],
                               zorder=3,
                               )
        legend_xs = numpy.linspace(len(xs)/2-len(xs)/3, ((len(xs))-1)/2+((len(xs))-1)/3, 3)
        ax.scatter(legend_xs[0]-.1, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.text(s='p<0.05', x=legend_xs[0], y=0.275, fontsize=20, ha='left', va='center')
        ax.scatter(legend_xs[1]-.1, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.scatter(legend_xs[1]-.2, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.text(s='p<0.005', x=legend_xs[1], y=0.275, fontsize=20, ha='left', va='center')
        ax.scatter(legend_xs[2]-.3, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.scatter(legend_xs[2]-.2, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.scatter(legend_xs[2]-.1, 0.275, marker='s', s=100, color='white', edgecolor='black')
        ax.text(s='p<0.005', x=legend_xs[2], y=0.275, fontsize=20, ha='left', va='center')
        pyplot.yticks(
                ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25],
                    fontsize=20,
                    )
        marker = 'Ability at' if '_' not in target else 'Improvement between'
        pyplot.title(
                     '{} {}\n\n'\
                     'Contribution of individual {} '\
                     'to RSA pattern correlation'.format(
                         marker,
                         target.replace('_', '-'),
                         group.replace(' ', ' measures at '),
                         ),
                         fontsize=30,
                         fontweight='bold',
                         )
        pyplot.savefig(os.path.join(curr_out, '{}_{}.jpg'.format(target, group)))
        pyplot.clf()
        pyplot.close()
