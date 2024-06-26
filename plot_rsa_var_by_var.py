import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats
from tqdm import tqdm

# Using Helvetica as a font
font_folder = 'fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

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

#with open('zz_results.tsv') as i:
results = dict()
for root, direc, fz in os.walk('rsa_results_zz'):
    for f in fz:
        #print(f)
        if 'tsv' not in f:
            continue
        with open(os.path.join(root, f)) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i == 0:
                    header = [h for h in line]
                    #print(header)
                    for h in header+['values']:
                        if h not in results.keys():
                            results[h] = list()
                    continue
                details = [line[h_i] for h_i, h in enumerate(header)]
                #print(details[header.index('predictor_name')])
                #print(details)
                for h, d in zip(header, details):
                    results[h].append(d)
                ### values
                vals = line[len(header):]
                if 'removal_bootstrap' not in line:
                    vals = numpy.array(vals, dtype=numpy.float64)
                else:
                    vals = {tuple(v.split(':')[0].split('+')) : float(v.split(':')[1]) for v in vals}
                results['values'].append(vals)
'''
with open('zz_results.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [h for h in line]
            results = {h : list() for h in header+['values']}
            continue
        for h_i, h in enumerate(header):
            results[h].append(line[h_i])
        ### values
        vals = line[len(header):]
        if 'removal_bootstrap' not in line:
            vals = numpy.array(vals, dtype=numpy.float64)
        else:
            vals = {tuple(v.split(':')[0].split('+')) : float(v.split(':')[1]) for v in vals}
        results['values'].append(vals)
'''
n_items = set([len(v) for v in results.values()])
assert len(n_items) == 1
n_items = list(n_items)[0]
p_thr = 0.05

possibilities = {k : set(v) for k, v in results.items() if k!='values'}

for metric in possibilities['metric']:
    for confound_method in possibilities['confound_method']:
        for confound_variable in possibilities['confound_variable']:
            if confound_method == 'raw' and confound_variable != 'none':
                continue
            if confound_method != 'raw' and confound_variable == 'none':
                continue
            cases = list()
            in_f = os.path.join('rsa_zz', confound_method, confound_variable)
            with open(os.path.join(in_f, '{}_{}_{}.results'.format(metric, confound_method, confound_variable))) as i:
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    if float(line[-1][:4]) <= 0.1:
                        cases.append((line[0], line[1]))
            ### collecting the data and p_values for correction
            rel_idxs = [i for i in range(n_items) if results['metric'][i]==metric and \
                                                  results['confound_method'][i] == confound_method and \
                                                  results['confound_variable'][i] == confound_variable]
            if len(rel_idxs) < 1:
                print([metric, confound_method, confound_variable])
                continue
            current_data = dict()
            for i in rel_idxs:
                predictor = results['predictor_name'][i]
                if predictor not in [
                                     'lesions',
                                     'activations T1',
                                     'activations T2',
                                     'connectivity T1',
                                     'connectivity T2',
                                     ]:
                    continue
                #print(predictor)
                target = results['target_name'][i]
                key = results['measure'][i]
                if 'removal' not in key:
                    continue
                if (target, predictor) not in cases:
                    continue
                vals = results['values'][i]
                if target not in current_data.keys():
                    current_data[target] = {predictor : dict()}
                else:
                    if predictor not in current_data[target].keys():
                        current_data[target][predictor] = dict()
                for k, v in tqdm(vals.items()):
                    for ind_k in k:
                        if ind_k not in current_data[target][predictor].keys():
                            current_data[target][predictor][ind_k] = [v]
                        else:
                            current_data[target][predictor][ind_k].append(v)
            ### computing p-values, and correcting them
            raw_ps = list()
            for c_targ, c_targ_data in current_data.items():
                for c_pred, c_pred_data in c_targ_data.items():
                    ind_preds = list(c_pred_data.keys())
                    assert len(ind_preds) > 2
                    for c_one_i, c_one in enumerate(ind_preds):
                        for c_two_i, c_two in enumerate(ind_preds):
                            if c_two_i <= c_one_i:
                                continue
                            ### permuting
                            ### real mean
                            data_one = c_pred_data[c_one]
                            data_two = c_pred_data[c_two]
                            avg_one = numpy.nanmean(data_one)
                            avg_two = numpy.nanmean(data_two)
                            full_set = data_one + data_two
                            if avg_one > avg_two:
                                real = avg_one-avg_two
                            else:
                                real = avg_two-avg_one
                            fakes = list()
                            for _ in range(1000):
                                #fake_one = random.sample(full_set, k=len(data_one))
                                #fake_two = random.sample(full_set, k=len(data_two))
                                #fake_two =
                                fake_one_idxs = random.sample(range(len(full_set)), k=len(data_one))
                                fake_one = [full_set[i] for i in fake_one_idxs]
                                fake_two = [full_set[i] for i in range(len(full_set)) if i not in fake_one_idxs]
                                ### fake mean
                                avg_fake_one = numpy.nanmean(fake_one)
                                avg_fake_two = numpy.nanmean(fake_two)
                                if avg_one > avg_two:
                                    fake = avg_one-avg_two
                                else:
                                    fake = avg_two-avg_one
                                fakes.append(fake)
                            raw_p = (sum([1 for _ in fakes if _>real])+1)/(len(fakes)+1)
                            current_data[c_targ][c_pred][tuple(sorted((c_one, c_two)))] = {'raw_permutation_p' : raw_p}
                            raw_ps.append(((c_targ, c_pred, c_one, c_two), raw_p))
            corr_ps = mne.stats.fdr_correction([v[1] for v in raw_ps])[1]
            for kv, p in zip(raw_ps, corr_ps):
                k = kv[0]
                if p< 0.05:
                    print(k)
                current_data[k[0]][k[1]][tuple(sorted((k[2], k[3])))]['corr_p'] = p
            for target, t_data in current_data.items():
                for group, g_data in t_data.items():
                    xs = [k for k in g_data.keys() if type(k)!=tuple]
                    assert len(xs) > 2
                    ### creating the folder
                    out = os.path.join('rsa_zz', confound_method, confound_variable, 'individual_variables', target, group)
                    os.makedirs(out, exist_ok=True)
                    for k in ['act', 'les', 'conn']:
                        if k in group:
                            colors = all_colors[k]
                    ys = [numpy.nanmean(g_data[x]) for x in xs]
                    fig, ax = pyplot.subplots(
                                              constrained_layout=True,
                                              figsize=(20, 10),
                                              )
                        #for x, y in zip(xs, ys):
                    ax.set_ylim(bottom=-(3+len(ys))*0.013, top=.32)
                    ax.hlines(
                              y=[0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
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
                        #plot_boot = g_data[xs[x]][random.sample(range(len(g_data[xs[x]])), k=int(len(g_data[xs[x]])/1))]
                        plot_boot = g_data[xs[x]]
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

                    pyplot.ylabel(ylabel='Spearman rho',
                                  fontsize=23,
                                  )
                    for i in range(len(xs)):
                        for j in range(len(xs)):
                            if i <= j:
                                continue
                            p = g_data[tuple(sorted((xs[i], xs[j])))]['corr_p']
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
                    ax.scatter(legend_xs[0]-.1, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.text(s='p<0.05', x=legend_xs[0], y=0.31, fontsize=20, ha='left', va='center')
                    ax.scatter(legend_xs[1]-.1, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.scatter(legend_xs[1]-.2, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.text(s='p<0.005', x=legend_xs[1], y=0.31, fontsize=20, ha='left', va='center')
                    ax.scatter(legend_xs[2]-.3, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.scatter(legend_xs[2]-.2, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.scatter(legend_xs[2]-.1, 0.31, marker='s', s=100, color='white', edgecolor='black')
                    ax.text(s='p<0.005', x=legend_xs[2], y=0.31, fontsize=20, ha='left', va='center')
                    pyplot.yticks(
                            ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
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
                    pyplot.savefig(os.path.join(out, '{}_{}.jpg'.format(target, group)))
                    pyplot.clf()
                    pyplot.close()
                    with open(os.path.join(out, '{}_{}.results'.format(target, group)), 'w') as o:
                        o.write('predictor_one\tpredictor_two\t')
                        o.write('average_corr_one\taverage_corr_two\t')
                        o.write('raw_permutation_p\tfdr_corrected_p\n')
                        for x_i, x in enumerate(xs):
                            for x_two_i, x_two in enumerate(xs):
                                if x_two_i <= x_i:
                                    continue
                                o.write('{}\t{}\t'.format(x, x_two))
                                o.write('{}\t{}\t'.format(ys[x_i], ys[x_two_i]))
                                o.write('{}\t'.format(g_data[tuple(sorted([x, x_two]))]['raw_permutation_p']))
                                o.write('{}\n'.format(g_data[tuple(sorted([x, x_two]))]['corr_p']))
