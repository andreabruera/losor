import matplotlib
import numpy
import os
import random
import re

from matplotlib import pyplot

### reading p-values
p_vals = dict()
with open(os.path.join('results', 'p-vals.tsv')) as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        assert len(line) == 5
        p_vals[(line[0].replace('_', '-'), line[1])] = float(line[3])

### reading results
predictors = set()
results = dict()
real_f = os.path.join('data', 'real')
assert os.path.exists(real_f)
for f in os.listdir(real_f):
    if 'tsv' not in f:
        continue
    case = f.split('.')[0]
    full_case = 'ability {}'.format(case) if len(case)==2 else 'improvement {}'.format(case.replace('_', '-'))
    results[full_case] = dict()
    real = dict()
    perm = dict()
    with open(os.path.join(real_f, f)) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')[4:]
            for pred in line:
                pred = [w.strip() for w in re.sub(r"\(|\)|'", '', pred).split(',')]
                results[full_case][pred[0]] = float(pred[1])
                predictors.add(pred[0])
'''
with open('sorted_coefs.tsv') as i:
    for l_i, l in enumerate(i):
        line = [w.strip() for w in l.split('\t')]
        if l_i == 0:
            header = [h.replace('Predicting ', '') for h in line]
            results = {h : dict() for h in header}
            continue
        for cell_i, cell in enumerate(line):
            if cell == '':
                continue
            cell = re.sub(r"\(|\)|'", '', cell)
            split_cell = [c.strip() for c in cell.split(',')]
            #if split_cell[0] == 'T1':
            if header[cell_i] == 'ability T3':
                print([header[cell_i], split_cell])
            assert len(split_cell) == 2
            predictors.add(split_cell[0])
            results[header[cell_i]][split_cell[0]] = float(split_cell[1])
'''

#### normalizing in the range 0, 1
#results = {k : {k_two : v_two/max(v.values()) for k_two, v_two in v.items()} for k, v in results.items()}

predictors = {
            'connectivity' : [w for w in predictors if '_to_' in w],
            'lesion' : [w for w in predictors if '_' in w and '_to_' not in w and '_T' not in w],
            'activation' : [w for w in predictors if '_' in w and '_to_' not in w and '_T' in w],
            'other' : [w for w in predictors if '_' not in w]
            }
print(predictors)
### plotting abilities
for folder in [
               'all_predictors',
               'significant_predictors',
               ]:
    for t_label, time_sorted in [
            (
              'ability', [
              'ability T2',
              'ability T3',
              ]
             ),
            (
              'improvement', [
              'improvement T2-T1',
              'improvement T3-T2',
              'improvement T3-T1',
              ]
              ),
            (
              'ability_and_improvement', [
              'improvement T2-T1',
              'ability T2',
              'improvement T3-T2',
              'improvement T3-T1',
              'ability T3',
              ]
              )
            ]:

        full_folder = os.path.join('plots', folder, t_label)
        os.makedirs(
                    full_folder,
                    exist_ok=True,
                     )
        xs = list(range(len(time_sorted)))
        for cat, labels in predictors.items():
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            title = 'Importance of {} ({} for {}'.format(cat, folder.replace('_', ') '), t_label)
            data = list()
            for k, v in results.items():
                #if mode not in k:
                #    continue
                for k_two, v_two in v.items():
                    if k_two not in labels:
                        continue
                    #if k_two == 'T1':
                    #    print([k, k_two, v_two])
                    data.append((k, k_two, v_two, p_vals[(k.split()[-1], k_two)]))
            #time_sorted = sorted(set([w[0] for w in data]))
            label_sorted = sorted(set([w[1] for w in data]))
            cmap = matplotlib.colormaps['tab20']
            colors=cmap(numpy.linspace(0.,1,len(label_sorted)))
            all_vals = list()
            star_marker = False
            for l_i, l in enumerate(label_sorted):
                ys = list()
                ps = list()
                ### filtering
                for t in time_sorted:
                    marker = False
                    for time, lab, val, p in data:
                        if l == lab and t == time:
                            ys.append(val)
                            ps.append(p)
                            all_vals.append(val)
                            marker = True
                    if marker == False:
                        ys.append(numpy.nan)
                        ps.append(1)
                rand_xs = [x+(random.randrange(-5, 5)*0.02) for x in xs]
                linewidth = 7
                label=l.replace('orb', '_ORB')\
                           .replace('l', 'left_')\
                           .replace('r', 'right_')\
                           .replace('PTL', 'PTlobe')\
                           .replace('L_', 'left_')\
                           .replace('R_', 'right_')\
                           .replace('lobe', 'L')\
                           .replace('_', ' ')
                for x, y, p in zip(rand_xs, ys, ps):
                    if p <= 0.05:
                        ax.scatter(
                                   x,
                                   y,
                                   marker='*',
                                   s=1000,
                                   facecolor="none",
                                   edgecolors='black',
                                   linewidths=3,
                                   zorder=3,
                                   )
                        ax.scatter(
                                   x,
                                   y,
                                   marker='*',
                                   s=2000,
                                   facecolor="none",
                                   edgecolors='white',
                                   linewidths=3,
                                   zorder=3,
                                   )
                        ax.scatter(
                               x,
                               y,
                               marker='*',
                               s=500,
                               color=colors[l_i],
                               #facecolor="none",
                               #edgecolors='white',
                               #linewidths=3,
                               zorder=3,
                               )
                    else:
                        if folder == 'significant_predictors' and True not in [True for p in ps if p<0.05]:
                            continue
                        else:
                            ax.scatter(
                                       x,
                                       y,
                                       marker='D',
                                       s=200,
                                       color=colors[l_i],
                                       )
                            ax.scatter(
                                       x,
                                       y,
                                       marker='D',
                                       s=300,
                                       facecolor="none",
                                       edgecolors='white',
                                       linewidths=3,
                                       zorder=3,
                                       )
                if folder == 'significant_predictors' and True not in [True for p in ps if p<0.05]:
                    continue
                else:
                    ax.plot(
                            rand_xs,
                            ys,
                            linewidth=linewidth,
                            color=colors[l_i],
                            label=label,
                            alpha=0.5
                            )
            if 'activation' in cat:
                mult = 3
            elif 'connectivity' in cat:
                mult = 6
            else:
                mult = 1
            ax.set_ylim(
                        bottom=min(all_vals)-(numpy.std(all_vals)*mult),
                        )
            ax.set_xlim(left=-0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            pyplot.xticks(
                          ticks=xs,
                          labels=[t.replace(' ', '\n') for t in time_sorted],
                          fontsize=20,
                          )
            pyplot.yticks(
                          fontsize=20,
                          )
            if cat == 'other':
                top = 1
            if cat == 'activation':
                top = 2
            if cat == 'connectivity':
                top = 3
            if cat == 'lesion':
                top = 2
            for n, label in enumerate(ax.yaxis.get_ticklabels()):
                if n < top:
                    label.set_visible(False)
            pyplot.ylabel(
                          ylabel='Average weight in Elastic Net',
                          fontsize=23,
                          fontweight='bold',
                          labelpad=10,
                          )
            pyplot.xlabel(
                          xlabel='Prediction target',
                          fontsize=23,
                          fontweight='bold',
                          labelpad=10,
                          )
            pyplot.title(
                         title.replace('_', ' '),
                         fontsize=25,
                         fontweight='bold',
                         )
            ax.hlines(
                      [0.],
                      xmin=-0.15,
                      xmax=len(xs)-.85,
                      color='gray',
                      linestyles='--',
                      alpha=0.5,
                      )
            ax.hlines(
                      ax.get_yticks(),
                      xmin=-0.15,
                      xmax=len(xs)-.85,
                      color='gray',
                      linestyles='--',
                      alpha=0.2,
                      )
            ax.scatter(
                       -1,
                       -2,
                       marker='*',
                       s=1000,
                       facecolor="none",
                       edgecolors='black',
                       linewidths=3,
                       zorder=3,
                       label='p(FDR)<0.05'
                       )
            pyplot.legend(
                          loc=8,
                          ncols=3,
                          fontsize=25,
                          )
            pyplot.savefig(
                           os.path.join(full_folder, '{}_{}_{}.jpg'.format(cat, t_label, folder)),
                           dpi=300,
                           )
            pyplot.clf()
            pyplot.close()
