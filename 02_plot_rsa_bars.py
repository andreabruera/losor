import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

#with open('_results.tsv') as i:
results = dict()
for root, direc, fz in os.walk('rsa_results'):
    for f in fz:
        #print(f)
        if 'tsv' not in f:
            continue
        with open(os.path.join(root, f)) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if 'removal_bootstrap' in line:
                    continue
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
                #    vals = {tuple(v.split(':')[0].split('+')) : float(v.split(':')[1]) for v in vals}
                #else:
                #print(line)
                vals = numpy.array(vals, dtype=numpy.float64)
                results['values'].append(vals)
n_items = set([len(v) for v in results.values()])
assert len(n_items) == 1
n_items = list(n_items)[0]

possibilities = {k : set(v) for k, v in results.items() if k!='values'}
colors_tuple = [
                ('T1', 'yellowgreen'),
                ('T2', 'pink'),
                ('T3', 'mediumaquamarine'),
                #('abilities', 'mediumaquamarine'),
                ('activations T1', 'khaki'),
                ('activations T2', 'darkkhaki'),
                ('connectivity T1', 'paleturquoise'),
                ('connectivity T2', 'lightskyblue'),
                ('lesions', 'lightsalmon'),
                ('age', 'gainsboro'),
                ]
colors = {k : v for k, v in colors_tuple}

for metric in possibilities['metric']:
    for confound_method in possibilities['confound_method']:
        for confound_variable in possibilities['confound_variable']:
            if confound_method == 'raw' and confound_variable != 'none':
                continue
            if confound_method != 'raw' and confound_variable == 'none':
                continue
            ### creating the folder
            out = os.path.join('rsa_plots', confound_method, confound_variable, metric)
            os.makedirs(out, exist_ok=True)
            ### collecting the data and p_values for correction
            rel_idxs = [i for i in range(n_items) if results['metric'][i]==metric and \
                                                  results['confound_method'][i] == confound_method and \
                                                  results['confound_variable'][i] == confound_variable]
            if len(rel_idxs) < 1:
                print([metric, confound_method, confound_variable])
                continue
            current_data = dict()
            raw_ps = dict()
            for i in rel_idxs:
                predictor = results['predictor_name'][i]
                if predictor not in colors.keys():
                    continue
                target = results['target_name'][i]
                key = results['measure'][i]
                #print(key)
                vals = results['values'][i]
                if target not in current_data.keys():
                    current_data[target] = {predictor : {key : vals,}}
                else:
                    if predictor not in current_data[target].keys():
                        current_data[target][predictor] = {key : vals,}
                    else:
                        if key not in current_data[target][predictor].keys():
                            current_data[target][predictor][key] = vals
                ### this allows us to avoid double counts
                if key == 'raw_permutation_p':
                    raw_ps[tuple(sorted((predictor, target)))] = vals[0]
            print(raw_ps.keys())
            ### correcting p-values
            ps = [raw_ps[p] for p in sorted(raw_ps.keys())]
            corr_ps = mne.stats.fdr_correction(ps)[1]
            for p_i, p in enumerate(sorted(raw_ps.keys())):
                corr_p = corr_ps[p_i]
                if p[0] in current_data.keys() and p[1] in current_data[p[0]].keys():
                    current_data[p[0]][p[1]]['corr_p'] = corr_p
                if p[1] in current_data.keys() and p[0] in current_data[p[1]].keys():
                    current_data[p[1]][p[0]]['corr_p'] = corr_p
            ### plotting

            fig, ax = pyplot.subplots(
                                      constrained_layout=True,
                                      figsize=(20, 10),
                                      )
            for k, v in colors_tuple:
                ax.bar(0, 0, label=k, color=v)
            xs_lst = ['T1', 'T2', 'T3', 'T2-T1', 'T3-T2', 'T3-T1']
            #xs_lst = ['abilities', 'T2-T1', 'T3-T2', 'T3-T1']
            xs = {a : _ for _, a in enumerate(xs_lst)}
            ax.text(
                    x=5.21,
                    y=0.73,
                    s='p<0.05',
                    fontsize=20,
                    )
            ax.scatter(
               5.075,
               0.735,
               s=200,
               marker = '*',
                color='white',
                zorder=3.,
                edgecolors='black'
               )
            ax.text(
                    x=5.21,
                    y=0.69,
                    s='p<=0.1',
                    fontsize=20,
                    )
            ax.text(
                    x=1,
                    y=0.72,
                    s='Language ability',
                    fontsize=25,
                    ha='center',
                    va='center',
                    fontweight='bold'
                    )
            ax.text(
                    x=4,
                    y=0.72,
                    s='Language improvement',
                    fontsize=25,
                    ha='center',
                    va='center',
                    fontweight='bold'
                    )
            pyplot.xticks(
                          ticks = range(len(xs.keys())),
                          labels = [x.replace('_', '-') for x in xs_lst],
                          fontsize=23,
                          )
            for _ in range(len(xs_lst)):
                ax.text(
                        x=_,
                        y=-0.05,
                        s=xs_lst[_].replace('_', '-'),
                        fontsize=28,
                        ha='center',
                        va='center',
                        fontweight='bold'
                        )

            ax.scatter(
               5.075,
               0.695,
               s=100,
               marker = 'o',
                color='white',
                zorder=3.,
                edgecolors='black'
               )
            ax.hlines(
                      y=[_*0.1 for _ in range(8)],
                      xmin=-.45,
                      xmax=len(xs)-.45,
                      alpha=0.2,
                      linestyles='--',
                      color='gray',
                      )
            ax.vlines(
                      x=[_+0.5 for _ in range(3)],
                      #x=[_+2+0.5 for _ in range(len(xs))],
                      ymin=-.05,
                      ymax=.75,
                      alpha=0.2,
                      linestyles='-',
                      color='black',
                      )
            ax.vlines(
                      x=[2.5],
                      #x=[_+2+0.5 for _ in range(len(xs))],
                      ymin=-.1,
                      ymax=.75,
                      linestyles='dotted',
                      color='black',
                      linewidth=5,
                      )
            ax.vlines(
                      x=[_+0.5 for _ in range(3, 5)],
                      #x=[_+2+0.5 for _ in range(len(xs))],
                      ymin=-.05,
                      ymax=.75,
                      alpha=0.2,
                      linestyles='-',
                      color='black',
                      )
            for target, target_data in current_data.items():
                n_xs = [v[0] for v in colors_tuple if v[0] in target_data.keys()]
                corrections = {dim : corr for corr, dim in zip(numpy.linspace(-.36, .36, len(n_xs)), n_xs)}
                #corrections = {dim : (_-2.5)*0.2 for _, dim in enumerate([v[0] for v in colors_tuple if v[0] in a_data.keys()])}
                #for dim, dim_data in a_data.items():
                for dim, _ in colors_tuple:
                    if dim not in corrections.keys():
                        continue
                    if len(xs) == 7:
                        wid = 0.135
                    else:
                        wid = 0.095
                    #print(target_data[dim].keys())
                    ax.bar(
                           xs[target]+corrections[dim],
                           numpy.average(target_data[dim]['real']),
                           width=0.1,
                           color=colors[dim]
                           )
                    if confound_method == 'cv-confound':
                        rel_k = 'real'
                    else:
                        rel_k = 'bootstrap'
                    plot_boot = numpy.array(target_data[dim][rel_k])
                    #[random.sample(range(len(boot)), k=int(len(boot)/10))]
                    ax.scatter(
                            [xs[target]+corrections[dim]+(random.choice(range(-30,30))*0.001) for _ in plot_boot],
                            plot_boot,
                            alpha=0.1,
                            edgecolors=colors[dim],
                            color='white',
                            zorder=2.5,
                            )
                    #print(target_data[dim].keys())
                    if float(str(target_data[dim]['corr_p'])[:4]) < 0.05:
                        ax.scatter(
                           xs[target]+corrections[dim],
                           0.02,
                           s=200,
                           marker = '*',
                            color='white',
                            zorder=3.,
                            edgecolors='black'
                           )
                    elif float(str(target_data[dim]['corr_p'])[:4]) <= 0.1:
                        ax.scatter(
                           xs[target]+corrections[dim],
                           0.02,
                           s=100,
                           marker = 'o',
                            color='white',
                            zorder=3.,
                            edgecolors='black'
                           )
            pyplot.legend(
                         fontsize=21,
                         ncols=9,
                         loc=9,
                         borderpad=0.2,
                         columnspacing=1.,
                         handletextpad=0.2,
                         )
            ax.set_ylim(top=0.82, bottom=-.12)
            ax.spines[['right', 'bottom', 'top']].set_visible(False)
            ax.margins(x=.01, y=0.)
            pyplot.xticks(ticks=())

            ##pyplot.title(
            #             'RSA pattern similarities',
            #             fontsize=25,
            #             fontweight='bold',
            #             )
            pyplot.ylabel(ylabel='Spearman rho', fontsize=23)
            pyplot.yticks(fontsize=20)
            pyplot.savefig(os.path.join(out, '{}_{}_{}.jpg'.format(confound_method, confound_variable, metric)))
            pyplot.clf()
            pyplot.close()
            with open(os.path.join(out, '{}_{}_{}.results'.format(metric, confound_method, confound_variable)), 'w') as o:
                o.write('target\tpredicted\tcorrelation\traw_p\tcorrected_p\n')
                for target, target_data in current_data.items():
                    for predictor, predictor_data in target_data.items():
                        o.write('{}\t{}\t{}\t{}\t{}\n'.format(target, predictor, numpy.average(predictor_data['real']), predictor_data['raw_permutation_p'][0], predictor_data['corr_p']))
