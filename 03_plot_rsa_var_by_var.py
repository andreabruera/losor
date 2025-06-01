import matplotlib
import mne
import numpy
import os
import random
import re
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats
from tqdm import tqdm

# Using Helvetica as a font
font_folder = '../fonts/'
if os.path.exists(font_folder):
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
                   'olive',
                   'paleturquoise',
                   'mediumvioletred',
                   ],
          'conn' : [
                   'paleturquoise',
                   'mediumaquamarine',
                   'forestgreen',
                   'mediumblue',
                   'palegoldenrod',
                   'khaki',
                   'palevioletred',
                   'mediumorchid',
                   'silver',
                   ],
          'les' : [
                   'deepskyblue',
                   'palevioletred',
                   'mediumseagreen',
                   'darkkhaki',
                   'chocolate',
                   'royalblue',
                   'goldenrod',
                   ],
          }

#with open('z_results.tsv') as i:
results = dict()
for root, direc, fz in os.walk('rsa_results'):
    for f in fz:
        #print(f)
        if 'tsv' not in f:
            continue
        if 'results' not in f:
            continue
        #if 'aphasia' in f:
        #    continue
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
                    vals_dict = dict()
                    for v in vals:
                        key = tuple(v.split(':')[0].split('+'))
                        val = float(v.split(':')[1])
                        if key not in vals_dict.keys():
                            vals_dict[key] = list()
                        vals_dict[key].append(val)
                    del vals
                    vals = {k : v for k, v in vals_dict.items()}
                results['values'].append(vals)
n_items = set([len(v) for v in results.values()])
assert len(n_items) == 1
n_items = list(n_items)[0]
p_thr = 0.05
approaching_thr = 0.1

possibilities = {k : set(v) for k, v in results.items() if k!='values'}

for metric in possibilities['metric']:
    for confound_method in possibilities['confound_method']:
        for confound_variable in possibilities['confound_variable']:
            if confound_method == 'raw' and confound_variable != 'none':
                continue
            if confound_method != 'raw' and confound_variable == 'none':
                continue
            cases = list()
            in_f = os.path.join('rsa_plots', confound_method, confound_variable, metric)
            with open(os.path.join(in_f, '{}_{}_{}.results'.format(metric, confound_method, confound_variable))) as i:
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    if float(line[-1][:4]) <= approaching_thr:
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
                    print(predictor)
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
                #assert sum([len(v) for v in vals.values()]) == 1000
                for k, v in vals.items():
                    for ind_k in k:
                        if ind_k not in current_data[target][predictor].keys():
                            current_data[target][predictor][ind_k] = v
                        else:
                            current_data[target][predictor][ind_k].extend(v)
            ### computing p-values, and correcting them
            raw_ps = list()
            with tqdm() as counter:
                for c_targ, c_targ_data in current_data.items():
                    for c_pred, c_pred_data in c_targ_data.items():
                        ind_preds = list(c_pred_data.keys())
                        assert len(ind_preds) > 2
                        for c_one_i, c_one in enumerate(ind_preds):
                            for c_two_i, c_two in enumerate(ind_preds):
                                if c_two_i <= c_one_i:
                                    continue
                                data_one = c_pred_data[c_one]
                                data_two = c_pred_data[c_two]
                                ### quicker:t-test
                                if numpy.nanmean(data_one) > numpy.nanmean(data_two):
                                    alt = 'greater'
                                else:
                                    alt = 'less'
                                raw_p = scipy.stats.ttest_ind(data_one, data_two, alternative=alt).pvalue
                                '''
                                ### permutation test
                                ### real mean
                                avg_one = numpy.nanmean(data_one)
                                avg_two = numpy.nanmean(data_two)
                                ### permuting
                                full_set = data_one + data_two
                                if avg_one > avg_two:
                                    real = avg_one-avg_two
                                else:
                                    real = avg_two-avg_one
                                fakes = list()
                                for _ in range(1000):
                                    fake_one_idxs = random.sample(range(len(full_set)), k=len(data_one))
                                    fake_one = [full_set[i] for i in fake_one_idxs]
                                    fake_two = [full_set[i] for i in range(len(full_set)) if i not in fake_one_idxs]
                                    ### fake mean
                                    avg_fake_one = numpy.nanmean(fake_one)
                                    avg_fake_two = numpy.nanmean(fake_two)
                                    if avg_one > avg_two:
                                        fake = avg_fake_one-avg_fake_two
                                    else:
                                        fake = avg_fake_two-avg_fake_one
                                    fakes.append(fake)
                                raw_p = (sum([1 for _ in fakes if _>real])+1)/(len(fakes)+1)
                                '''
                                current_data[c_targ][c_pred][tuple(sorted((c_one, c_two)))] = {'raw_permutation_p' : raw_p}
                                raw_ps.append(((c_targ, c_pred, c_one, c_two), raw_p))
                                counter.update(1)
            corr_ps = mne.stats.fdr_correction([v[1] for v in raw_ps])[1]
            for kv, p in zip(raw_ps, corr_ps):
                k = kv[0]
                if p< 0.05:
                    print([k, p])
                current_data[k[0]][k[1]][tuple(sorted((k[2], k[3])))]['corr_p'] = p
            ### variables other than lesions
            for target, t_data in current_data.items():
                for group, g_data in t_data.items():
                    if group == 'lesions':
                        continue
                    fig, ax = pyplot.subplots(
                                              constrained_layout=True,
                                              figsize=(20, 10),
                                              )
                    ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
                    ax.margins(x=.01, y=0.)
                    pyplot.xticks(ticks=())
                    if 'les' in group:
                        xs = sorted([k for k in g_data.keys() if type(k)!=tuple])
                    if 'conn' in group or 'act' in group:
                        dlpfc = sorted([k for k in g_data.keys() if type(k)!=tuple and 'DLP' in k])
                        sma = sorted([k for k in g_data.keys() if type(k)!=tuple and 'SMA' in k])
                        ptl =sorted([k for k in g_data.keys() if type(k)!=tuple and k not in dlpfc and k not in sma and 'PTL' in k])
                        lang =sorted([k for k in g_data.keys() if type(k)!=tuple and k not in dlpfc and k not in sma and k not in ptl])
                        xs = dlpfc+sma+ptl+lang
                        ax.vlines(
                                  x=[len(dlpfc)+len(sma)-.5],
                                  #x=[_+2+0.5 for _ in range(len(xs))],
                                  #ymin=len(xs)-,
                                  ymin=-(len(xs)*2.1)*0.013,
                                  ymax=.3,
                                  linestyles='dotted',
                                  color='black',
                                  linewidth=5,
                                  )
                        if 'act' in group:
                            xpos=1.
                        else:
                            xpos=2.5
                        ax.text(
                                #x=((len(dlpfc)+len(sma))/2)-1,
                                x=xpos,
                                y=0.275,
                                s='Domain-general',
                                fontsize=23,
                                ha='center',
                                fontweight='bold',
                                )
                        if 'act' in group:
                            xpos=4.
                        else:
                            xpos=7
                        ax.text(
                                #x=((len(dlpfc)+len(sma))+((len(sma)+len(ptl))/2))-1,
                                x=xpos,
                                y=0.275,
                                s='Language-specific',
                                fontsize=23,
                                fontweight='bold',
                                ha='center',
                                )
                    print(xs)
                    assert len(xs) > 2
                    ### creating the folder
                    out = os.path.join(
                                       'rsa_plots',
                                       confound_method,
                                       confound_variable,
                                       metric,
                                       'individual_variables',
                                       target,
                                       group,
                                       )
                    os.makedirs(out, exist_ok=True)
                    for k in ['act', 'les', 'conn']:
                        if k in group:
                            colors = all_colors[k]
                    ys = [numpy.nanmean(g_data[x]) for x in xs]
                        #for x, y in zip(xs, ys):
                    ax.set_ylim(bottom=-(len(ys)*2.)*0.013, top=.32)
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
                        plot_boot = g_data[xs[x]]
                        ax.scatter(
                                [(x+random.choice(range(-360,360))*0.001) for _ in plot_boot],
                                plot_boot,
                                alpha=0.2,
                                edgecolors=colors[x],
                                color='white',
                                zorder=2.5,
                                )
                    edited_xticks = list()
                    for x in xs:
                        var = re.sub('^l|^L_', 'left ', x)
                        var = re.sub('^r|^R_', 'right ', var)
                        var = re.sub('_r', '_right ', var)
                        var = re.sub('_l', '_left ', var)
                        var = var.replace('_', '\n')
                        edited_xticks.append(var)
                    for x_i, x in enumerate(edited_xticks):
                        ax.text(
                                x=x_i,
                                y=-(3+len(xs))*0.013,
                                #y=0,
                                #y=-(len(xs)*0.01),
                                s=x,
                                fontsize=20,
                                fontweight='bold',
                                va='top',
                                ha='center',
                                )
                    pyplot.ylabel(ylabel='Spearman rho',
                                  fontsize=23,
                                  #loc='top',
                                  y=.75,
                                  #va='center',
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
                                #if p < p_thr:
                                corrs = [0]
                            else:
                                corrs = []
                            #if len(corrs)>0:
                            #    print([p, corrs])
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
                    if 'conn' in group:
                        xpos=-.8
                        y=-((len(xs)*1.5)/2)*0.013
                    elif 'act' in group:
                        xpos=-.65
                        y=-((len(xs)*2)/2)*0.013
                    else:
                        xpos=-.5
                        y=-((len(xs)*2.5)/2)*0.013
                    ax.text(
                            x=xpos,
                            y=y,
                            s='significant\ncomparisons',
                            fontsize=20,
                            rotation=90,
                            va='center',
                            )

                    pyplot.yticks(
                            ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                                fontsize=20,
                                )
                    marker = 'Ability at' if '-' not in target else 'Improvement between'
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
                    with open(os.path.join(out, '{}_{}_results.tsv'.format(target, group)), 'w') as o:
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
            ### lesions
            ### here we have only one plot for abilities/improvements
            order = {
                     'abilities' : ['T1', 'T2', 'T3'],
                     'improvements' : ['T2-T1', 'T3-T2', 'T3-T1'],
                     'both' : ['T1', 'T2', 'T3', 'T2-T1', 'T3-T2', 'T3-T1']
                     }
            for case_marker in ['abilities', 'improvements', 'both']:
                xs = dict()
                for target, t_data in current_data.items():
                    if case_marker == 'improvements' and '-' not in target:
                        continue
                    elif case_marker == 'abilities' and '-' in target:
                        continue
                    for group, g_data in t_data.items():
                        if group != 'lesions':
                            continue
                        if type(group) == tuple:
                            continue
                        xs[target] = g_data
                try:
                    for k in order[case_marker]:
                        assert k in xs.keys()
                except AssertionError:
                    continue

                fig, ax = pyplot.subplots(
                                          constrained_layout=True,
                                          figsize=(20, 10),
                                          )
                ax.spines[['left', 'right', 'bottom', 'top']].set_visible(False)
                ax.margins(x=.01, y=0.)
                pyplot.xticks(ticks=())
                legended = dict()
                assert len(xs.keys()) in [3, 6]
                ### creating the folder
                out = os.path.join(
                                   'rsa_plots',
                                   confound_method,
                                   confound_variable,
                                   metric,
                                   'individual_variables',
                                   'lesions',
                                   case_marker,
                                   )
                os.makedirs(out, exist_ok=True)
                colors = all_colors['les']
                ax.set_ylim(bottom=-(3*2.)*0.013, top=.38)
                ax.hlines(
                          y=[0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                          xmin=-.4,
                          xmax=len(xs)-.6,
                          alpha=0.2,
                          linestyles='--',
                          color='gray',
                          )
                #for correction, vals in enumerate(xs.items()):
                for correction, target in enumerate(order[case_marker]):
                    data = {k : v for k, v in xs[target].items() if type(k)!=tuple}
                    ys = [numpy.nanmean(data[x]) for x in sorted(data.keys())]
                    curr_xs = [-0.3, 0, .3]
                    #for x in range(len(ys)):
                    for x_i, x in enumerate(curr_xs):
                        key = sorted(data.keys())[x_i]
                        if case_marker != 'both':
                            color = colors[x_i]
                        else:
                            if correction < 3:
                                color = colors[x_i]
                            else:
                                #color = colors[x_i+3]
                                color = colors[x_i]
                        if color not in legended.keys():
                            ax.bar(0, 0, color=color, label=key.replace('_', ' '))
                            legended[color] = True
                        ax.bar(
                               x+correction,
                               ys[x_i],
                               color=color,
                               width=0.25
                               )
                        #plot_boot = g_data[xs[x]]
                        plot_boot = current_data[target]['lesions'][key]
                        ax.scatter(
                                [(x+correction+random.choice(range(-100,100))*0.001) for _ in plot_boot],
                                plot_boot,
                                alpha=0.2,
                                edgecolors=color,
                                color='white',
                                zorder=2.5,
                                )
                        for j in range(len(ys)):
                            if x_i == j:
                                continue
                            p_key_two = sorted(data.keys())[j]
                            p = current_data[target]['lesions'][tuple(sorted((key, p_key_two)))]['corr_p']
                            #if numpy.nanmean(ys[x_i]) < numpy.nanmean(ys[j]):
                            #    continue
                            if p < p_thr*0.01:
                                corrs = [-.05, 0, .05]
                            elif p < p_thr*0.1:
                                corrs = [-.05, .05]
                            elif p < p_thr:
                                #if p < p_thr:
                                corrs = [0]
                            else:
                                corrs = []
                            #if len(corrs)>0:
                            #    print([p, corrs])
                            for _ in corrs:
                                if ys[x_i]>ys[j]:
                                    if case_marker != 'both':
                                        other_color = colors[j]
                                    else:
                                        if correction < 3:
                                            other_color = colors[j]
                                        else:
                                            #other_color = colors[j+3]
                                            other_color = colors[j]
                                    ax.scatter(
                                           correction+x+_,
                                           -(3+j)*0.013,
                                           marker='s',
                                           s=100,
                                           color=other_color,
                                           zorder=3,
                                           )
                                else:
                                    continue
                                    ax.scatter(
                                           correction+curr_xs[j]+_,
                                           -(3+x_i)*0.013,
                                           marker='s',
                                           s=100,
                                           color=color,
                                           zorder=3,
                                           )
                edited_xticks = list()
                for x in order[case_marker]:
                    var = re.sub('^l|^L_', 'left ', x)
                    var = re.sub('^r|^R_', 'right ', var)
                    var = re.sub('_r', '_right ', var)
                    var = re.sub('_l', '_left ', var)
                    var = var.replace('_', '\n')
                    edited_xticks.append(var)
                for x_i, x in enumerate(edited_xticks):
                    ax.text(
                            x=x_i,
                            y=-(5+2)*0.013,
                            #y=0,
                            #y=-(len(xs)*0.01),
                            s=x,
                            fontsize=30,
                            fontweight='bold',
                            va='top',
                            ha='center',
                            )
                if case_marker == 'both':
                    for x_m_i, x_m in enumerate((['Language ability', 'Language improvement'])):
                        x_coord = 1 if x_m_i == 0 else 4
                        ax.text(
                                x=x_coord,
                                y=.305,
                                s=x_m,
                                fontsize=30,
                                fontweight='bold',
                                va='center',
                                ha='center',
                                )
                pyplot.ylabel(ylabel='Spearman rho',
                              fontsize=23,
                              #loc='top',
                              y=.75,
                              #va='center',
                              )
                if case_marker == 'both':
                    ax.vlines(
                              x=[.5, 1.5, 3.5, 4.5],
                              ymin=-(3*2.1)*0.013,
                              ymax=.28,
                              linestyles='dotted',
                              color='silver',
                              linewidth=5,
                              )
                    ax.vlines(
                              x=[2.5,],
                              ymin=-(3*2.1)*0.013,
                              ymax=.3,
                              linestyles='dashed',
                              color='black',
                              linewidth=7,
                              )
                else:
                    ax.vlines(
                              x=[.5, 1.5,],
                              ymin=-(3*2.1)*0.013,
                              ymax=.3,
                              linestyles='dotted',
                              color='black',
                              linewidth=5,
                              )
                xpos=-.68
                y=-((3.8*2.5)/2)*0.013
                ax.text(
                        #x=-(len(xs)*0.09),
                        x=xpos,
                        y=y,
                        #y=-((len(xs)*1.5)/2)*0.013,
                        #s='statistical\nsignificance',
                        s='significant\ncomparisons',
                        fontsize=20,
                        rotation=90,
                        va='center',
                        )
                ax.legend(loc=9, fontsize=23, ncols=3)
                pyplot.yticks(
                        ticks=[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            fontsize=20,
                            )
                marker = 'Ability at' if '-' not in target else 'Improvement between'
                pyplot.title(
                             '{}\n\n'\
                             'Contribution of individual lesions '\
                             'to RSA pattern correlation'.format(
                                 case_marker.capitalize(),
                                 ),
                                 fontsize=30,
                                 fontweight='bold',
                                 )
                pyplot.savefig(os.path.join(out, '{}.jpg'.format(case_marker)))
                pyplot.clf()
                pyplot.close()
                with open(os.path.join(out, '{}_results.tsv'.format(case_marker)), 'w') as o:
                    o.write('target\tpredictor_one\tpredictor_two\t')
                    o.write('average_corr_one\taverage_corr_two\t')
                    o.write('raw_permutation_p\tfdr_corrected_p\n')
                    for target in order[case_marker]:
                        keys = [k for k in xs[target].keys() if type(k)!=tuple]
                        for x_i, x in enumerate(keys):
                            for x_two_i, x_two in enumerate(keys):
                                if x_two_i <= x_i:
                                    continue
                                o.write('{}\t{}\t{}\t'.format(target, x, x_two))
                                #o.write('{}\t{}\t'.format(ys[x_i], ys[x_two_i]))
                                o.write('{}\t{}\t'.format(numpy.nanmean(xs[target][x]), numpy.nanmean(xs[target][x_two])))
                                o.write('{}\t'.format(current_data[target]['lesions'][tuple(sorted([x, x_two]))]['raw_permutation_p']))
                                o.write('{}\n'.format(current_data[target]['lesions'][tuple(sorted([x, x_two]))]['corr_p']))
