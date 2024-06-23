import itertools
import matplotlib
import mne
import numpy
import os
import random
import scipy
import sklearn

from matplotlib import font_manager, pyplot
from scipy import spatial
from sklearn import linear_model, metrics
from tqdm import tqdm

# Using Helvetica as a font
font_folder = 'fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

#with open('zz_full.tsv') as i:
with open('zz.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.split('\t')
        line[-1] = line[-1].strip()
        if l_i == 0:
            header = [w for w in line]
            raw_data = {h : list() for h in header[2:] if h!='L_SMA' and 'adjusted' not in h}
            continue
        for d in raw_data.keys():
            val = line[header.index(d)].replace(',', '.')
            if val == '':
                raw_data[d].append(numpy.nan)
            else:
                raw_data[d].append(float(val))
### z scoring
#data = {k : [(val-numpy.average(v))/numpy.std(v) for val in v] for k, v in raw_data.items()}
data = {k : v for k, v in raw_data.items() if len(v)>0}

abilities = ['T1', 'T2', 'T3']
improvements = ['T2_T1', 'T3_T2', 'T3_T1']

age = ['Age']
lesions = ['L_DLPFC', 'L_IFGorb', 'L_PTL',
           #'L_SMA',
           ]
to_be_removed = dict()
for d in raw_data.keys():
    if d in lesions:
        for v_i, v in enumerate(raw_data[d]):
            if v_i not in to_be_removed.keys():
                to_be_removed[v_i] = v
            else:
                to_be_removed[v_i] += v
to_be_removed = [v_i for v_i, v in to_be_removed.items() if v == 0.]
print(to_be_removed)
data = {k : [v[i] for i in range(len(v)) if i not in to_be_removed] for k, v in data.items()}

activations_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T1' in k]
activations_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T2' in k]
conn_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T1' in k]
conn_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T2' in k]

### classic RSA
print('### Classic RSA ###\n')
### bootstrap with 30 subjects
bootstrap_n = len([1 for _ in range(30) for __ in range(30) if __>_])

models = [
        #('age', age),
        ('lesions', lesions),
        ('activations T1', activations_t1),
        ('activations T2', activations_t2),
        ('connectivity T1', conn_t1),
        ('connectivity T2', conn_t2),
        ]
metrics = [
           #'pearson_r',
           'spearman_r',
           #'cosine',
           #'euclidean',
           #'manhattan',
           ]
corrections = {k : (v-1.)*0.15 for k, v in zip(metrics, range(len(metrics)))}
colors = ['teal', 'goldenrod', 'plum', 'gray']
colors = {k : v for k, v in zip(metrics, colors)}
plot_colors = ['a', 'b', 'c', 'd']

removal_results = dict()
single_plot = dict()
montecarlo = 100

#for ab in abilities:
for ab in abilities+improvements:
    fig, ax = pyplot.subplots(
                              constrained_layout=True,
                              figsize=(20, 10),
                              )
    ax.set_ylim(
                bottom=-.1,
                top=0.75,
                )
    if '_' in ab:
        add = 3
    else:
        add = 2
    ax.hlines(
              y=[_*0.1 for _ in range(-1, 8)],
              xmin=-.25,
              xmax=len(models)+add-.75,
              alpha=0.2,
              linestyles='--'
              )
    ticks = list()
    for metric in metrics:
        if metric not in single_plot.keys():
            single_plot[metric] = dict()
            removal_results[metric] = dict()
        if ab not in single_plot[metric].keys():
            single_plot[metric][ab] = dict()
            removal_results[metric][ab] = dict()
        ax.bar(
                0.,
                0.,
                width=0.12,
                color=colors[metric],
                label=metric,
               )
    #other_abils = [(k, [k]) for k in abilities if k!=ab]
    other_abils = [(k, [k]) for k in abilities if k!=ab and int(k[-1])!=int(ab[1])-1]
    ### removing other abilities
    #remov_abils = [k if int(k[-1])<int(ab[-1])]
    remov_abils = [k for k in abilities if int(k[-1])==int(ab[1])-1]
    if len(remov_abils) == 0:
        ab_sims = [abs(data[ab][k_one]-data[ab][k_two]) if numpy.nan not in [data[ab][k_one], data[ab][k_two]] else numpy.nan for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
    else:
        if len(remov_abils) == 1:
            to_be_removed_data = [[w] for w in data[remov_abils[0]]]
        else:
            to_be_removed_data = [[i, j] for i, j in zip(data[remov_abils[0]], data[remov_abils[1]])]
        model = sklearn.linear_model.LinearRegression().fit(to_be_removed_data, data[ab])
        part_data = [real-pred for real, pred in zip(data[ab], model.predict(to_be_removed_data))]
        ab_sims = [abs(part_data[k_one]-part_data[k_two]) if numpy.nan not in [data[ab][k_one], data[ab][k_two]] else numpy.nan for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
    rand_ab_sims = random.sample(ab_sims, k=len(ab_sims))
    xs = {m[0] : _ for _, m in enumerate(models+other_abils)}
    for dim_type, dims in models+other_abils:
        ticks.append(dim_type)
        print(dims)
        curr_data = numpy.array([[data[d][k_i] for d in dims] for k_i in range(len(data[d]))])
        if len(remov_abils) == 0:
            pass
        elif len(remov_abils) == 1:
            to_be_removed_data = [[w] for w in data[remov_abils[0]]]
            model = sklearn.linear_model.LinearRegression().fit(to_be_removed_data, curr_data)
            part_data = [real-pred for real, pred in zip(curr_data, model.predict(to_be_removed_data))]
            curr_data = [r for r in part_data]
        ### z scoring
        #if dim_type!= 'age':
        #    #curr_data = [[(val-numpy.average(v))/numpy.std(v) for val in v] for v in curr_data]
        #    #curr_data = [[val-numpy.average(v) for val in v] for v in curr_data]
        ### l2 norm
        #if dim_type!= 'age':
        #    for c_i in range(len(curr_data)):
        #        norm = numpy.sqrt(sum([numpy.power(abs(v), 2) for v in curr_data[c_i]]))
        #        curr_data[c_i] = [numpy.log(1+(v/norm)) for v in curr_data[c_i]]
        print([ab, dim_type])
        print('\n')
        for metric in metrics:
            if dim_type not in single_plot[metric][ab].keys():
                single_plot[metric][ab][dim_type] = dict()
                removal_results[metric][ab][dim_type] = dict()
            if dim_type in ['age']+abilities:
                if metric in ['pearson_r', 'spearman_r']:
                    curr_sims = [abs(curr_data[k_one][0]-curr_data[k_two][0]) if numpy.nan not in [data[ab][k_one], data[ab][k_two]] else numpy.nan for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                else:
                    if metric == 'manhattan':
                        curr_sims = [scipy.spatial.distance.cityblock(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'cosine':
                        curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'euclidean':
                        curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            else:
                if metric == 'pearson_r':
                    curr_sims = [-scipy.stats.pearsonr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                elif metric == 'spearman_r':
                    curr_sims = [-scipy.stats.spearmanr(curr_data[k_one], curr_data[k_two], nan_policy='omit').statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                elif metric == 'manhattan':
                    curr_sims = [scipy.spatial.distance.cityblock(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                elif metric == 'cosine':
                    curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                elif metric == 'euclidean':
                    curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            for metric_two in [
                               'spearman_r',
                               #'pearson_r',
                               ]:
                if metric_two == 'pearson_r':
                    ### real
                    corr = scipy.stats.pearsonr(curr_sims, ab_sims).statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(10):
                        ### perm
                        rand_corr = scipy.stats.pearsonr(curr_sims, random.sample(ab_sims, k=len(ab_sims))).statistic
                        rand.append(rand_corr)
                        ### bootstrap
                        idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                        boot_corr = scipy.stats.pearsonr(
                                    [curr_sims[i] for i in idxs],
                                    [ab_sims[i] for i in idxs]).statistic
                        boot.append(boot_corr)
                elif metric_two == 'spearman_r':
                    ### real
                    corr = scipy.stats.spearmanr(curr_sims, ab_sims, nan_policy='omit').statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(montecarlo):
                        rand_corr = scipy.stats.spearmanr(curr_sims, random.sample(ab_sims, k=len(ab_sims)), nan_policy='omit').statistic
                        rand.append(rand_corr)
                        ### bootstrap
                        idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                        '''
                        idxs = random.choices(dims, k=7)
                        boot_data = [[data[d][k_i] for d in idxs] for k_i in range(len(data[d]))]
                        boot_sims = [-scipy.stats.spearmanr(boot_data[k_one], boot_data[k_two], nan_policy='omit').statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                        boot_corr = scipy.stats.spearmanr(
                                    boot_sims,
                                    ab_sims,
                                    nan_policy='omit').statistic
                        '''
                        idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                        boot_corr = scipy.stats.spearmanr(
                                    [curr_sims[i] for i in idxs],
                                    [ab_sims[i] for i in idxs],
                                    nan_policy='omit',
                                    ).statistic
                        boot.append(boot_corr)
                        boot.append(boot_corr)

                ### p-value
                p = round((sum([1 if val>corr else 0 for val in rand])+1)/(montecarlo+1), 6)
                ### adding to the single plot
                single_plot[metric][ab][dim_type]['rand'] = rand
                single_plot[metric][ab][dim_type]['p'] = p
                single_plot[metric][ab][dim_type]['boot'] = boot
                single_plot[metric][ab][dim_type]['sim'] = corr
                print(
                       [
                       metric,
                       metric_two,
                       round(corr, 6),
                       'p={}'.format(p),
                       'boot avg={}'.format(round(numpy.average(boot), 6)),
                       ]
                       )
                if p<0.05:
                    ax.scatter(
                            xs[dim_type]+corrections[metric],
                            0.02,
                            s=200,
                            marker='*',
                            #color=colors[metric],
                            color='white',
                            zorder=3.,
                            edgecolors='white'
                           )
                ax.bar(
                        xs[dim_type]+corrections[metric],
                        corr,
                        width=0.12,
                        color=colors[metric],
                       )
                plot_boot = numpy.array(boot)[random.sample(range(len(boot)), k=int(len(boot)/10))]
                ax.scatter(
                        [xs[dim_type]+corrections[metric]+(random.choice(range(-35,35))*0.001) for _ in plot_boot],
                        plot_boot,
                        alpha=0.25,
                        edgecolors=colors[metric],
                        color='white',
                        zorder=2.5,
                        )

        print('\n')
    pyplot.xticks(
                  ticks=range(len(ticks)),
                  labels=[x.replace(' ', '\n') for x in ticks],
                  fontsize=23,
                  )
    pyplot.legend(
                 fontsize=23,
                 ncols=4,
                 loc=9,
                 )
    pyplot.title(
                 '{}'.format(ab.replace('_', ' - ')),
                 fontsize=25,
                 fontweight='bold',
                 )
    pyplot.savefig('{}.jpg'.format(ab))
    pyplot.clf()
    pyplot.close()
    ### one-by-one removal
    for dim_type, dims in models:
        print(dim_type)
        if len(dims) < 2:
            continue
        #possibilities = [] +\
        #                [[d] for d in dims] +\
        #                list(itertools.product(dims, repeat=2)) +\
        #                list(itertools.product(dims, repeat=3))
        possibilities = list()
        #for _ in range(1, len(dims)):
        #for _ in range(1, len(dims)):
        #    possibilities.extend(list(itertools.product(dims, repeat=_)))
        #possibilities = possibilities[:min(1000, len(possibilities))]
        possibilities = [random.choices(dims, k=7) for _ in range(100)]
        for poss in tqdm(possibilities):
            ### remaining predictors
            curr_data = [[data[d][k_i] for d in poss] for k_i in range(len(data[d]))]
            for metric in metrics:
                if len(poss) == 1:
                    curr_sims = [abs(curr_data[k_one][0]-curr_data[k_two][0]) if numpy.nan not in [data[ab][k_one], data[ab][k_two]] else numpy.nan for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                else:
                    if metric == 'pearson_r':
                        curr_sims = [1-scipy.stats.pearsonr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'spearman_r':
                        curr_sims = [1-scipy.stats.spearmanr(curr_data[k_one], curr_data[k_two], nan_policy='omit').statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'manhattan':
                        curr_sims = [scipy.spatial.distance.cityblock(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'cosine':
                        curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                    elif metric == 'euclidean':
                        curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                for metric_two in [
                                   'spearman_r',
                                   #'pearson_r',
                                   ]:
                    if metric_two == 'pearson_r':
                        ### real
                        corr = scipy.stats.pearsonr(curr_sims, ab_sims).statistic
                        ### rand
                        rand = scipy.stats.pearsonr(curr_sims, rand_ab_sims).statistic
                    elif metric_two == 'spearman_r':
                        ### real
                        corr = scipy.stats.spearmanr(curr_sims, ab_sims, nan_policy='omit').statistic
                        ### rand
                        rand = scipy.stats.spearmanr(curr_sims, rand_ab_sims, nan_policy='omit').statistic

                    ### p-value
                    #p = round((sum([1 if val>corr else 0 for val in rand])+1)/(montecarlo+1), 6)
                    ### adding to the single plot
                    #for p in poss:
                    for p in poss:
                        if p not in removal_results[metric][ab][dim_type].keys():
                            removal_results[metric][ab][dim_type][p] = dict()
                            removal_results[metric][ab][dim_type][p]['sim'] = list()
                            removal_results[metric][ab][dim_type][p]['rand'] = list()
                        #removal_results[metric][ab][dim_type][remov_d]['rand'] = rand
                        #removal_results[metric][ab][dim_type][remov_d]['p'] = p
                        #removal_results[metric][ab][dim_type][remov_d]['boot'] = boot
                        removal_results[metric][ab][dim_type][p]['sim'].append(corr)
                        removal_results[metric][ab][dim_type][p]['rand'].append(rand)
                        #original_res = single_plot[metric][ab][dim_type]['sim']
                        #removal_results[metric][ab][dim_type][remov_d]['diff'] = original_res - corr
                        #if original_res - corr < 0.:
                        #    print([remov_d, original_res-corr])

with open('bootstrap.txt', 'w') as o:
    for ab, ab_data in removal_results['spearman_r'].items():
        for dim, dim_data in ab_data.items():
            if len(dim_data.keys()) == 0:
                continue
            for remov, remov_data in dim_data.items():
                o.write('{}\t{}\t{}\tsim\t'.format(ab, dim, remov))
                for val in dim_data[remov]['sim']:
                    o.write('{}\t'.format(val))
                o.write('\n')
                o.write('{}\t{}\t{}\trand\t'.format(ab, dim, remov))
                for val in dim_data[remov]['rand']:
                    o.write('{}\t'.format(val))
                o.write('\n')

### p correction
for metric, m_data in single_plot.items():
    ps = dict()
    for ab, ab_data in m_data.items():
        for dim, dim_data in ab_data.items():
            ps[(metric, ab, dim)] = dim_data['p']
    srt = sorted(ps.keys())
    corr_ps = mne.stats.fdr_correction([ps[k] for k in srt])[1]
    for k, p in zip(srt, corr_ps):
        single_plot[k[0]][k[1]][k[2]]['corr_p'] = p

#plot_colors = set([v for vec in single_plot.values() for val in vec.values() for v in val.keys()])

colors_tuple = [
                ('T1', 'yellowgreen'),
                ('T2', 'pink'),
                ('T3', 'mediumaquamarine'),
                ('activations T1', 'khaki'),
                ('activations T2', 'darkkhaki'),
                ('connectivity T1', 'paleturquoise'),
                ('connectivity T2', 'lightskyblue'),
                ('lesions', 'lightsalmon'),
                #('age', 'gainsboro'),
                ]
colors = {k : v for k, v in colors_tuple}

for metric, m_data in single_plot.items():
    fig, ax = pyplot.subplots(
                              constrained_layout=True,
                              figsize=(20, 10),
                              )
    for k, v in colors_tuple:
        ax.bar(0, 0, label=k, color=v)
    xs = {a : _ for _, a in enumerate(m_data.keys())}
    ax.hlines(
              y=[_*0.1 for _ in range(-1, 8)],
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
    for ab, a_data in m_data.items():
        n_xs = [v[0] for v in colors_tuple if v[0] in a_data.keys()]
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
            ax.bar(
                   xs[ab]+corrections[dim],
                   a_data[dim]['sim'],
                   width=0.1,
                   color=colors[dim]
                   )
            plot_boot = numpy.array(a_data[dim]['boot'])[random.sample(range(len(boot)), k=int(len(boot)/10))]
            ax.scatter(
                    [xs[ab]+corrections[dim]+(random.choice(range(-30,30))*0.001) for _ in plot_boot],
                    plot_boot,
                    alpha=0.25,
                    edgecolors=colors[dim],
                    color='white',
                    zorder=2.5,
                    )
            if a_data[dim]['corr_p'] < 0.05:
                ax.scatter(
                   xs[ab]+corrections[dim],
                   0.02,
                   s=200,
                   marker = '*',
                    color='white',
                    zorder=3.,
                    edgecolors='black'
                   )
    pyplot.xticks(
                  ticks = range(len(xs.keys())),
                  labels = [x.replace('_', '-') for x in m_data.keys()],
                  fontsize=23,
                  )
    pyplot.legend(
                 fontsize=18,
                 ncols=9,
                 loc=9,
                 )
    ax.set_ylim(top=0.82, bottom=-.1)

    pyplot.title(
                 'RSA pattern similarities',
                 fontsize=25,
                 fontweight='bold',
                 )
    pyplot.ylabel(ylabel='Spearman rho', fontsize=20)
    pyplot.yticks(fontsize=15)
    pyplot.savefig('{}.jpg'.format(metric))
    pyplot.clf()
    pyplot.close()
    with open('{}.results'.format(metric), 'w') as o:
        o.write('target\tvariables\traw_p\tcorrected_p\n')
        for ab, a_data in m_data.items():
            for dim, dim_data in a_data.items():
                o.write('{}\t{}\t{}\t{}\n'.format(ab, dim, dim_data['p'], dim_data['corr_p']))

'''
print('### RSA Encoding ###\n')

### RSA encoding
for dim_type, dims in [
                        ('lesions', lesions),
                        ('activations T1', activations_t1),
                        ('activations T2', activations_t2),
                        ('connectivity T1', conn_t1),
                        ('connectivity T2', conn_t2),
                        ]:
    curr_data = [[data[d][k_i] for d in dims] for k_i in range(len(data[d]))]
    for ab in abilities:
        loso = list()
        ### 80/20 splits
        #splits = list(range(0, len(data[ab]), 6))
        splits = [random.sample(range(len(data[ab])), k=6) for _ in range(100)]
        results = dict()
        for leave_outs in splits:
            #print(leave_outs)
            for leave_out in leave_outs:
                if leave_out not in results.keys():
                    results[leave_out] = list()
                #curr_sims = [1+scipy.stats.pearsonr(curr_data[leave_out], curr_data[k_one]).statistic for k_one in range(len(data[ab])) if k_one not in leave_outs]
                curr_sims = [1+scipy.stats.spearmanr(curr_data[leave_out], curr_data[k_one]).statistic for k_one in range(len(data[ab])) if k_one not in leave_outs]
                #curr_sims = [1-scipy.spatial.distance.cosine(curr_data[leave_out], curr_data[k_one]) for k_one in range(len(data[ab])) if k_one not in leave_outs]
                #curr_sims = [scipy.spatial.distance.euclidean(curr_data[leave_out], curr_data[k_one]) for k_one in range(len(data[ab])) if k_one not in leave_outs]
                #print(curr_sims)
                curr_ab = [data[ab][k_one] for k_one in range(len(data[ab])) if k_one not in leave_outs]
                #pred = numpy.array(sum([sim*abil for sim, abil in zip(curr_sims, curr_ab)]))/sum(curr_sims)
                pred = numpy.sum([sim*abil for sim, abil in zip(curr_sims, curr_ab)]) / sum(curr_sims)
                #loso.append(pred)
                results[leave_out].append(pred)
        loso = [numpy.average(results[_]) for _ in range(len(data[ab]))]
        corr = scipy.stats.spearmanr(loso, data[ab]).statistic
        #corr = scipy.stats.spearmanr(loso, data[ab]).statistic
        pred_errors = numpy.power(numpy.array(data[ab]) - numpy.array(loso), 2)
        pred_avg = numpy.power(data[ab] - numpy.average(data[ab]), 2)
        r_squared = 1 - (sum(pred_errors) / sum(pred_avg))
        #r2 = sklearn.metrics.r2_score(data[ab], loso)
        print([ab, dim_type])
        print(corr)
        print(r_squared)
        #print(r2)
        print('\n')
'''
