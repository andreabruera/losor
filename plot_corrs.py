import itertools
import matplotlib
import mne
import numpy
import os
import random
import re
import scipy
import sklearn

from matplotlib import font_manager, pyplot
from scipy import spatial
from sklearn import linear_model, metrics
from tqdm import tqdm

# Using Helvetica as a font
font_folder = '../fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

with open('zz.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [w for w in line]
            raw_data = dict()
            corr_header = list()
            for var_i, var in enumerate(header):
                var = re.sub('^l|^L_', 'left ', var)
                var = re.sub('^r|^R_', 'right ', var)
                var = re.sub('_r', '_right ', var)
                var = re.sub('_l', '_left ', var)
                corr_header.append(var)
                if var_i > 1:
                    raw_data[var] = list()
            #raw_data = {h : list() for h in header[2:]}
            continue
        for d in raw_data.keys():
            val = float(line[corr_header.index(d)].replace(',', '.'))
            if len(d) == 2:
                val = 1.-float(val)
            raw_data[d].append(val)
data = {k : v for k, v in raw_data.items()}

abilities = ['T1', 'T2', 'T3']
improvements = ['T2_T1', 'T3_T2', 'T3_T1']

age = ['Age']
lesions = ['left DLPFC', 'left IFGorb', 'left PTL',
           #'left SMA',
           ]

activations_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T1' in k]
activations_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T2' in k]
conn_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T1' in k]
conn_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T2' in k]


models = [
        ('age', age),
        ('lesions', lesions),
        ('activations T1', activations_t1),
        ('activations T2', activations_t2),
        ('connectivity T1', conn_t1),
        ('connectivity T2', conn_t2),
        ]
out = 'corrs'

for mode in [
             'raw',
             #'cv-confound_lesions',
             ]:
    for ab in abilities+improvements:
        curr_out = os.path.join(out, mode, ab)
        os.makedirs(curr_out, exist_ok=True)
        for dim_type, dims in models:
            for dim in dims:
                fig, ax = pyplot.subplots(
                                         constrained_layout=True,
                                         figsize=(20, 20),
                                         )
                ys = data[ab]
                if mode != 'raw':
                    all_preds = [[data[d][i] for d in lesions] for i in range(len(data['T1']))]
                    all_ys = {_ : list() for _ in range(len(ys))}
                    for _ in range(100):
                        train_idxs = random.sample(range(len(ys)), k=int(len(ys)/2))
                        test_idxs = [i for i in range(len(ys)) if i not in train_idxs]
                        model = sklearn.linear_model.LinearRegression()
                        train_input = [all_preds[i] for i in train_idxs]
                        train_target = [ys[i] for i in train_idxs]
                        #print(train_input.shape)
                        #print(train_target.shape)
                        model.fit(train_input, train_target)
                        test_input = [all_preds[i] for i in test_idxs]
                        pred = model.predict(test_input)
                        real = [ys[i] for i in test_idxs]
                        data_residual = [r-p for r, p in zip(real, pred)]
                        for r, _ in zip(data_residual, test_idxs):
                            all_ys[_].append(r)
                    ys = [numpy.average(all_ys[k]) for k in sorted(all_ys.keys())]

                xs = data[dim]
                '''
                if len(ab) > 2:
                    if 'T1' in dim or 'T2' in dim:
                        ### subtracting...
                        t_pres = 'T{}'.format(ab[1])
                        xs_pres = data[dim[:-2]+t_pres]
                        t_past = 'T{}'.format(ab[4])
                        xs_past = data[dim[:-2]+t_past]
                        xs = [pres-past for pres, past in zip(xs_pres, xs_past)]
                '''

                model = sklearn.linear_model.LinearRegression().fit(
                                                        [[x] for x in xs],
                                                        [[y] for y in ys],
                                                        )
                ax.scatter(
                           xs,
                           ys,
                           s=300,
                           color='silver',
                           edgecolors='black',
                           )
                ax.plot(
                        xs,
                        model.predict(
                                      [[x] for x in xs],
                                      ),
                        color='b',
                        linewidth=7,
                        )
                ax.set_ylim(bottom=-0.05, top=1.05)
                ax.set_xlim(left=min(xs)-(numpy.std(xs)/4), right=max(xs)+(numpy.std(xs)/4))
                corr = scipy.stats.spearmanr(xs, ys).statistic
                ax.text(
                        x=min(xs),
                        y=0.1,
                        s='rho={}'.format(round(corr, 4)),
                        fontsize=50,
                        fontweight='bold',
                        )
                pyplot.xticks(fontsize=30)
                pyplot.xlabel(
                              xlabel=dim.replace('_', ' '),
                              fontsize=50,
                              fontweight='bold',
                              )
                pyplot.yticks(fontsize=30)
                pyplot.ylabel(
                              ylabel=ab.replace('_', '-'),
                              fontsize=50,
                              fontweight='bold',
                              )
                pyplot.savefig(os.path.join(curr_out, '{}.jpg'.format(dim)))
                pyplot.clf()
                pyplot.close()
