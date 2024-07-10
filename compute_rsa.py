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


def remove_confound(
                    confound,
                    data,
                    train_idxs,
                    test_idxs,
                    subjects,
                    ):
    if len(data.shape) == 1:
        #print(data)
        #print(data.shape)
        data = data.reshape(-1, 1)
        #print(data.shape)
    if confound.shape == (subjects, 0):
        ### nothing to do here...
        data_residual = data[test_idxs, :]
    else:
        if len(confound.shape) == 1:
            #print(confound.shape)
            confound = confound.reshape(-1, 1)
            #print(confound.shape)
        assert len(confound) == len(data)
        assert len(train_idxs) > 0
        confound_shape = set([v.shape for v in confound])
        assert len(confound_shape) == 1
        ### removing variance associated with confound
        model = sklearn.linear_model.LinearRegression()
        #print(confound.shape)
        #print(data.shape)
        #print(train_idxs)
        train_input = confound[train_idxs, :]
        train_target = data[train_idxs, :]
        #print(train_input.shape)
        #print(train_target.shape)
        model.fit(train_input, train_target)
        test_input = [confound[i] for i in test_idxs]
        pred = model.predict(test_input)
        real = [data[i] for i in test_idxs]
        data_residual = numpy.array([r-p for r, p in zip(real, pred)])

    return data_residual

def first_level_rsa(
             data,
             metric='spearman_r',
             direction='similarity',
             randomize=False
             ):
    ### checks
    assert metric in [
           'pearson_r',
           'spearman_r',
           'cosine',
           'euclidean',
           'manhattan',
           ]
    assert direction in ['similarity', 'dissimilarity']
    data_size = set([v.shape for v in data])
    assert len(data_size) == 1
    data_size = list(data_size)[0]
    #print(data_size)
    ### forcing absolute distance for 1-dimensional vectors
    if data_size == ():
        data = data.reshape(-1, 1)
        metric = 'euclidean'
    if data_size == (1,):
        metric = 'euclidean'
        #data = data.reshape(data.shape[0])
    if randomize == True:
        rng = numpy.random.default_rng()
        rng.shuffle(data, axis=0)
    ### correlation metrics
    if metric == 'pearson_r':
        ### correlation does not work with one number
        assert data_size[-1] != 1
        if direction == 'similarity':
            curr_sims = [scipy.stats.pearsonr(
                                      data[k_one],
                                      data[k_two],
                                      nan_policy='omit',
                                      ).statistic for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
        else:
            curr_sims = [-scipy.stats.pearsonr(
                                      data[k_one],
                                      data[k_two],
                                      nan_policy='omit',
                                      ).statistic for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
    elif metric == 'spearman_r':
        ### correlation does not work with one number
        assert data_size[-1] != 1
        if direction == 'similarity':
            curr_sims = [scipy.stats.spearmanr(
                                      data[k_one],
                                      data[k_two],
                                      nan_policy='omit',
                                      ).statistic for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
        else:
            curr_sims = [-scipy.stats.spearmanr(
                                      data[k_one],
                                      data[k_two],
                                      nan_policy='omit',
                                      ).statistic for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
    elif metric == 'manhattan':
        if direction == 'similarity':
            curr_sims = [1-scipy.spatial.distance.cityblock(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
        else:
            curr_sims = [scipy.spatial.distance.cityblock(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
    elif metric == 'cosine':
        if direction == 'similarity':
            curr_sims = [1-scipy.spatial.distance.cosine(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
        else:
            curr_sims = [scipy.spatial.distance.cosine(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
    elif metric == 'euclidean':
        if direction == 'similarity':
            curr_sims = [1-scipy.spatial.distance.euclidean(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
        else:
            curr_sims = [scipy.spatial.distance.euclidean(
                                      data[k_one],
                                      data[k_two]
                                      ) for \
                                            k_one in range(len(data)) for \
                                            k_two in range(len(data)) if \
                                            k_two>k_one\
                                            ]
    ### checking size
    curr_sims = numpy.array(curr_sims)
    assert curr_sims.shape == (int((len(data)*(len(data)-1))/2), )

    return curr_sims

def second_level_rsa(
             predictor_sims,
             target_sims,
             metric='spearman_r',
             ):
    #print(predictor_sims)
    #print(target_sims)
    ### checking all's fine and square
    assert predictor_sims.shape == target_sims.shape
    ### checking there are at least 7 items (cf. Nili et al. 2014)
    min_rsa = 7
    min_len = (min_rsa*(min_rsa-1))/2
    #print(min_len)
    assert predictor_sims.shape[0] >= min_len
    if metric == 'spearman_r':
        corr = scipy.stats.spearmanr(
                              predictor_sims,
                              target_sims,
                              nan_policy='omit',
                              ).statistic
    elif metric == 'pearson_r':
        corr = scipy.stats.pearsonr(
                              predictor_sims,
                              target_sims,
                              nan_policy='omit',
                              ).statistic
    #print(corr)
    return corr

def run_rsa(
            full_data,
            labels,
            predictor_name,
            target_name,
            metric,
            confound_variable,
            confound_method,
            ):
    ### setting up some variables...
    '''
    for seed in [
            5,
            12,
            38,
            86,
            115,
            173,
            200,
            301,
            315,
            317,
            353,
            363,
            440,
            ]:
    '''
    #for seed in range(0, 1000):
    #seed = 100
    #seed = 860
    #seed = 14
    seed = 40
    n_folds = 1000
    perms = 1000
    all_boots = 1000
    key = (predictor_name, target_name, metric, confound_variable, confound_method)
    #print(key)
    ### computing total number of subjects
    subjects = set([len(v) for v in full_data.values()])
    assert len(subjects) == 1
    subjects = list(subjects)[0]
    ### load confound data
    if confound_method == 'raw':
        remov_abils = list()
    else:
        if confound_variable == 'T-minus-one':
            remov_abils = [k for k in labels['abilities'] if int(k[-1])==int(target_name[1])-1]
            assert len(remov_abils) in [0, 1]
            if len(remov_abils) == 1:
                if remov_abils[0] == target_name:
                    remov_abils = list()
        if confound_variable == 'lesions':
            if predictor_name == 'lesions':
                remov_abils = list()
            else:
                remov_abils = labels['lesions']
                assert len(remov_abils) == 3
    confound_data = numpy.array([[full_data[d][i] for d in remov_abils] for i in range(subjects)])
    ### load predictor data
    ### if difference, do the difference
    digits = len(re.sub('\D', '', predictor_name))
    if digits < 2:
        raw_predictor_data = numpy.array([[full_data[d][i] for d in labels[predictor_name]] for i in range(subjects)])
        alternatives = labels[predictor_name]
    else:
        pres = predictor_name[-4]
        pres_labels = sorted(labels[predictor_name[:-5]+'T'+pres])
        past = predictor_name[-1]
        past_labels = sorted(labels[predictor_name[:-5]+'T'+past])
        #pres_predictor_data = numpy.array([[full_data[d][i] for d in labels[predictor_name]] for i in range(subjects)])
        pres_predictor_data = numpy.array([[full_data[d][i] for d in pres_labels] for i in range(subjects)])
        past_predictor_data = numpy.array([[full_data[d][i] for d in past_labels] for i in range(subjects)])
        raw_predictor_data = numpy.subtract(pres_predictor_data, past_predictor_data)
        assert raw_predictor_data.shape == pres_predictor_data.shape
        alternatives = [p[:-2]+predictor_name[-5:] for p in pres_labels]
    ### load target data
    raw_target_data = numpy.array([full_data[target_name][i] for i in range(subjects)])
    #print(raw_target_data)
    ### setting cv folds if required
    random.seed(seed)
    if confound_method in ['raw', 'partial']:
        train_size = subjects
        test_size = subjects
        folds = [(
                  list(range(subjects)),
                  list(range(subjects)),
                  )]
    else:
        train_size = int(subjects/2)
        test_size = subjects-train_size
        train_fold = [random.sample(range(subjects), k=train_size) for _ in range(n_folds)]
        folds = [(
                 t_fold,
                 [i for i in list(range(subjects)) if i not in t_fold],
                 ) for t_fold in train_fold]
    ### setting variables after dividing in folds
    boots = int(all_boots/len(folds))
    randoms = int(perms/len(folds))
    ### starting collecting results
    results = {
               'real' : list(),
               'bootstrap' : list(),
               'random' : list(),
               }
    #for train_subjects, test_subjects in tqdm(folds):
    if len(alternatives) >= 2:
        results['removal_bootstrap'] = list()
    for train_subjects, test_subjects in folds:
        ### remove confounds from predictor (partial correlation)
        predictor_data = remove_confound(
                                          confound_data,
                                          raw_predictor_data,
                                          train_subjects,
                                          test_subjects,
                                          subjects,
                                          )
        ### remove confounds from target (confound control)
        target_data = remove_confound(
                                          confound_data,
                                          raw_target_data,
                                          train_subjects,
                                          test_subjects,
                                          subjects,
                                          )
        #print(target_data)
        ### run first-level on predictor
        predictor_sims = first_level_rsa(
                                         predictor_data,
                                         metric,
                                         )
        ### run first-level on target
        #print(target_data)
        target_sims = first_level_rsa(
                                     target_data,
                                     metric,
                                     )
        ### run second level
        #print(target_sims)
        real_corr = second_level_rsa(
                                     predictor_sims,
                                     target_sims,
                                     )
        results['real'].append(real_corr)
        ### run bootstrap
        #print('running bootstraps...')
        #for _ in tqdm(range(boots)):
        #random.seed(seed)
        for _ in range(boots):
            ### preparing mask
            ### 2/3 of subjects
            rand_idxs = random.sample(
                            range(test_size),
                            k=int((test_size/3)*2),
                            )
            #print(rand_idxs)
            #print(rand_idxs)
            mask_idxs = [True if i in rand_idxs else False for i in range(test_size)]
            mask = [True if i==True and j==True else False for i_i, i in enumerate(mask_idxs) for i_j, j in enumerate(mask_idxs) if i_j>i_i]
            masked_preds = predictor_sims[mask]
            masked_targs = target_sims[mask]
            #print(mask)
            #print(masked_preds.shape)
            boot_corr = second_level_rsa(
                                         masked_preds,
                                         masked_targs,
                                         )
            results['bootstrap'].append(boot_corr)
        ### run random
        #print('running monte carlo randomizations...')
        #for _ in tqdm(range(randoms)):
        rng = numpy.random.default_rng(seed=seed)
        for _ in range(randoms):
            ### run first-level on randomized targets
            '''
            rand_target_sims = first_level_rsa(
                                         target_data,
                                         metric,
                                         randomize=True
                                         )
            rand_predictor_sims = first_level_rsa(
                                         predictor_data,
                                         metric,
                                         randomize=True
                                         )
            '''
            rand_target_sims = target_sims.copy()
            rng.shuffle(rand_target_sims, axis=0)
            ### run randomized second level
            rand_corr = second_level_rsa(
                                         #rand_predictor_sims,
                                         predictor_sims,
                                         rand_target_sims,
                                         )
            results['random'].append(rand_corr)
        ### running bootstraps with removal...
        #random.seed(seed)
        for _ in range(randoms):
            ### seven items with replacement
            if len(alternatives) < 2:
                continue
            #print(alternatives)
            #fixed = 5
            fixed = len(alternatives)
            #rand_idxs = random.sample(
            #rand_idxs = random.choices(
            #                          range(len(alternatives)),
            #                          #k=random.choice(range(1, alternatives))
            #                          k=random.choice(range(1, fixed))
            #                          #k=3,
            #                          #k=fixed,
            #                          )
            #rand_idxs = random.sample(
            rand_idxs = random.choices(
                                      range(len(alternatives)),
                                      #k=random.choice(range(1, alternatives))
                                      k=random.choice(range(1, fixed))
                                      )
            present_labels = list(set([alternatives[i] for i in rand_idxs]))
            #print(present_labels)
            rand_predictor_data = list()
            for d in predictor_data:
                lst = d.tolist()
                new_d = [lst[i] for i in rand_idxs]
                rand_predictor_data.append(new_d)
            boot_predictor_data = numpy.array(rand_predictor_data)
            ### run first-level on predictor
            boot_predictor_sims = first_level_rsa(
                                             boot_predictor_data,
                                             metric,
                                             )
            #print(mask)
            #print(masked_preds.shape)
            boot_corr = second_level_rsa(
                                         boot_predictor_sims,
                                         target_sims,
                                         )
            #print(boot_corr)
            results['removal_bootstrap'].append('+'.join(present_labels)+':{}'.format(boot_corr))
    if len(alternatives) >= 2:
        assert len(results['removal_bootstrap']) == perms
    assert len(results['bootstrap']) == all_boots
    assert len(results['random']) == perms
    ### computing and writing p-value
    avg = numpy.nanmean(results['real'])
    uncorr_p = (sum([1 for _ in results['random'] if _>avg])+1)/(len(results['random'])+1)
    results['raw_permutation_p'] = [uncorr_p]
    avg = numpy.nanmean(results['real'])
    avg_rand = numpy.nanmean(results['random'])
    print(seed)
    print([key, avg, avg_rand, results['raw_permutation_p']])

    return (key, results)

### start


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
            #print(val)
            if val == '':
                raw_data[d].append(numpy.nan)
            else:
                #if len(d) == 2:
                #    val = 1.-float(val)
                raw_data[d].append(float(val))
### z scoring
#data = {k : [(val-numpy.average(v))/numpy.std(v) for val in v] for k, v in raw_data.items()}
full_data = {k : v for k, v in raw_data.items() if len(v)>0}

abilities = [
             'T1',
             'T2',
             'T3',
             ]
improvements = [
                'T2_T1',
                'T3_T2',
                'T3_T1',
                ]

age = [
       'Age',
       ]
lesions = [
           'L_DLPFC',
           'L_IFGorb',
           'L_PTL',
           #'L_SMA',
           ]

### removing subjects if they lack sufficient damage
to_be_removed = dict()
for d in full_data.keys():
    if d in lesions:
        for v_i, v in enumerate(full_data[d]):
            if v_i not in to_be_removed.keys():
                to_be_removed[v_i] = v
            else:
                to_be_removed[v_i] += v
to_be_removed = [v_i for v_i, v in to_be_removed.items() if v == 0.]
print('removing subjects for lack of data: {}'.format(to_be_removed))
full_data = {k : [v[i] for i in range(len(v)) if i not in to_be_removed] for k, v in full_data.items()}

### setting up data names
labels = dict()
labels['lesions'] = lesions
labels['abilities'] = abilities
labels['improvements'] = improvements
labels['T1'] = ['T1']
labels['T2'] = ['T2']
labels['T3'] = ['T3']
labels['age'] = ['Age']
labels['activations T1'] = [k for k in full_data.keys() if \
                                     k not in abilities and \
                                     k not in improvements and \
                                     k not in lesions \
                                     and '_to_' not in k and \
                                     'T1' in k\
                                ]
labels['activations T2'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' not in k and \
                                          'T2' in k\
                                          ]
labels['activations T3'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' not in k and \
                                          'T3' in k\
                                          ]
labels['connectivity T1'] = [k for k in full_data.keys() if \
                                          k not in abilities and \
                                          k not in improvements and \
                                          k not in lesions and \
                                          '_to_' in k and \
                                          'T1' in k\
                                          ]
labels['connectivity T2'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                           '_to_' in k and \
                                           'T2' in k\
                                           ]
labels['connectivity T3'] = [k for k in full_data.keys() if \
                                           k not in abilities and \
                                           k not in improvements and \
                                           k not in lesions and \
                                           '_to_' in k and \
                                           'T3' in k\
                                           ]
### targets
targets = [v for v in abilities+improvements]
### predictors
predictors = [
              'age',
              'T1',
              'T2',
              'T3',
              #'abilities',
              'lesions',
              'activations T1',
              'activations T2',
              #'activations T3',
              #'activations T2-T1',
              #'activations T3-T2',
              #'activations T3-T1',
              'connectivity T1',
              'connectivity T2',
              #'connectivity T3',
              #'connectivity T2-T1',
              #'connectivity T3-T2',
              #'connectivity T3-T1',
              ]

### various setups...
### different metrics
metrics = [
           #'pearson_r',
           #'spearman_r',
           #'cosine',
           'euclidean',
           #'manhattan',
           ]
### various ways of dealing with confound variables
confound_methods = [
                    #'raw',
                    #'partial',
                    'cv-confound',
                    ]
confound_variables = [
                    #'T-minus-one',
                    'lesions',
                    #'none'
                    ]
all_results = dict()
for predictor_name in predictors:
    for target_name in targets:
        if target_name == predictor_name:
            continue
        #if target_name not in ['T2']:
        #    continue
        #if target_name not in ['T2', 'T3']:
        #    continue
        pred_digits = re.sub('\D', '', predictor_name)
        targ_digits = re.sub('\D', '', target_name)
        if len(pred_digits) == 2 and pred_digits!=targ_digits:
            continue
        for metric in metrics:
            for confound_variable in confound_variables:
                for confound_method in confound_methods:
                    if confound_method == 'raw' and confound_variable != 'none':
                        continue
                    if confound_method != 'raw' and confound_variable == 'none':
                        continue
                    key, results = run_rsa(
                                full_data,
                                labels,
                                predictor_name,
                                target_name,
                                metric,
                                confound_variable,
                                confound_method,
                                )
                    avg = numpy.nanmean(results['real'])
                    avg_rand = numpy.nanmean(results['random'])
                    print([key, avg, avg_rand, results['raw_permutation_p']])
                    target_out_name = target_name.replace('_', '-')
                    #all_results[key] = results
                    out = os.path.join(
                                 'rsa_results_zz',
                                 metric,
                                 confound_method,
                                 confound_variable,
                                 target_out_name,)
                    os.makedirs(out, exist_ok=True)
                    with open(os.path.join(
                                   out,
                                   '{}_{}_{}_{}_{}_results.tsv'.format(
                                       predictor_name,
                                       target_out_name,
                                       metric,
                                       confound_method,
                                       confound_variable)
                                   ), 'w') as o:
                        o.write('predictor_name\t')
                        o.write('target_name\t')
                        o.write('metric\t')
                        o.write('confound_variable\t')
                        o.write('confound_method\t')
                        o.write('measure\n')
                        for k, res in results.items():
                            o.write('{}\t{}\t{}\t{}\t{}\t'.format(
                                             predictor_name,
                                             target_out_name,
                                             metric,
                                             confound_variable,
                                             confound_method,
                                             )
                                    )
                            o.write('{}\t'.format(k))
                            for r in res:
                                o.write('{}\t'.format(r))
                            o.write('\n')

'''
for k, v in all_results.items():
    pred = k[0]
    targ = k[1]
    metric = k[2]
    confound_var = k[3]
    confound_meth = k[4]
    ### creating the folder
    out = os.path.join('rsa_results_zz', metric, confound_meth, confound_var)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, '{}_{}_{}_results.tsv'.format(metric, confound_meth, confound_var)), 'w') as o:
        o.write('predictor_name\t')
        o.write('target_name\t')
        o.write('metric\t')
        o.write('confound_variable\t')
        o.write('confound_method\t')
        o.write('measure\n')
        for key, res in v.items():
            o.write('{}\t{}\t{}\t{}\t{}\t'.format(
                                                 pred,
                                                 targ,
                                                 metric,
                                                 confound_var,
                                                 confound_meth
                                                 )
                    )
            o.write('{}\t'.format(key))
            for r in res:
                o.write('{}\t'.format(r))
            o.write('\n')

import pdb; pdb.set_trace()

### setting variables for metrics plot
metric_correc = {k : (v-1.)*0.15 for k, v in zip(metrics, range(len(metrics)))}
list_cols = ['teal', 'goldenrod', 'plum', 'gray', 'orange',]
metric_colors = {k : v for k, v in zip(metrics, colors)}

### classic RSA
print('### Classic RSA ###\n')
### bootstrap with 30 subjects
bootstrap_n = len([1 for _ in range(30) for __ in range(30) if __>_])

for mode in mode:
    print(mode)
    montecarlo = 1000

    removal_results = dict()
    single_plot = dict()
    out = os.path.join('zz_out', mode)
    os.makedirs(out, exist_ok=True)
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
        ### removing other abilities if required
        if 'raw' in model:
            basic_sims = [(
                          list(range(len(data[ab]))),
                          [abs(data[ab][k_one]-data[ab][k_two]) if \
                                  numpy.nan not in [data[ab][k_one], data[ab][k_two]]\
                                  else numpy.nan for k_one in \
                                  range(len(data[ab])) for k_two in \
                                  range(len(data[ab])) if k_two>k_one],
                          )]
            corr_sims = [(
                          list(range(len(data[ab]))),
                          [abs(data[ab][k_one]-data[ab][k_two]) if \
                                  numpy.nan not in [data[ab][k_one], data[ab][k_two]]\
                                  else numpy.nan for k_one in \
                                  range(len(data[ab])) for k_two in \
                                  range(len(data[ab])) if k_two>k_one],
                          )]
            rand_ab_sims = [(
                             corr_sims[0],
                             random.sample(corr_sims[1], k=len(corr_sims[1]))
                             )]
        else:
            print(remov_abils)
            if 'partial' in mode:
                test_subs = [list(range(len(data[ab])))]
            elif 'cv' in mode:
                test_subs = [sorted(random.sample(range(len(data[ab])), k=20)) for _ in range(100)]
            all_sims = list()
            for test in test_subs:
                if len(remov_abils) == 0:
                    it_corr_sims = [(
                          list(range(len(data[ab]))),
                          [abs(data[ab][k_one]-data[ab][k_two]) if \
                                  numpy.nan not in [data[ab][k_one], data[ab][k_two]]\
                                  else numpy.nan for k_one in \
                                  test for k_two in \
                                  test if k_two>k_one],
                          )]
                    it_rand_ab_sims = [(
                             corr_sims[0],
                             random.sample(corr_sims[1], k=len(corr_sims[1]))
                             )]
                    all_sims.append(it_corr_sims)
                else:
                    train = [i for i in range(len(data[ab])) if i not in test]
                    to_be_removed_data = [[data[r][i] for r in remov_abils] for i in train]
                    assert len(to_be_removed_data[0]) == len(remov_abils)
                    model = sklearn.linear_model.LinearRegression().fit(
                                                    to_be_removed_data,
                                                    [data[ab][i] for i in train],
                                                    )
                    to_be_predicted_data = [[data[r][i] for r in remov_abils] for i in test]
                    part_data = [real-pred for real, pred in zip(
                                   [data[ab][i] for i in test],
                                   model.predict(to_be_predicted_data)
                                   )
                                 ]
                    it_corr_sims = [(
                                  test,
                                  abs(part_data[k_one]-part_data[k_two]) \
                                   for k_one in range(len(part_data)) for \
                                   k_two in range(len(part_data)) if k_two>k_one,
                                 )]
                    it_rand_ab_sims = [(
                             corr_sims[0],
                             random.sample(corr_sims[1], k=len(corr_sims[1]))
                             )]
                    all_sims.append(it_corr_sims)
        xs = {m[0] : _ for _, m in enumerate(models+other_abils)}
        for dim_type, dims in models+other_abils:
            ticks.append(dim_type)
            print(dims)
            all_model = list()
            for test, corr_sims in all_sims:
                all_curr_data = numpy.array([[data[d][k_i] for d in dims] for k_i in test])
                if 'lesions' in mode and dim_type == 'lesions':
                    untouched_data = numpy.array([[data[ab][k_i] for d in dims] for k_i in test])
                    #ab_sims = [s for s in basic_sims]
                    pass
                elif 'T-minus-one' in mode and dim_type in remov_abils:
                    untouched_data = numpy.array([[data[ab][k_i] for d in dims] for k_i in test])
                    #ab_sims = [s for s in basic_sims]
                    pass
                else:
                    #ab_sims = [s for s in corr_sims]
                    if 'partial' in mode:
                        to_be_removed_data = [[w] for w in data[remov_abils[0]]]
                        model = sklearn.linear_model.LinearRegression().fit(to_be_removed_data, curr_data)
                        part_data = [real-pred for real, pred in zip(curr_data, model.predict(to_be_removed_data))]
                        all_curr_data = [r for r in part_data]
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
                results = list()
                for curr_data in all_curr_data:
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
                    for metric_two in [
                                       'spearman_r',
                                       #'pearson_r',
                                       ]:
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
                            results.append(corr)
                single_plot[metric][ab][dim_type]['sim'] = numpy.average(corr)
                single_plot[metric][ab][dim_type]['p'] = 'na'
                single_plot[metric][ab][dim_type]['corr_p'] = 'na'
'''
'''
                        ### perm
                        rand = list()
                        ### bootstrap
                        boot = list()
                        for _ in range(montecarlo):
                            rand_corr = scipy.stats.spearmanr(curr_sims, random.sample(ab_sims, k=len(ab_sims)), nan_policy='omit').statistic
                            rand.append(rand_corr)
                            ### bootstrap
                            idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                            #idxs = random.choices(dims, k=7)
                            #boot_data = [[data[d][k_i] for d in idxs] for k_i in range(len(data[d]))]
                            #boot_sims = [-scipy.stats.spearmanr(boot_data[k_one], boot_data[k_two], nan_policy='omit').statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                            #boot_corr = scipy.stats.spearmanr(
                            #            boot_sims,
                            #            ab_sims,
                            #            nan_policy='omit').statistic
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
'''
'''

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
        pyplot.savefig(os.path.join(
                                    out,
                                    '{}.jpg'.format(ab),
                                    ))
        pyplot.clf()
        pyplot.close()
'''
'''
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

    with open(os.path.join(out, 'bootstrap.txt'), 'w') as o:
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
'''
'''

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
'''
'''
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
'''
'''
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
        pyplot.savefig(os.path.join(out, '{}.jpg'.format(metric)))
        pyplot.clf()
        pyplot.close()
        with open(os.path.join(out, '{}.results'.format(metric)), 'w') as o:
            o.write('target\tvariables\traw_p\tcorrected_p\n')
            for ab, a_data in m_data.items():
                for dim, dim_data in a_data.items():
                    o.write('{}\t{}\t{}\t{}\n'.format(ab, dim, dim_data['p'], dim_data['corr_p']))

'''
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
