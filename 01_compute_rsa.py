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
        confound_pred_only = False
        if confound_variable == 'T-minus-one':
            remov_abils = [k for k in labels['abilities'] if int(k[-1])==int(target_name[1])-1]
            assert len(remov_abils) in [0, 1]
            if len(remov_abils) == 1:
                if remov_abils[0] == target_name:
                    remov_abils = list()
        if confound_variable == 'lesions':
            ### when removing lesions, only in the target
            if predictor_name == 'lesions':
                remov_abils = list()
            else:
                remov_abils = labels['lesions']
                assert len(remov_abils) == 3
        if confound_variable == 'lesions_n_aphasia':
            ### when removing lesions+T1, not removing it from either
            if predictor_name in ['lesions', 'T1'] and target_name == 'T1':
                remov_abils = list()
            elif predictor_name in ['lesions', 'T1'] and target_name != 'T1':
                remov_abils = labels['lesions'] + labels['T1']
                assert len(remov_abils) == 4
                confound_pred_only = True
            else:
                remov_abils = labels['lesions'] + labels['T1']
                assert len(remov_abils) == 4
    confound_data = numpy.array([[full_data[d][i] for d in remov_abils] for i in range(subjects)])
    ### load predictor data
    ### if difference, do the difference
    digits = len(re.sub('\D', '', predictor_name))
    if digits < 2:
        raw_predictor_data = numpy.array([[full_data[d][i] for d in labels[predictor_name]] for i in range(subjects)])
        alternatives = labels[predictor_name]
        pres_labels = labels[predictor_name]
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
    cv_correlations = {l : list() for l in pres_labels}
    for train_subjects, test_subjects in folds:
        ### remove confounds from predictor and target (partial correlation)
        predictor_data = remove_confound(
                                          confound_data,
                                          raw_predictor_data,
                                          train_subjects,
                                          test_subjects,
                                          subjects,
                                          )
        target_data = remove_confound(
                                      confound_data,
                                      raw_target_data,
                                      train_subjects,
                                      test_subjects,
                                      subjects,
                                      )
        '''
        if len(raw_predictor_data.shape) == 1:
            predictor_data = raw_predictor_data.reshape(-1, 1)[test_subjects, :]
        else:
            predictor_data = raw_predictor_data[test_subjects, :]
        ### removing T1 from target defeats the purpose of looking at improvement
        if len(raw_target_data.shape) == 1:
            target_data = raw_target_data.reshape(-1, 1)[test_subjects, :]
        else:
            target_data = raw_target_data[test_subjects, :]
        '''
        ### doesn't make sense removing T1 from T1...
        #if confound_les_only:
        #    #target_data = raw_target_data.reshape(-1, 1)[test_subjects, :]
        #    confound_data = confound_data[:, :-1]
        '''
        if 'aphasia' in confound_variable:
            target_data = remove_confound(
                                      confound_data[:, :-1],
                                      raw_target_data,
                                      train_subjects,
                                      test_subjects,
                                      subjects,
                                      )
        else:
            target_data = remove_confound(
                                      confound_data,
                                      raw_target_data,
                                      train_subjects,
                                      test_subjects,
                                      subjects,
                                      )
        '''
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
        ### simple correlation
        for l_i, l in enumerate(pres_labels):
            simple_corr = scipy.stats.spearmanr(predictor_data[:, l_i], target_data).statistic
            cv_correlations[l].append(simple_corr)

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
                                         #rand_target_sims,
                                         predictor_sims,
                                         rand_target_sims
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
            #fixed = len(alternatives)
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
                                      #k=random.choice(range(1, fixed))
                                      k=random.choice(range(1, len(alternatives)))
                                      #k=fixed
                                      #k=len(alternatives)
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

    return (key, results, cv_correlations)

### start
old_impros = ['T2_T1', 'T3_T2', 'T3_T1']

with open(os.path.join('dataset', 'data_41_corrected_improvement.tsv')) as i:
    for l_i, l in enumerate(i):
        line = l.split('\t')
        line[-1] = line[-1].strip()
        if l_i == 0:
            header = [w for w in line]
            raw_data = {h : list() for h in header[2:] if h!='L_SMA' and 'adjusted' not in h and h not in old_impros}
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
                'T2-T1',
                'T3-T2',
                'T3-T1',
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
                    #'lesions',
                    'lesions_n_aphasia',
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
                    key, results, cv_correlations = run_rsa(
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
                                 'rsa_results',
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
                    for var, var_corrs in cv_correlations.items():
                        with open(os.path.join(
                                   out,
                                  '{}_{}_{}_{}_{}_cv-correlations.tsv'.format(
                                       var,
                                       target_out_name,
                                       metric,
                                       confound_method,
                                       confound_variable)
                                   ), 'w') as o:
                            o.write('variable_name\t')
                            o.write('target_name\t')
                            o.write('metric\t')
                            o.write('confound_variable\t')
                            o.write('confound_method\t')
                            o.write('avg_corr\t')
                            o.write('raw_two_sided_p\t')
                            o.write('individual_corrs\n')
                            o.write('{}\t{}\t{}\t{}\t{}\t'.format(
                                             var,
                                             target_out_name,
                                             metric,
                                             confound_variable,
                                             confound_method,
                                             )
                                    )
                            o.write('{}\t'.format(float(numpy.average(var_corrs))))
                            ### two sided p
                            one = sum([1 for v in var_corrs if v>0.])/len(var_corrs)
                            two = sum([1 for v in var_corrs if v<0.])/len(var_corrs)
                            p_val = min([one, two])*2
                            o.write('{}\t'.format(float(p_val)))
                            for r in var_corrs:
                                o.write('{}\t'.format(float(r)))
                            o.write('\n')
