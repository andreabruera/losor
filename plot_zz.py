import matplotlib
import numpy
import os
import random
import scipy
import sklearn

from matplotlib import pyplot
from scipy import spatial
from sklearn import metrics

with open('zz.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            header = [w for w in line]
            data = {h : list() for h in header[2:]}
            continue
        for d in data.keys():
            data[d].append(float(line[header.index(d)].replace(',', '.')))

abilities = ['T1', 'T2', 'T3']
improvements = ['T2_T1', 'T3_T2', 'T3_T1']
lesions = ['L_DLPFC', 'L_IFGorb', 'L_PTL', 'L_SMA']

activations_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T1' in k]
activations_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' not in k and 'T2' in k]
conn_t1 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T1' in k]
conn_t2 = [k for k in data.keys() if k not in abilities and k not in improvements and k not in lesions and '_to_' in k and 'T2' in k]

### classic RSA
print('### Classic RSA ###\n')
### bootstrap with 30 subjects
bootstrap_n = len([1 for _ in range(30) for __ in range(30) if __>_])

for ab in abilities:
    ab_sims = [abs(data[ab][k_one]-data[ab][k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
    for dim_type, dims in [
                            ('lesions', lesions),
                            ('activations T1', activations_t1),
                            ('activations T2', activations_t2),
                            ('connectivity T1', conn_t1),
                            ('connectivity T2', conn_t2),
                            ]:
        curr_data = [[data[d][k_i] for d in dims] for k_i in range(len(data[d]))]
        print([ab, dim_type])
        print('\n')
        for metric in ['pearson_r', 'spearman_r', 'cosine', 'euclidean']:
            if metric == 'pearson_r':
                curr_sims = [-scipy.stats.pearsonr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'spearman_r':
                curr_sims = [-scipy.stats.spearmanr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'cosine':
                curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'euclidean':
                curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            for metric_two in ['spearman_r', 'pearson_r']:
                if metric_two == 'pearson_r':
                    ### real
                    corr = scipy.stats.pearsonr(curr_sims, ab_sims).statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(1000):
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
                    corr = scipy.stats.spearmanr(curr_sims, ab_sims).statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(1000):
                        rand_corr = scipy.stats.spearmanr(curr_sims, random.sample(ab_sims, k=len(ab_sims))).statistic
                        rand.append(rand_corr)
                        ### bootstrap
                        idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                        boot_corr = scipy.stats.spearmanr(
                                    [curr_sims[i] for i in idxs],
                                    [ab_sims[i] for i in idxs]).statistic
                        boot.append(boot_corr)
                ### p-value
                p = round((sum([1 if val>corr else 0 for val in rand])+1)/(1000+1), 6)
                print(
                       [
                       metric,
                       metric_two,
                       round(corr, 6),
                       'p={}'.format(p),
                       'boot avg={}'.format(round(numpy.average(boot), 6)),
                       ]
                       )
        print('\n')

for ab in improvements:
    ab_sims = [abs(data[ab][k_one]-data[ab][k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
    for dim_type, dims in [
                            ('lesions', lesions),
                            ('activations T1', activations_t1),
                            ('activations T2', activations_t2),
                            ('connectivity T1', conn_t1),
                            ('connectivity T2', conn_t2),
                            ]:
        curr_data = [[data[d][k_i] for d in dims] for k_i in range(len(data[d]))]
        print([ab, dim_type])
        print('\n')
        for metric in ['pearson_r', 'spearman_r', 'cosine', 'euclidean']:
            if metric == 'pearson_r':
                curr_sims = [-scipy.stats.pearsonr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'spearman_r':
                curr_sims = [-scipy.stats.spearmanr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'cosine':
                curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            elif metric == 'euclidean':
                curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
            for metric_two in ['spearman_r', 'pearson_r']:
                if metric_two == 'pearson_r':
                    ### real
                    corr = scipy.stats.pearsonr(curr_sims, ab_sims).statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(1000):
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
                    corr = scipy.stats.spearmanr(curr_sims, ab_sims).statistic
                    ### perm
                    rand = list()
                    ### bootstrap
                    boot = list()
                    for _ in range(1000):
                        rand_corr = scipy.stats.spearmanr(curr_sims, random.sample(ab_sims, k=len(ab_sims))).statistic
                        rand.append(rand_corr)
                        ### bootstrap
                        idxs = random.sample(range(len(curr_sims)), k=bootstrap_n)
                        boot_corr = scipy.stats.spearmanr(
                                    [curr_sims[i] for i in idxs],
                                    [ab_sims[i] for i in idxs]).statistic
                        boot.append(boot_corr)
                ### p-value
                p = round((sum([1 if val>corr else 0 for val in rand])+1)/(1000+1), 6)
                print(
                       [
                       metric,
                       metric_two,
                       round(corr, 6),
                       'p={}'.format(p),
                       'boot avg={}'.format(round(numpy.average(boot), 6)),
                       ]
                       )
        print('\n')
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
        splits = list(range(0, len(data[ab]), 6))
        for start in splits:
            if start != splits[-1]:
                leave_outs = [start+_ for _ in range(6)]
            else:
                leave_outs = list(set([min(start+_, len(data[ab])-1) for _ in range(6)]))
            #print(leave_outs)
            for leave_out in leave_outs:
                curr_sims = [1 + scipy.stats.pearsonr(curr_data[leave_out], curr_data[k_one]).statistic for k_one in range(len(data[ab])) if k_one not in leave_outs]
                curr_ab = [data[ab][k_one] for k_one in range(len(data[ab])) if k_one not in leave_outs]
                #pred = numpy.array(sum([sim*abil for sim, abil in zip(curr_sims, curr_ab)]))/sum(curr_sims)
                pred = numpy.sum([sim*abil for sim, abil in zip(curr_sims, curr_ab)]) / sum(curr_sims)
                loso.append(pred)
                #ab_sims = [abs(data[ab][leave_out]-data[ab][k_one]) for k_one in range(len(data[ab])) if leave_out!=k_one]
                #ab_sims = [abs(data[ab][k_one]-data[ab][k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                #curr_sims = [scipy.spatial.distance.cosine(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                #curr_sims = [1-scipy.stats.pearsonr(curr_data[k_one], curr_data[k_two]).statistic for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
                #curr_sims = [scipy.spatial.distance.euclidean(curr_data[k_one], curr_data[k_two]) for k_one in range(len(data[ab])) for k_two in range(len(data[ab])) if k_two>k_one]
        corr = scipy.stats.pearsonr(loso, data[ab]).statistic
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
