import numpy
import mne
import os
import re
import scipy

from scipy import stats

tmax_p_vals = dict()
p_vals = dict()
### reading originals
real_f = os.path.join('data', 'real')
assert os.path.exists(real_f)
perm_f = os.path.join('data', 'permutation')
assert os.path.exists(perm_f)
for f in os.listdir(real_f):
    if 'tsv' not in f:
        continue
    case = f.split('.')[0]
    real = dict()
    perm = dict()
    with open(os.path.join(real_f, f)) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')[4:]
            for pred in line:
                pred = [w.strip() for w in re.sub(r"\(|\)|'", '', pred).split(',')]
                real[pred[0]] = float(pred[1])
    ### reading perm
    for _ in range(1000):
        with open(os.path.join(perm_f, f.replace('.tsv', '_{}.tsv'.format(_)))) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')[4:]
                for pred in line:
                    pred = [w.strip() for w in re.sub(r"\(|\)|'", '', pred).split(',')]
                    try:
                        perm[pred[0]].append(float(pred[1]))
                    except KeyError:
                        perm[pred[0]] = [float(pred[1])]
    ### t-max
    t_max = list()
    for v in perm.values():
        assert len(v) == 1000
    for _ in range(1000):
        current_perm = [vals[_] for vals in perm.values()]
        abs_current_perm = [abs(vals[_]) for vals in perm.values()]
        t_max.append(current_perm[abs_current_perm.index(max(abs_current_perm))])
    ###
    ### uncorrected p-value
    for var, real_beta in real.items():
        perm_betas = perm[var]
        assert len(perm_betas) == 1000
        #sort_lst = [v[0] for v in sorted(perm_betas + [('real', real_beta)], key=lambda item : item[1])]
        #p_val = 1 - (sort_lst.index('real')/len(sort_lst))
        ### symmetric thingy - let's avoid that...
        #left_p = (len([v for v in perm_betas if v>= max(real_beta, -real_beta)])+1)/(len(perm_betas)+1)
        #right_p = (len([v for v in perm_betas if v<= min(real_beta, -real_beta)])+1)/(len(perm_betas)+1)
        #print([case, var, left_p, right_p, p])
        #p_vals[(case, var)] = left_p + right_p
        ### Ernst 2004 method
        p = (sum([1 for val in perm_betas if abs(val-numpy.average(perm_betas))>=abs(real_beta-numpy.average(perm_betas))])+1)/(len(perm_betas)+1)
        p_vals[(case, var)] = p
        ### t-max
        tmax_p = (sum([1 for val in t_max if abs(val-numpy.average(t_max))>=abs(real_beta-numpy.average(t_max))])+1)/(len(t_max)+1)
        tmax_p_vals[(case, var)] = tmax_p
        print([case, var, real_beta, p, tmax_p])

keyz = list(p_vals.keys())
fdr_corr_p_vals = mne.stats.fdr_correction([p_vals[k] for k in keyz])[1]

results = 'results'
os.makedirs(results, exist_ok=True)
### writing to file
with open(os.path.join(results, 'p-vals.tsv'), 'w') as o:
    o.write('target\tpredictor\traw_p\tfdr_p\tt-max_p\n')
    for k, corr_p in zip(keyz, fdr_corr_p_vals):
        o.write('{}\t{}\t{}\t{}\t{}\n'.format(k[0], k[1], p_vals[k], corr_p, tmax_p_vals[k]))
