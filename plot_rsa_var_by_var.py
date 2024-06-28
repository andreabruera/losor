import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import font_manager, pyplot
from scipy import stats

# Using Helvetica as a font
font_folder = 'fonts/'
font_dirs = [font_folder, ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for p in font_files:
    font_manager.fontManager.addfont(p)
matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

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
            #print(vals)
        results['values'].append(vals)
n_items = set([len(v) for v in results.values()])
assert len(n_items) == 1
n_items = list(n_items)[0]

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
                    if float(line[-1]) < 0.1:
                        cases.append((line[0], line[1]))
            ### creating the folder
            out = os.path.join('rsa_zz', confound_method, confound_variable, 'individual_variables')
            os.makedirs(out, exist_ok=True)
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
                for k, v in vals.items():
                    for ind_k in k:
                        if ind_k not in current_data[target][predictor].keys():
                            current_data[target][predictor][ind_k] = [v]
                        else:
                            current_data[target][predictor][ind_k].append(v)
            ### computing p-values, and correcting them
            for c_targ, c_targ_data in current_data.items():
                for c_pred, c_pred_data in c_targ_data.items():
                    ind_preds = list(c_pred_data.keys())
                    assert len(ind_preds) > 2
