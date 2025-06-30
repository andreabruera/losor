import numpy
import os
import scipy

def read_correlations():
    ### checking what we need to keep
    folder = os.path.join(
                          'rsa_plots',
                          'cv_partial-corr',
                          'mixed',
                          'euclidean',
                          'individual_variables',
                          )
    targets = set()
    variables = set()
    families = dict()
    for target in os.listdir(folder):
        targets.add(target)
        #for f in os.listdir(os.path.join(folder, target)):
        for root, direc, fz in os.walk(os.path.join(folder, target)):
            if 'lesions' in root:
                continue
            for f in fz:
                if 'tsv' not in f:
                    continue
                #print(root)
                if int(root.split('/')[-1][-1]) > int(root.split('/')[-2][1]):
                    print(f)
                    continue
                family = root.split('/')[-1]
                #if family not in families.keys():
                #    families[family] = set()
                with open(os.path.join(root, f)) as i:
                    for l_i, l in enumerate(i):
                        if l_i == 0:
                            continue
                        line = l.strip().split('\t')
                        variables.add((target, line[0]))
                        variables.add((target, line[1]))
                        families[line[0]] = family
                        families[line[1]] = family

    #rsa_results/euclidean/cv_partial-corr/mixed/T1/
    results = dict()
    folder = 'rsa_results'
    for root, direc, fz in os.walk(folder):
        for f in fz:
            if 'correlations' not in f:
                continue
            measure = root.split('/')[-4]
            if measure != 'euclidean':
                continue
            if measure not in results.keys():
                results[measure] = dict()
            confound_method = root.split('/')[-3]
            if confound_method != 'cv_partial-corr':
                continue
            if confound_method not in results[measure].keys():
                results[measure][confound_method] = dict()
            confound_variable = root.split('/')[-2]
            if confound_variable != 'mixed':
                continue
            if confound_variable not in results[measure][confound_method].keys():
                results[measure][confound_method][confound_variable] = dict()
            target = root.split('/')[-1]
            if target not in targets:
                continue
            if target not in results[measure][confound_method][confound_variable].keys():
                results[measure][confound_method][confound_variable][target] = dict()
            lines = list()
            with open(os.path.join(root, f)) as i:
                for l in i:
                    lines.append(l.strip().split('\t'))
            assert len(lines) == 2
            p_val = float(lines[1][lines[0].index('raw_two_sided_p')])
            corr = float(lines[1][lines[0].index('avg_corr')])
            pred = lines[1][lines[0].index('variable_name')]
            if (target, pred) not in variables:
                print(pred)
                continue
            results[measure][confound_method][confound_variable][target][pred] = {'raw_p' : p_val, 'avg_corr' : corr}
    to_be_corrected = list()
    for target, target_data in results['euclidean']['cv_partial-corr']['mixed'].items():
        for pred, pred_data in target_data.items():
            to_be_corrected.append((target, pred, pred_data['raw_p']))
    corrected = scipy.stats.false_discovery_control([v[2] for v in to_be_corrected])
    for orig, corr in zip(to_be_corrected, corrected):
        results['euclidean']['cv_partial-corr']['mixed'][orig[0]][orig[1]]['corr_p'] = corr
    folder = os.path.join(
                          'rsa_plots',
                          'cv_partial-corr',
                          'mixed',
                          'euclidean',
                          'individual_variables',
                          )
    with open(os.path.join(folder, 'single_correlations.tsv'), 'w') as o:
        o.write('target\tpredictor_family\tpredictor\tavg_corr\traw_p\tfdr-corrected_p\n')
        for target, target_data in results['euclidean']['cv_partial-corr']['mixed'].items():
            for pred, pred_data in target_data.items():
                o.write('{}\t'.format(
                                      target))
                o.write('{}\t'.format(
                                      families[pred]))
                o.write('{}\t'.format(
                                      pred))
                o.write('{}\t'.format(
                                      pred_data['avg_corr']))
                o.write('{}\t'.format(
                                      pred_data['raw_p']))
                o.write('{}\n'.format(
                                      pred_data['corr_p']))
    return results

read_correlations()
