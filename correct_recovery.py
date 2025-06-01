import os

additional = [['T2-T1', 'T3-T2', 'T3-T1']]
original = list()

with open(os.path.join('dataset', 'data_41.tsv')) as i:
    for l_i, l in enumerate(i):
        line = [w.strip().replace(',',  '.') for w in l.split('\t')]
        original.append(line)
        sub = list()
        if l_i == 0:
            header = [w for w in line]
            continue
        for k in additional[0]:
            subacute = k.split('-')[0]
            acute = k.split('-')[1]
            sub_val = float(line[header.index(subacute)])
            acu_val = float(line[header.index(acute)])
            if acu_val == 1 and sub_val == 1:
                val = 0.
            else:
                val = (sub_val-acu_val)/(1-acu_val)
            sub.append(val)
        additional.append(sub)
print(additional)

with open(os.path.join('dataset', 'data_41_corrected_improvement.tsv'), 'w') as o:
    for l_i, l in enumerate(original):
        o.write('\t'.join(l))
        o.write('\t')
        o.write('\t'.join([str(w) for w in additional[l_i]]))
        o.write('\n')
