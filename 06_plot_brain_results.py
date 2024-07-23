import matplotlib
import nilearn
import numpy
import os

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from nilearn import datasets, plotting, surface

fsaverage = datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
maps = dataset['maps']
maps_data = maps.get_fdata()
labels = dataset['labels']
rel_names = {
             'lesions' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              ]
                              },
             'activations' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            ],
                    'sma' : [
                            'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              ]
                              },
             }

vals = {
        'ifg' :  {'max' : .135, 'min' : 0.011},
        'dlpfc' : {'max' : .425, 'min' : 0.29},
        'ptl' : {'max' : .7, 'min' : 0.575},
        'sma' : {'max' : 1., 'min' : 0.86},
             }
mapper = {
          'L_DLPFC' : 'left_dlpfc',
          'L_IFGorb' : 'left_ifg',
          'L_PTL' : 'left_ptl',
          'L_DLPFC_T1' : 'left_dlpfc',
          'L_IFGorb_T1' : 'left_ifg',
          'R_DLPFC_T1' : 'right_dlpfc',
          'R_IFGorb_T1' : 'right_ifg',
          'L_SMA_T1' : 'left_sma',
          'R_SMA_T1' : 'right_sma',
          'L_PTL_T1' : 'left_ptl',
          'L_DLPFC_T2' : 'left_dlpfc',
          'L_IFGorb_T2' : 'left_ifg',
          'R_DLPFC_T2' : 'right_dlpfc',
          'R_IFGorb_T2' : 'right_ifg',
          'L_SMA_T2' : 'left_sma',
          'R_SMA_T2' : 'right_sma',
          'L_PTL_T2' : 'left_ptl',
          }
colors=['white', 'mediumseagreen', 'white',  'goldenrod', 'white', 'mediumvioletred', 'white', 'dodgerblue']
pers_cmap = LinearSegmentedColormap.from_list("mycmap", colors)
cmaps = [
         #pers_cmap,
         #'cividis',
         'YlOrBr',
         #'coolwarm',
         ]

### reading results
for cmap in cmaps:
    for root, direc, fz in os.walk('rsa_plots/'):
        if 'individual' not in root:
            continue
        for f in fz:
            if '.results' not in f:
                continue
            print([root, f])
            with open(os.path.join(root, f)) as i:
                f_name = f.split('.')[0].split('_')
                target = f_name[0]
                predictors = f_name[1]
                if 'conn' in predictors:
                    continue
                avgs = dict()
                for l_i, l in enumerate(i):
                    if l_i == 0:
                        continue
                    line = l.strip().split('\t')
                    #avgs[mapper[line[0]]] = float(line[2])/0.3
                    #avgs[mapper[line[1]]] = float(line[3])/0.3
                    avgs[mapper[line[0]]] = float(line[2])
                    avgs[mapper[line[1]]] = float(line[3])
                #vmax=max(avgs.values())
                if predictors == 'lesions' and len(target) == 2:
                    vmax=0.25
                elif predictors == 'lesions' and len(target) > 2:
                    vmax=0.3
                else:
                    vmax=0.2
                vmax = 0.3
                #vmin=min(avgs.values())
                vmin=0.
                print(avgs)
                case = [k for k in rel_names.keys() if k in predictors]
                assert len(case) == 1
                relevant_labels_names = rel_names[case[0]]
                out = os.path.join('brain_plots', cmap, target, predictors, )
                os.makedirs(out, exist_ok=True)
                for side in ['left', 'right']:
                    if 'lesions' in predictors and side == 'right':
                        continue
                    if side == 'right' and 'act' in predictors:
                        relevant_labels_names = {l : v for l, v in relevant_labels_names.items() if 'ptl' not in l and 'sma' not in l}
                    for view in [
                                 #'dorsal',
                                 'lateral',
                                 #'ventral',
                                 #'anterior',
                                 ]:
                        #for view in [[20, 180]]:
                        if view == 'lateral':
                            alpha = 0.1
                        else:
                            alpha = 0.05
                        #for area_i, area in enumerate(relevant_labels_names.keys()):
                        #relevant_labels = [i for i, l in enumerate(labels) if l in [val for v in relevant_labels_names.values() for val in v]]
                        #cols =
                        rel_avgs = {k.split('_')[1] : v for k, v in avgs.items() if side in k}
                        #relevant_labels = {i : vals[k]['max'] for i, l in enumerate(labels) for k, v in relevant_labels_names.items() if l in v}
                        #relevant_labels = {i : (rel_avgs[k]-vals[k]['min'])/(vals[k]['max']-vals[k]['min']) for i, l in enumerate(labels) for k, v in relevant_labels_names.items() if l in v}
                        relevant_labels = {i : rel_avgs[k] for i, l in enumerate(labels) for k, v in relevant_labels_names.items() if l in v}
                        msk = numpy.array([relevant_labels[v] if v in relevant_labels.keys() else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                        #cmap=colormaps[area]
                        #cmap='cividis'
                        #alpha=0.1 if area_i==0 else 0.
                        '''
                        if 'conn' in side:
                            msk = numpy.array([v+1 if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                        elif 'act' in side:
                            msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                            cmap = 'Paired'
                        else:
                            msk = numpy.array([v+1 if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                            cmap = 'Set2'
                        '''
                        #print(side)
                        print(sum(msk.flatten()))
                        #out = os.path.join('region_maps', 'maps', side)
                        #os.makedirs(out, exist_ok=True)
                        atl_img = nilearn.image.new_img_like(maps, msk)
                        #atl_img.to_filename(os.path.join('region_maps', 'maps', side, '{}.nii.gz'.format(side)))
                        ### Right
                        if side == 'right':
                            texture = surface.vol_to_surf(
                                                          atl_img,
                                                          fsaverage.pial_right,
                                                          interpolation='nearest',
                                                          radius=0.,
                                                          n_samples=1,
                                    )
                            r= plotting.plot_surf_stat_map(
                                    fsaverage.pial_right,
                                    texture,
                                    hemi='right',
                                    title='{} - right hemisphere'.format(side),
                                    threshold=0.01,
                                    colorbar=True,
                                    #colorbar=False,
                                    #bg_map=fsaverage.sulc_right,
                                    bg_on_data=False,
                                    bg_on_map=False,
                                    bg_map=None,
                                    darkness=0.6,
                                    #cmap='cividis',
                                    #view='medial',
                                    alpha=alpha,
                                    view=view,
                                    #cmap='cividis'
                                    #avg_method='median',
                                    cmap=cmap,
                                    vmin=vmin,
                                    vmax=vmax,
                                    )
                            #r.savefig(os.path.join(out, \
                            pyplot.savefig(os.path.join(out, \
                                    'surface_{}_{}.svg'.format(side, view),
                                    ),
                                    dpi=600
                                    )
                            #r.savefig(os.path.join(out, \
                            pyplot.savefig(os.path.join(out, \
                                    'surface_{}_{}.jpg'.format(side, view),
                                    ),
                                    dpi=600
                                    )
                            pyplot.clf()
                            pyplot.close()
                        else:
                            ### Left
                            texture = surface.vol_to_surf(
                                                          atl_img,
                                                          fsaverage.pial_left,
                                                          interpolation='nearest',
                                                          radius=0.,
                                                          n_samples=1,
                                    )
                            l = plotting.plot_surf_stat_map(
                                    fsaverage.pial_left,
                                    texture,
                                    hemi='left',
                                    title='{} - left hemisphere'.format(side),
                                    #colorbar=False,
                                    colorbar=True,
                                    threshold=0.01,
                                    #bg_map=fsaverage.sulc_left,
                                    bg_on_data=False,
                                    bg_on_map=False,
                                    bg_map=None,
                                    #cmap='Spectral_R',
                                    #cmap='Wistia',
                                    #cmap=cmaps[side],
                                    #cmap='cividis',
                                    cmap=cmap,
                                    #view='medial',
                                    view=view,
                                    darkness=0.5,
                                    alpha=alpha,
                                    vmin=vmin,
                                    vmax=vmax,
                                    )
                            l.savefig(os.path.join(out, \
                                        'surface_{}_{}.svg'.format(side, view),
                                        ),
                                        dpi=600
                                        )
                            l.savefig(os.path.join(out, \
                                        'surface_{}_{}.jpg'.format(side, view),
                                        ),
                                        dpi=600
                                        )
                            pyplot.clf()
                            pyplot.close()
