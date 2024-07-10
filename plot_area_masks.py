import matplotlib
import nilearn
import numpy
import os

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from nilearn import datasets, plotting, surface

fsaverage = datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
dataset =nilearn.datasets.fetch_atlas_aal(version='SPM12')
maps = dataset['maps']
maps_data = nilearn.image.load_img(maps).get_fdata()
idxs = [float(v) for v in dataset['indices']]
labels = dataset['labels']
assert len(idxs) == len(labels)
#maps = dataset['maps']
#maps_data = maps.get_fdata()
rel_names = {
             'lesions' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            'Frontal_Inf_Orb_L',
                            'Frontal_Inf_Orb_R',
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                            'Temporal_Sup_L', 'Temporal_Sup_R',
                            'Temporal_Mid_L', 'Temporal_Mid_R',
                            'Temporal_Inf_L', 'Temporal_Inf_R',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              'Frontal_Mid_L',
                              'Frontal_Mid_R'
                              ]
                              },
             'activations' : {
                    'ifg' : [
                            'Frontal Orbital Cortex',
                            'Frontal_Inf_Orb_L',
                            'Frontal_Inf_Orb_R',
                            ],
                    'sma' : [
                            'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)',
                            'Supp_Motor_Area_L'
                            ],
                    'ptl' : [
                            'Superior Temporal Gyrus, posterior division',
                            'Superior Temporal Gyrus, posterior division',
                            'Middle Temporal Gyrus, posterior division',
                            'Temporal_Sup_L', 'Temporal_Sup_R',
                            'Temporal_Mid_L', 'Temporal_Mid_R',
                            'Temporal_Inf_L', 'Temporal_Inf_R',
                              ],
                    'dlpfc' : [
                              'Middle Frontal Gyrus',
                              'Frontal_Mid_L',
                              'Frontal_Mid_R'
                              ]
                              },
             }

colormaps = {
             'ifg' : 'BuGn',
             'ptl' : 'RdPu',
             'sma' : 'Blues',
             'dlpfc' : 'YlOrBr',
             }
vals = {
        'ifg' :  {'max' : .135, 'min' : 0.011},
        'dlpfc' : {'max' : .425, 'min' : 0.29},
        'ptl' : {'max' : .7, 'min' : 0.575},
        'sma' : {'max' : 1., 'min' : 0.86},
             }
### generic cmap
cmaps=dict()
colors=[
        'white',
        'mediumseagreen',
        'white',
        'goldenrod',
        'white',
        'mediumvioletred',
        'white',
        'dodgerblue',
        ]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
cmaps['symmetric'] = cmap
### right
right=[
       'white',
       'mediumvioletred',
       'white',
       'forestgreen',
       ]
right_cmap = LinearSegmentedColormap.from_list("right", right)
### left
left=[
      'white',
      'paleturquoise',
      'white',
      'lightpink',
      'white',
      'khaki',
      'white',
      'steelblue',
      ]
left_cmap = LinearSegmentedColormap.from_list("left", left)
cmaps['asymmetric'] = {
                       'left' : left_cmap,
                       'right' : right_cmap,
                       }
for k, both_cmap in cmaps.items():
    for side, relevant_labels_names in rel_names.items():
        if k == 'asymmetric':
            if side == 'lesions':
                continue
        out = os.path.join(
                           'region_maps',
                           'plots',
                           k,
                           side,
                           )
        os.makedirs(
                    out,
                    exist_ok=True,
                    )
        for view in [
                     'lateral',
                     'ventral',
                     #'dorsal',
                     #'anterior',
                     ]:
            if view == 'lateral':
                alpha = 0.1
            else:
                alpha = 0.05
            if k == 'symmetric':
                relevant_labels = {i : vals[k]['max'] for i, l in enumerate(labels) for k, v in relevant_labels_names.items() if l in v}
                #relevant_labels = {i : vals[k]['max'] for i, l in zip(idxs, labels) for k, v in relevant_labels_names.items() if l in v}

                msk = numpy.array([relevant_labels[v] if v in relevant_labels.keys() else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                atl_img = nilearn.image.new_img_like(maps, msk)
                cmap = both_cmap
            if k == 'asymmetric':
                ### right
                if side == 'activations':
                    missing = [
                               'sma',
                               'ptl',
                               ]
                    curr_vals = {
                        'ifg' :  {'max' : .33, 'min' : 0.011},
                        'dlpfc' : {'max' : 1., 'min' : 0.66},
                             }
                else:
                    missing = [
                               'sma',
                               ]
                    curr_vals = {
                        'ifg' :  {'max' : .2, 'min' : 0.011},
                        'dlpfc' : {'max' : .6, 'min' : 0.4},
                        'ptl' : {'max' : .99, 'min' : 0.8},
                             }
                relevant_labels = dict()
                for i, l in enumerate(labels):
                    for lab, v in relevant_labels_names.items():
                        if lab in missing:
                            continue
                        if l in v:
                            relevant_labels[i] = curr_vals[lab]['max']
                cmap = both_cmap['right']
                msk = numpy.array(
                            [relevant_labels[v] if v in relevant_labels.keys() else 0. for v in maps_data.flatten()]
                            ).reshape(maps_data.shape)
                atl_img = nilearn.image.new_img_like(maps, msk)
            print(sum(msk.flatten()))
            ### Right
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
                    bg_on_data=False,
                    bg_on_map=False,
                    bg_map=None,
                    darkness=0.6,
                    alpha=alpha,
                    view=view,
                    cmap=cmap,
                    vmin=0.,
                    vmax=1.
                    )
            #pyplot.savefig(os.path.join(out, \
            r.savefig(os.path.join(
                   out,
                   'surface_right_{}_{}.svg'.format(
                                               side,
                                               view,
                                               ),
                    ),
                    dpi=600
                    )
            #pyplot.savefig(os.path.join(out, \
            r.savefig(os.path.join(
                    out,
                    'surface_right_{}_{}.jpg'.format(
                                               side,
                                               view,
                                               ),
                    ),
                    dpi=600
                    )
            pyplot.clf()
            pyplot.close()
            ### Left
            if k == 'asymmetric':
                del relevant_labels
                del atl_img
                del cmap
                del msk
                ### left
                relevant_labels = {i : vals[k]['max'] for i, l in enumerate(labels) for k, v in relevant_labels_names.items() if l in v}
                #relevant_labels = {i : vals[k]['max'] for i, l in zip(idxs, labels) for k, v in relevant_labels_names.items() if l in v}
                msk = numpy.array([relevant_labels[v] if v in relevant_labels.keys() else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
                atl_img = nilearn.image.new_img_like(maps, msk)
                cmap = both_cmap['left']
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
                    colorbar=True,
                    threshold=0.01,
                    bg_on_data=False,
                    bg_on_map=False,
                    bg_map=None,
                    cmap=cmap,
                    view=view,
                    darkness=0.5,
                    alpha=alpha,
                    vmin=0.,
                    vmax=1.
                    )
            l.savefig(os.path.join(out, \
                        'surface_left_{}_{}.svg'.format(side, view),
                        ),
                        dpi=600
                        )
            l.savefig(os.path.join(out, \
                        'surface_left_{}_{}.jpg'.format(side, view),
                        ),
                        dpi=600
                        )
            pyplot.clf()
            pyplot.close()
