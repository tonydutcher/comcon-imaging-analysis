#group plotting
import os
import numpy as np
import pandas as pd
from plotting import heatmap

#data directory
DATA_DIR    = '/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/'

#subjects
subject_ids = [#'comcon_201603141', # dan's data is fuckkkkked.
               'comcon_201603142',
               'comcon_201603161',
               'comcon_201604211',
               'comcon_201604261',
               'comcon_201604281',
               'comcon_201605021']

#the mask_beta to be used for group averages
mask = '_obj+cat-1EXP_01BIN'
#mask = 'LOC_VT_localizer_object_id_CORR_searchlight_radius4mm_30vox'
#mask = 'LOC_VT_localizer_object_id_CORR_searchlight_radius4mm_20vox'
mask = 'LOC_VT_localizer_object_id_SVC_searchlight_radius4mm_30vox'



#different correlations files
cat_runs       = '{}_cat_runs_corr.csv'.format(mask)
objects        = '{}_objects_corr.csv'.format(mask)



#file and analysis to be run - make sure this matches!!!!
to_be_analyzed = objects
if to_be_analyzed is cat_runs:
	analysis = 'cat-runs'
else:
	analysis = 'objects'



#import data from subjects into a python dictionary - pandas panels are terrible
frame=dict()
ROWS=[]
for subject in subject_ids:
	file = os.path.join(DATA_DIR, subject, 'results/nov2016', to_be_analyzed)
	if to_be_analyzed is objects:
		d = pd.read_csv(file, delimiter=',', skipinitialspace=True, header=0)
	else:
		d = pd.read_csv(file, delimiter=',', skipinitialspace=True, header=0)
	d = d.drop(d.columns[0], 1)
	d.index = d.columns
	frame[subject] = d
	if analysis=='objects':
		unique=[r for r in list(frame[subject].index) if r not in ROWS and (ROWS.append(r) or True)]
	else:
		pass



#put dictionary into pandas panel and get mean
df = pd.Panel.from_dict(frame)


if to_be_analyzed is cat_runs:
	df_mean = df.mean(axis=0)
else:
	obj_ids = map(str, sorted(map(int, ROWS)))
	df_mean = pd.DataFrame(df.mean(axis=0), index=obj_ids, columns=obj_ids)


#plot heatmap
if to_be_analyzed is objects:
	f = heatmap(df_mean, figsize=(20,18), vmin=-0.5, vmax=0.5, title='group correlations for {} using {}'.format(analysis, mask))
	fname = os.path.join(DATA_DIR, 'group_results_nov2016', '{}_{}object-object_heatmap.png'.format('GROUP', mask))
	df_mean.to_csv(os.path.join(DATA_DIR, 'group_results_nov2016', '{}_{}object-object_corr.csv'.format('GROUP', mask)))
	f.savefig(fname)

if to_be_analyzed is cat_runs:
	f = heatmap(df_mean, vmin=-0.6,vmax=0.6, title='group correlations for {} using {}'.format(analysis, mask))
	fname = os.path.join(DATA_DIR, 'group_results_nov2016', '{}_{}cat_runs_GROUP_corr_heatmap.png'.format('GROUP', mask))
	df_mean.to_csv(os.path.join(DATA_DIR, 'group_results_nov2016', '{}_{}cat_runs_GROUP_corr.csv'.format('GROUP', mask)))
	f.savefig(fname)



#
df_mean_catrun_CORR30  = df_mean
df_mean_objects_CORR30 = df_mean

df_mean_catrun_CORR20  = df_mean
df_mean_objects_CORR20 = df_mean

df_mean_catrun_SVC30   = df_mean
df_mean_objects_SVC30  = df_mean
