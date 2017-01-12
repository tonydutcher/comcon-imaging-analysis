import os
import argparse
import numpy as np
import pandas as pd
from glob import glob
from config import config
from process_subject import Subject
from plotting import heatmap

features_mask = 'localizer_object_id_searchlight_radius4mm_20vox'
#features_mask = 'localizer_object_id_searchlight_radius4mm_30vox'
#features_mask = 'localizer_object_id_searchlight_SVC_radius4mm_20vox'
#features_mask = 'localizer_object_id_searchlight_SVC_radius4mm_30vox'

mask = config.MASK_VOL

DATA_DIR='/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/'
group_results_dir = os.path.join(DATA_DIR, 'group_results_nov2016')

def testing(subject_ids):

	#empty dictionary for rank order correaltions across subjects
	SUBJECT_SCENE      = {}
	SUBJECT_POS        = {}
	SUBJECT_BOTH       = {}
	SUBJECT_PATTERNS   = {}
	SUBJECT_REACTCORRS = {}
	SUBJECT_CORR_RANKS = {}

	for subject_id in subject_ids:

		print subject_id
		# get subject data
		subject = Subject(subject_id, n_voxels=500, use_betas=True, 
			features_mask=features_mask, mask=mask, feature_mask_type='.npy', 
			force_load=True, run_feature_selection=False, process_retrieval=True)
		
		# # QA over object-object relationships
		# df = subject.localizer.compare_objects()
		# fname = os.path.join(group_results_dir,'{}_REG_obj+cat-1EXP_01BIN_object-object'.format(subject_id))
		# df.to_csv('{}_corr.csv'.format(fname))
		# f = heatmap(df)
		# f.savefig('{}_heatmap.png'.format(fname))
		# print "saved object-object relationships"

		# # QA over category-runs relationships
		# df = subject.localizer.compare_category_run()
		# fname = os.path.join(group_results_dir,'{}_REG_obj+cat-1EXP_01BIN_category-runs'.format(subject_id))
		# df.to_csv('{}_corr.csv'.format(fname))
		# f = heatmap(df)
		# f.savefig('{}_heatmap.png'.format(fname))
		# print "saved category-runs relationships heatmap"

		# df = subject.localizer.compare_runs()
		# fname = os.path.join(group_results_dir,'{}_REG_obj+cat-1EXP_01BIN_runs'.format(subject_id))
		# df.to_csv('{}_corr.csv'.format(fname))
		# f = heatmap(df)
		# f.savefig('{}_heatmap.png'.format(fname))
		# print "saved runs relationships heatmap"

		# df = subject.localizer.compare_category()
		# f = heatmap(df)
		# fname = os.path.join(group_results_dir,'{}_obj+cat-1EXP_01BIN_categories_heatmap.png'.format(subject_id))
		# f.savefig(fname)
		# print "saved category relationships heatmap"

		loc_labels  = pd.DataFrame(subject.localizer.label)
		retr_labels = pd.DataFrame(subject.retrieval.label)
		labs        = np.hstack([ 'S','P','B', [ x for x in map(str, sorted(map(int, subject.localizer.object_ids))) ] ] )
		families    = [x for x in np.unique(subject.retrieval.label['family_id']) if str(x) != 'nan']
		recog_file = os.path.join(subject.behav_dir, '{}_recognition.csv'.format(subject.id))
		recog      = pd.read_table(recog_file, sep=',')

		# empty dictionarys
		PATTERNS      = {}
		CORR_RANKS    = {}
		REACTCORRS    = {}
		SIMPLE_P      = {}
		SIMPLE_S      = {}
		SIMPLE_B      = {}
		countP = 0
		countS = 0
		countB = 0

		for fam_id in families:

			# gets the objects and target within the family id
			in_fam = (loc_labels.scene_id==fam_id).values
			object_ids = np.unique(loc_labels[in_fam].object_id)
			target = np.unique(loc_labels[in_fam].target_id)

			print fam_id, target, object_ids
			# creates matrix for mean patterns of activity
			pattern = np.zeros([len(labs), sum(subject.retrieval.features)])

			# puts the scene and position patterns of activity in matrix
			pattern[0] = subject.retrieval.mean_pattern('scene', str(fam_id))
			pattern[1] = subject.retrieval.mean_pattern('position', str(fam_id))
			pattern[2] = subject.retrieval.mean_pattern('both', str(fam_id))

				# runs through the object ids and populates rest of pattern matrix
			for i, ind in enumerate(labs):
				# hack, passes over scene and position rows
				if ind == 'S':
					pass
				elif ind == 'P':
					pass
				elif ind == 'B':
					pass
				else:
					# gets pattern for object id in localizer
					pattern[i] = subject.localizer.mean_beta_pattern(object_id=str(labs[i]))
					#corr=pd.DataFrame(np.corrcoef(pattern), columns=labs, index=labs)

			# runs a correlation for all objects with the scene and pattern
			# this can be done better... very clunky and slow

			corr=pd.DataFrame(np.corrcoef(pattern), columns=labs, index=labs)

			reactivationCorrelations = pd.DataFrame(np.array(corr[['S', 'P', 'B']][:3]).reshape([1,9]))
			REACTCORRS[fam_id] = reactivationCorrelations

			# 	######### PATTERNS OF BRAIN ACITIVTY FOR SCENE AND POSITION CUE #########
			# 	#########################################################################
			# sorts data into ranks, lots of hacking here... :|
			SCENE=pd.DataFrame(corr['S'][3:].order(kind="quicksort"))
			SCENE['rank']=[x+1 for x in list(reversed(range(0,len(labs[3:]))))]

			POS=pd.DataFrame(corr['P'][3:].order(kind="quicksort"))
			POS['rank']=[x+1 for x in list(reversed(range(0,len(labs[3:]))))]

			BOTH=pd.DataFrame(corr['B'][3:].order(kind="quicksort"))
			BOTH['rank']=[x+1 for x in list(reversed(range(0,len(labs[3:]))))]

			# this cycles through the three reactivation periods, collecting information through each
			for TYPE in ['S', 'P', 'B']:
				for i, obj_id in enumerate(object_ids):
					if TYPE == 'P':
						# for position 
						if obj_id == str(int(target)):
							dist = POS.loc[str(int(target)),'rank']-POS.loc[str(int(obj_id)),'rank']
							SIMPLE_P[countP] = {
							'family id': fam_id, 
							'object id': obj_id, 
							'rank': POS.loc[str(int(obj_id)),'rank'],
							'tarPos': POS.loc[str(int(target)),'rank'],
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0],
							'dist': dist, 
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': 999,
							'object_pos': 999,
							'object_type': 999,
							'target_type': 999
							}
							countP = countP+1
						else:
							dist = POS.loc[str(int(target)),'rank']-POS.loc[str(int(obj_id)),'rank']
							SIMPLE_P[countP] = {
							'family id': fam_id, 
							'object id': obj_id, 
							'rank': POS.loc[str(int(obj_id)),'rank'],
							'tarPos': POS.loc[str(int(target)),'rank'],
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0],
							'dist': dist, 
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': recog.loc[recog.object_id.astype(np.str)==obj_id,'object_resp'].values[0],
							'object_pos': recog.loc[recog.object_id.astype(np.str)==obj_id, 'object_position'].values[0],
							'object_type': recog.loc[recog.object_id.astype(np.str)==obj_id, 'object_type'].values[0],
							'target_type': recog.loc[recog.object_id.astype(np.str)==obj_id, 'target_type'].values[0]
							}
							countP = countP+1
					elif TYPE == 'S':	
						# for scene
						if obj_id == str(int(target)):
							dist = SCENE.loc[str(int(target)),'rank']-SCENE.loc[str(int(obj_id)),'rank']
							SIMPLE_S[countS] = {
							'family id': fam_id, 
							'object id': obj_id, 
							'rank': SCENE.loc[str(int(obj_id)),'rank'],
							'tarPos': SCENE.loc[str(int(target)),'rank'],
							'dist': dist, 
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0],
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': 999,
							'object_pos': 999,
							'object_type': 999,
							'target_type': 999
							}
							countS = countS+1
						else:
							dist = SCENE.loc[str(int(target)),'rank']-SCENE.loc[str(int(obj_id)),'rank']
							SIMPLE_S[countS] = {
							'family id': fam_id,
							'object id': obj_id,
							'rank': SCENE.loc[str(int(obj_id)),'rank'],
							'tarPos': SCENE.loc[str(int(target)),'rank'], 
							'dist': dist,
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0], 
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': recog.loc[recog.object_id.astype(np.str)==obj_id,'object_resp'].values[0],						
							'object_pos': recog.loc[recog.object_id.astype(np.str)==obj_id, 'object_position'].values[0],
							'object_type': recog.loc[recog.object_id.astype(np.str)==obj_id, 'object_type'].values[0],
							'target_type': recog.loc[recog.object_id.astype(np.str)==obj_id, 'target_type'].values[0]
							}
							countS = countS+1
					else:	
						# for scene
						if obj_id == str(int(target)):
							dist = BOTH.loc[str(int(target)),'rank']-BOTH.loc[str(int(obj_id)),'rank']
							SIMPLE_B[countB] = {
							'family id': fam_id, 
							'object id': obj_id, 
							'rank': BOTH.loc[str(int(obj_id)),'rank'],
							'tarPos': BOTH.loc[str(int(target)),'rank'],
							'dist': dist, 
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0],
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': 999,
							'object_pos': 999,
							'object_type': 999,
							'target_type': 999
							}
							countB = countB+1
						else:
							dist = BOTH.loc[str(int(target)),'rank']-BOTH.loc[str(int(obj_id)),'rank']
							SIMPLE_B[countB] = {
							'family id': fam_id,
							'object id': obj_id,
							'rank': BOTH.loc[str(int(obj_id)),'rank'],
							'tarPos': BOTH.loc[str(int(target)),'rank'], 
							'dist': dist,
							'target': retr_labels.loc[retr_labels.family_id==fam_id,'target'].values[0], 
							'retr_resp': str(retr_labels.loc[retr_labels.family_id==fam_id,'probe_resp'].values[0]),
							'recog_resp': recog.loc[recog.object_id.astype(np.str)==obj_id,'object_resp'].values[0],						
							'object_pos': recog.loc[recog.object_id.astype(np.str)==obj_id,'object_position'].values[0],
							'object_type': recog.loc[recog.object_id.astype(np.str)==obj_id,'object_type'].values[0],
							'target_type': recog.loc[recog.object_id.astype(np.str)==obj_id,'target_type'].values[0]
							}
							countB = countB+1

			#dictionary for of family id for a subject
			CORR_RANKS[fam_id] = { 'position': POS, 'scene': SCENE, 'both':BOTH }
			PATTERNS[fam_id]   = corr

		#dictionary for subjects			
		SUBJECT_PATTERNS[subject.id]   = pattern
		SUBJECT_SCENE[subject.id]      = SIMPLE_S
		SUBJECT_POS[subject.id]        = SIMPLE_P 
		SUBJECT_BOTH[subject.id]       = SIMPLE_B
		SUBJECT_CORR_RANKS[subject.id] = CORR_RANKS


	return SUBJECT_SCENE, SUBJECT_POS, SUBJECT_BOTH, SUBJECT_CORR_RANKS, SUBJECT_PATTERNS
