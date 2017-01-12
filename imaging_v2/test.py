import os
from config import config
from plotting import heatmap, heatmap2
from subject_tony import Subject_T



# subject_ids = [#'comcon_201603141', # dan's data is fuckkkkked.
#                'comcon_201603142',
#                'comcon_201603161',
#                'comcon_201604211',
#                'comcon_201604261',
#                'comcon_201604281',
#                'comcon_201605021']



subject_ids = ['comcon_201604281']



# relevant variable labels
item_label     = config.LOCALIZER_ITEM_LABEL
category       = config.LOCALIZER_CATEGORY_LABEL


# whose label coding is being used when averaging the pattern of activation 
AUTHOR         = 'MT'

# settings for partial localizer
roi_file                 = config.MASK_VOL #'test_roi.nii.gz'
feature_mask             = 'MT_localizer_object_id_searchlight_radius8mm_149vox'
use_backup               = True


# create localizer instance
# subject = Subject_T(subject_ids[0], features_mask=feature_mask, mask=roi_file, run_prefix=config.LOCALIZER_RUN_PREFIX, use_backup=True)

# testing(subject_ids)

def testing(subject_ids=subject_ids):
	for subject_id in subject_ids:
		subject = Subject_T(subject_id, features_mask=feature_mask, 
			mask=roi_file, run_prefix=config.LOCALIZER_RUN_PREFIX, use_backup=True)
		
		# category-run
		df = subject.localizer.compare_category_run_betas()
		f  = heatmap(df)
		fname = os.path.join(subject.localizer.features_dir,'{}_{}_cat_runs_heatmap.png'.format(AUTHOR, feature_mask))
		df.to_csv(os.path.join(subject.localizer.features_dir, '{}_{}_cat_runs{}'.format(AUTHOR, feature_mask, '_corr.csv')))
		f.savefig(fname)
		
		# item
		df = subject.localizer.compare_objects_betas()
		f  = heatmap(df)
		fname = os.path.join(subject.localizer.features_dir,'{}_{}_objects_heatmap.png'.format(AUTHOR, feature_mask))
		df.to_csv(os.path.join(subject.localizer.features_dir, '{}_{}_objects{}'.format(AUTHOR, feature_mask, '_corr.csv')))
		f.savefig(fname)
		

