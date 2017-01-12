###COMBINE MASKS###
import os
from process_subject import Subject
import numpy as np

subject_ids = [#'comcon_201603141', # dan's data is fuckkkkked.
               'comcon_201603142',
               'comcon_201603161',
               'comcon_201604211',
               'comcon_201604261',
               'comcon_201604281',
               'comcon_201605021']

#subject_ids = ['comcon_201603142']

feature_masks = ['localizer_object_id_searchlight_radius4mm_601vox_M', 'localizer_object_id_category_searchlight_radius4mm_600vox_M']

def combine_masks(subject_ids, mask='LOC_VTmask_ADJbin.nii.gz', feature_masks=None, feature_nii_mask=False):
	if feature_nii_mask:
		print " taking feature mask from .nii"
		run=0

	elif not feature_nii_mask and len(feature_masks)<2:
		print " need more than 1 feature mask to combine "

	else: 
		print " NOT taking feature mask from .nii, using .npy "
		run=1

	for subject_id in subject_ids:

		print subject_id

		if run ==1:
			print "getting masks for subject"
			sub_mask1 = Subject(subject_id, n_voxels=300, features_mask=feature_masks[0], mask=mask, force_load=True, run_feature_selection=False, process_retrieval=False)
			sub_mask2 = Subject(subject_id, n_voxels=300, features_mask=feature_masks[1], mask=mask, force_load=True, run_feature_selection=False, process_retrieval=False)
			
			mask1 = sub_mask1.localizer.features
			mask2 = sub_mask2.localizer.features
			
			print "combining masks"
			combo_mask = np.array([all(tup) for tup in zip(mask1, mask2)])
			
			mask_name = 'localizer_object_id+category_searchlight_radius4mm'

			mask_dir = os.path.join(
				sub_mask1.dir, 'masks', mask.split('.')[0], '{}.npy'.format(mask_name))

			print "saved new mask {} for {}".format(mask_name, subject_id)

			np.save(mask_dir, combo_mask)

		if run ==0



			mask = NEW_MASK
			masker = NiftiMasker(mask_img=mask, standardize=True,detrend=True, memory="nilearn_cache", memory_level=5)
			masker.fit()
			imgs_mean = mean_img(imgs)
	        scores_img = new_img_like(imgs_mean, scores)


	        print "creating mask file"
	        # flatten scores.
	        scores_flat = self.masker.transform(scores_img).ravel()

	        # find threshold
	        thresh = scores_flat.take(scores_flat.argsort()[::-1][nvox])

	        # flat features
	        self.features = (scores_flat>thresh) & (scores_flat!=0)