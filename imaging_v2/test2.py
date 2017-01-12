import os
from localiza import Localiza

from config import config
from plotting import heatmap, heatmap2

# file paths
subject_dir   = '/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/comcon_201603142/'
metadata_file = os.path.join(subject_dir, 'behavior','comcon_201603142_localizer.csv')


# create localizer instance
l = Localiza(subject_dir, use_betas=True, mask=config.MASK_VOL, run_prefix=config.LOCALIZER_RUN_PREFIX)


# load metadata
l.load_metadata(metadata_file, 'global_onset')


# load beta labels
l.get_beta_labels()


# features selection
l.feature_selection(nvox=50, method='searchlight', label='object_id', use_backup=False, use_betas=True, radius=4, n_jobs=4, use_category=False)


# manual HACKERY - feature selection - DEBUGGING
use_category = False
use_betas    = True
label        = 'object_id'
cat_str      = ''
method       = 'searchlight'
radius       = 4
folds        = 4
n_jobs       = 4
nvox         = 50
base_str     = 'localizer_TEST-BETA-SVC_{}{}_{}_radius{}mm_{}vox'.format(label, cat_str, method, radius, nvox)

# paths
l.vox_selection_base   = os.path.join(l.features_dir, base_str)
l.vox_selection_backup = '{}.npy'.format(l.vox_selection_base)
l.vox_selection_scores = '{}.nii.gz'.format(l.vox_selection_base)
l.vox_selection_mask   = '{}_mask.nii.gz'.format(l.vox_selection_base)

l._feature_selection_rsa_searchlight(nvox, label, use_betas=use_betas, use_category=use_category, radius=radius)






imgs   = l.masker_beta.inverse_transform(l.imgs)
labels = l.beta_labels[label]

#imgs   = l.masker.inverse_transform(l.imgs)
#labels = l.label[label]

#SET-UP FOR SEARCHLIGHT   -------------------------------------------


#RSA ESTIMATOR            -------------------------------------------




from localiza import Localiza
localizer = Localiza(l.subject_dir, l.mask)
localizer.imgs = X
localizer.features  = None
localizer.nfeatures = X.shape[1]
localizer.nframes   = X.shape[0]
localizer.beta_labels = {}

#SEARCHLIGHT              -------------------------------------------
#### SPHERE EXTRACTOR     -------------------------------------------

from estimatas import RSAClassifier
rsa = RSAClassifier(l.subject_dir, l.mask, labels)

from sklearn.cross_validation import KFold
from nilearn.decoding import SearchLight

# get patterns for each object

kfold = KFold(l.imgs.shape[0], n_folds=folds)
searchlight = SearchLight(l.mask, radius=radius, verbose=2, n_jobs=4, estimator='svc', cv=kfold, process_mask_img=l.mask)
searchlight.fit(imgs, labels)
scores = searchlight.scores_
scores_3D = searchlight.scores_3D
coordinates = searchlight.scores_coordinates
process_mask = searchlight.process_mask

import numpy as np
from nilearn.image import new_img_like, mean_img
from nilearn import masking

#### SPHERE EXTRACTOR     -------------------------------------------
print "searchlight FIT done"
scores = searchlight.scores_
scores_3D = searchlight.scores_3D
#self.scores_3D = scores_3D
#self.scores = scores

print "searchlight coordinates and process mask!"
coordinates = searchlight.scores_coordinates
process_mask = searchlight.process_mask
self.coordinates = coordinates
self.process_mask = process_mask

self.searchlight = searchlight

# nifti map for score
print "saving searchlight score image map"
imgs_mean = mean_img(imgs)
scores_img = new_img_like(imgs_mean, scores)
scores_img.to_filename(self.vox_selection_scores)

#threshold scores 
# lets get the top spheres
thresh = scores.take(scores.argsort()[::-1][nvox])
feats = (scores>thresh) & (scores!=0)

#extract spheres of thresholded scores
select = np.hstack([np.array([0])]*len(scores))
for i,ind in enumerate(coordinates):
    if feats[i]:
        print coordinates.rows[i]
        sphere=map(int, np.array(coordinates.rows[i]))
        select[sphere] = 1
    else:
        pass

#put thresholded spheres back into 3D mask
thresh_scores_3D=np.zeros(process_mask.shape)
thresh_scores_3D[process_mask]=select
thresh_scores_img=new_img_like(imgs_mean, thresh_scores_3D)
thresh_scores_img.to_filename(self.vox_selection_mask)

#### SPHERE EXTRACTOR     -------------------------------------------


#SEARCHLIGHT              -------------------------------------------

# name of the feature selected area
name_of_analysis = 'test_FULL_LOC_VT'

# heatmap object-object relationships
corr = l.compare_objects_betas()
fig = heatmap(corr)
fig.savefig('{}_{}heatmap.png'.format(name_of_analysis, 'object-object'))
#corr.to_csv('{}_{}corr.csv'.format(name_of_analysis, 'object-object'))


# heatmap category-runs relationships
corr = l.compare_category_runs_betas()
fig = heatmap(corr)
fig.savefig('{}_{}heatmap.png'.format(name_of_analysis, 'category-by-run'))
#corr.to_csv('{}_{}corr.csv'.format(name_of_analysis, 'category-by-run'))


# heatmap run-run relationships
corr = l.compare_runs_betas()
fig = heatmap(corr)
f.savefig('{}_{}heatmap.png'.format(name_of_analysis, 'runs_'))


# heatmap category-category relationship
corr = l.compare_category_betas()
f = fig = heatmap(corr)
f.savefig('{}_{}heatmap.png'.format(name_of_analysis, 'category_'))
