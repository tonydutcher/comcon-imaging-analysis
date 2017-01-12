import os
import numpy as np
import nibabel as nb
from config import config
from localiza import Localiza
from retrieve import Retrieve

#DATA_DIR='/Users/mh46989/lewpealab/DATA/comcon/fmri/'
DATA_DIR='/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/'
#mask='/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/comcon_201603142/masks/LOC_VTmask_ADJbin/localizer_object_id++category_searchlight_radius_4mm_1EXP_01BIN_mask.nii.gz'

class Subject(object):
    def __init__(self, 
        subject_id, 
        data_dir=DATA_DIR, 
        n_voxels=None, 
        features_mask=None,
        mask=config.MASK_VOL, 
        use_backup=False,
        force_load=False,
        use_betas=False,
        run_feature_selection=False,
        use_category=None,
        feature_mask_type=None,
        process_retrieval=False, 
        **kwargs):

        self.id = subject_id
        self.n_voxels = n_voxels
        self.data_dir = data_dir
        self.dir = os.path.join(data_dir, subject_id)
        self.mask = os.path.join(self.dir,'masks', mask)
        self.behav_dir = os.path.join(self.dir, 'behavior')
        self.results_dir = os.path.join(self.dir, 'results/nov2016')

        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        if not os.path.exists(self.behav_dir):
            raise Exception("no behavior directory found: {}".format(self.behav_dir))

        if features_mask is not None:
            self.features_mask = os.path.join(self.dir, 'masks', mask.split('.')[0], features_mask)

        if use_category:
            print "USING CATEGORY!"
        else:
            print "NOT USING CATEGORY"

        self.load_localizer(use_betas=use_betas, run_feature_selection=run_feature_selection, 
            use_category=use_category, feature_mask_type=feature_mask_type, **kwargs)
        
        print self.localizer.nfeatures, self.localizer.features

        if process_retrieval:
            self.load_retrieval()

    def load_localizer(self, 
        prefix=None, 
        use_backup=False, 
        use_betas=None,
        info_file=None, 
        force_load=True, 
        run_feature_selection=False, 
        use_category=None,
        feature_mask_type=None,
        n_jobs=4):
    
        # localizer info file
        if info_file is None:
            fname = '{}_localizer.csv'.format(self.id)
            self.localizer_behav_file = os.path.join(self.behav_dir, fname)

        if feature_mask_type is None:
            use_previous_feature_mask=False

        else:
            use_previous_feature_mask=True

        # localizer
        print "loading localizer"
        self.localizer = Localiza(self.dir, use_betas=use_betas, mask=self.mask, run_prefix=prefix)
        self.localizer.load_metadata(self.localizer_behav_file, onset_col='global_onset')

        if use_betas:
            self.localizer.get_beta_labels()

        # feature selection 
        if run_feature_selection:
            print "running feature selection"

            if use_category:
                print "USING CATEGORY!"
            else:
                print "NOT USING CATEGORY"

            print self.n_voxels
            self.localizer.feature_selection(use_betas=use_betas, nvox=self.n_voxels, method='searchlight', 
                use_category=use_category, n_jobs=4, use_backup=False, radius=4)
            
            self.features  = self.localizer.features
            self.nfeatures = sum(self.features)
            print self.features, self.nfeatures
            print "feature selection finished"
        
        elif use_previous_feature_mask:
            print "loading previous feature mask {}".format(self.features_mask)

            if feature_mask_type == '.nii':
                print " running feature selection with existing .nii {}".format(self.features_mask)
                self.localizer.vox_selection_mask = '{}_mask.nii.gz'.format(self.features_mask)
                self.localizer.vox_selection_backup = '{}.npy'.format(self.features_mask)

                # load in and flatten .nii mask
                print " loading .nii mask "
                select_scores = nb.load(self.localizer.vox_selection_mask)
                select_flat = self.localizer.masker.transform(select_scores).ravel()
                print "success!"

                self.localizer.features = select_flat!=0
                print " new flattened features!"

                # save to file
                np.save(self.localizer.vox_selection_backup, self.localizer.features)

            elif feature_mask_type == '.npy':
                print " running feature selection with existing .npy {}".format(self.features_mask)
                self.localizer.vox_selection_backup = '{}.npy'.format(self.features_mask)
                self.localizer.features  = np.load(self.localizer.vox_selection_backup)
                self.localizer.nfeatures = sum(self.localizer.features)

        else:
            print "No feature selection used and no previous feature selection mask used"
            #self.localizer.features = np.array([i!=None for i in self.localizer.beta_imgs]).astype(int)
            self.localizer.features=None
            print self.localizer.features

    def load_retrieval(self, 
        prefix=config.RETRIEVAL_RUN_PREFIX, 
        info_file=None, 
        force_load=False, **kwargs):

        # retrieval info file
        if info_file is None:
            fname = '{}_retrieval.csv'.format(self.id)
            self.retrieval_behav_file = os.path.join(self.behav_dir, fname)

        # retrieval
        print "loading retrieval"
        self.retrieval = Retrieve(self.dir, 'comcon_retrieval', self.mask, force_load=force_load)
        self.retrieval.load_metadata(self.retrieval_behav_file, onset_cols=['scene_onset','position_onset'])
        
        # match features with localizer
        print "matching localizer features for retrieval phase features"
        self.retrieval.features = self.localizer.features
        print self.retrieval.features        