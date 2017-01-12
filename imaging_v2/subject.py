import os
from config import config

from localizer import Localizer
from retrieval import Retrieval

#DATA_DIR='/Users/mh46989/lewpealab/DATA/comcon/fmri/'
DATA_DIR='/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/'
#MASK_VOL='vt.nii.gz'
#MASK_VOL='vt_100vox_searchlight_mask.nii.gz'

LOC_PREFIX='comcon_localizer_epi_64ch' # test
RET_PREFIX='comcon_retrieval'

class Subject(object):
    def __init__(self, subject_id, data_dir=DATA_DIR, n_voxels=500,
                       mask=config.MASK_VOL, force_load=False, run_feature_selection=True, **kwargs):
        self.id = subject_id
        self.n_voxels = n_voxels
        self.data_dir = data_dir
        self.dir = os.path.join(data_dir,subject_id)
        self.mask = os.path.join(self.dir,'masks',mask)
        self.behav_dir = os.path.join(self.dir, 'behavior')
        self.results_dir = os.path.join(self.dir, 'results')

        if not os.path.exists(self.results_dir):
            os.mkdir(self.results_dir)

        if not os.path.exists(self.behav_dir):
            raise Exception("no behavior directory found: {}".format(self.behav_dir))

        self.load_localizer(run_feature_selection=run_feature_selection, **kwargs)
        # self.load_retrieval(force_load=force_load, **kwargs)

    def load_localizer(self, prefix=LOC_PREFIX, info_file=None, force_load=False, run_feature_selection=True, n_jobs=4):
        # localizer info file
        if info_file is None:
            fname = '{}_localizer.csv'.format(self.id)
            self.localizer_behav_file = os.path.join(self.behav_dir, fname)

        # localizer
        self.localizer = Localizer(self.dir, self.mask, run_prefix=prefix)
        self.localizer.load_metadata(self.localizer_behav_file, onset_col='global_onset')

        # feature selection 
        if run_feature_selection:
            self.localizer.feature_selection(self.n_voxels, method='svc')
            self.features = self.localizer.features

    def load_retrieval(self, prefix=RET_PREFIX, info_file=None, force_load=False, **kwargs):
        # retrieval info file
        if info_file is None:
            fname = '{}_retrieval.csv'.format(self.id)
            self.retrieval_behav_file = os.path.join(self.behav_dir, fname)

        # retrieval
        self.retrieval = Retrieval(self.dir, prefix, self.mask, force_load=force_load)
        self.retrieval.load_metadata(self.retrieval_behav_file, onset_cols=['scene_onset','position_onset'])

        # match features with localizer
        self.retrieval.features = self.localizer.features
