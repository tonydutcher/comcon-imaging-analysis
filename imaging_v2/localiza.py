import os
import nibabel as nb
import numpy as np
import pandas as pd
import glob
from nilearn.input_data import NiftiMasker

from nilearn.masking import apply_mask
from nilearn.image import new_img_like, mean_img
from nilearn.signal import clean
from nilearn.decoding import SearchLight
from sklearn.svm import SVC, SVR
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cross_validation import KFold

# project specifics
from config import config
from estimatas import RSAClassifier

# column names
item_label     = config.LOCALIZER_ITEM_LABEL
category_label = config.LOCALIZER_CATEGORY_LABEL
exposure_label = config.LOCALIZER_EXPOSURE_LABEL
nBetas = 480
nruns  = 8
TRs    = 2000

########################################
# localizer object - bulk of it all
########################################

class Localiza(object):
    def __init__(self, subject_dir, use_betas=False, mask=config.MASK_VOL, run_prefix=None):
        self.subject_dir  = subject_dir
        self.mask         = os.path.join(self.subject_dir, 'masks', mask)
        self.imgs_backup  = os.path.join(self.subject_dir, 'bold',
                                         'localizer_{}.nii.gz'.format(mask.split('.')[0]))
        self.features_dir = os.path.join(self.subject_dir, 'features', mask.split('.')[0])
        self.features     = None

        if use_betas is True:
            # mask object for betas (may or may not be used)
            self.masker_beta = NiftiMasker(
                mask_img=self.mask,
                standardize=False,
                detrend=False,
                memory="nilearn_cache",
                memory_level=5)
            self.masker_beta.fit()

            # load betas
            self.beta_dir     = os.path.join(self.subject_dir, 'trial_beta_estimation_combo/AllRuns')
            self.beta_objects = [ nb.load(r) for i,r in enumerate(glob.glob(os.path.join(self.beta_dir, "beta*"))) ]
            # hacked!!
            self.imgs    = clean( np.concatenate( 
                [ self.masker_beta.fit_transform(r) for i,r in enumerate(self.beta_objects[:nBetas]) ] ), 
            standardize=True
            )
            self.nframes      = self.imgs.shape[0]
            self.nfeatures    = self.imgs.shape[1]

        else:
            # mask object (may or may not be used)
            self.masker = NiftiMasker(
                mask_img=self.mask,
                standardize=True,
                detrend=True,
                memory="nilearn_cache",
                memory_level=5)
            self.masker.fit() 
            # load data
            if run_prefix is not None:
                self.find_runs(run_prefix)
                self.load_runs()

            self.set_fmri_properties()

    def find_runs(self, run_prefix=config.LOCALIZER_RUN_PREFIX):
        print "finding runs..."
        match_this = os.path.join(self.subject_dir, 'bold', "{}*".format(run_prefix))
        run_dirs = [d for d in glob.glob(match_this) if os.path.isdir(d)]
        print "found: ", run_dirs
        # load all runs
        self.runs = []
        for run in sorted(run_dirs):
            fname = self.run_volume(run)
            print "including vol: {}".format(fname)
            # load nifti
            img = nb.load(fname)
            self.runs.append(img)

    def run_volume(self, run_dir):
        fname = os.path.join(run_dir, config.FUNC_FILE)
        if not os.path.exists(fname):
            raise Exception("Cannot find volume: {}".format(fname))
        return fname

    def load_runs(self):
        print "loading data and preprocessing (if necessary)..." 
        self.imgs = clean(
                        np.concatenate(
                            [self.masker.fit_transform(r) #,
                             for i,r in enumerate(self.runs)]
                        ),
                        detrend = True,
                        standardize = True
                    )
		# imgs = np.concatenate(masker.fit_transform(r) for i,r in enumerate(runs)]

        # save properties
        self.nframes   = self.imgs.shape[0]
        self.nfeatures = self.imgs.shape[1]

    def set_fmri_properties(self, mri_shift=4, mri_nobvs=4):
        self.mri_shift = mri_shift
        self.mri_nobvs = mri_nobvs

    def load_metadata(self, metadata_file, onset_col):
        print "loading metadata..."

        self.info_file = metadata_file

        # load csv file containing metadata
        if not os.path.exists(metadata_file):
            raise Exception("Cannot find file: {}".format(metadata_file))

        self.info = np.recfromcsv(metadata_file, delimiter=',')

        # dictionary for all volume-based labels
        label = {}

        # align labels with volume shape
        for i,row in enumerate(self.info):
            ind = int(round(row[onset_col]))

            # if timepoint is outside of the current scope, skip.
            if ind > TRs -1: continue 

            # save all columns
            for col in self.info.dtype.names:
                # initialize if col doesn't exist
                if col not in label:
                    label[col]  = [np.nan for _ in range(TRs)]
                # fill in timepoint with label value
                label[col][ind] = row[col]

        # data type correction
        for key,val in label.iteritems():
            label[key] = np.array(val)

        for column in [item_label, category_label]:
            try:
                # change object id to str (after casting to int)
                is_nan = np.isnan(label[column]) 
                label[column] = label[column].astype(np.int).astype(np.str)
            except:
                # change object id to str
                label[column] = label[column].astype(np.str)
                is_nan = (label[column] == '') | (label[column] == 'nan')

            # create none category
            label[column][is_nan] = 'none'

        def make_set(key, ignore=[]):
            if label[key].dtype == np.number: 
                items = np.unique( label[key][~np.isnan(label[key])] )
            else:
                items = np.unique( label[key] )
            return set([x for x in items if x not in ignore])

        self.object_ids = make_set(item_label, ignore=['none'])
        self.categories = make_set(category_label, ignore=['none'])
        self.exposures  = make_set(exposure_label)
        self.label = label

    def feature_selection(self, use_betas=None, nvox=500, method='svc', label=item_label, use_backup=False, use_category=True, radius=8, **kwargs):
        if use_betas:
            print "using beta images after convolution for feature selection"
            self.get_beta_labels()
            self.imgs = self.imgs
        else:
            print " NOT, I repeat, NOT, using beta images after convolution for feature selection"

        print "feature selection (method={},voxels={})...".format(method,nvox)

        # ensure we have metadata to use for feature selection
        if not hasattr(self,'label') and self.label is not None:
            raise Exception('Unable to run feature selection. No metadata found.')

        cat_str = '_category' if use_category else ''
        if method=='searchlight':
            base_str = 'localizer_{}{}_{}_SVC_radius{}mm_{}vox'.format(label, cat_str, method, radius, nvox)
        else:
            base_str = 'localizer_{}{}_{}_{}vox'.format(label, cat_str, method, nvox)
        
        print base_str

        # paths
        self.vox_selection_base = os.path.join(
            self.features_dir,
            base_str)
        self.vox_selection_backup = '{}.npy'.format(self.vox_selection_base)
        self.vox_selection_scores = '{}.nii.gz'.format(self.vox_selection_base)
        self.vox_selection_mask   = '{}_mask.nii.gz'.format(self.vox_selection_base)

        if not os.path.exists(self.features_dir):
            os.makedirs(self.features_dir)

        if os.path.exists(self.vox_selection_backup) and use_backup:
            print "using backup feature selection (nvox={}; method={})".format(nvox,method)
            self.features = np.load(self.vox_selection_backup)

        elif method=='searchlight':
            self._feature_selection_rsa_searchlight(nvox, label, use_betas=use_betas,
                use_category=use_category, radius=radius, **kwargs)

            self.nfeatures = sum(self.features)

        else:
            self._feature_selection_classifier(nvox, method, label)

            # update num of features
            self.nfeatures = nvox

        # save to file
        np.save(self.vox_selection_backup, self.features)

    def _feature_selection_rsa_searchlight(self, nvox, label, use_betas=None, radius=4, folds=4, use_backup=False, n_jobs=4, use_category=False, expand_mask=True):
        # flat -> nii-like image
        # searchlight requires non-flat data, so we must transform (or load) 3D data
        print "flat img --> 3D nii-like image"
 
        #todo: loading masked backups is causing an error
        if use_backup and os.path.exists(self.imgs_backup):
            print "using backup imgs (masked)."
            #imgs = nb.load(self.imgs_backup)
        else:
            if use_betas:
                print "using beta images"
                imgs = self.masker_beta.inverse_transform(self.imgs)
                # object labels (id)
                labels = self.beta_labels[label]

            else:
                print "using non-estimated images"
                imgs = self.masker.inverse_transform(self.imgs)
                # object labels (id)
                labels = self.get_labels(label, 2)
                #nb.save(imgs, self.imgs_backup)

        if use_category:
            # categories
            if use_betas:
                cats = self.beta_labels[category_label]
            else:
                cats = self.get_labels(category_label, 2)
            # create RSA classifier (item & category)
            rsa = RSAClassifier(self.subject_dir, self.mask, labels, cats)
            print "used category"

        else:
            rsa = RSAClassifier(self.subject_dir, self.mask, labels)
            print "did not use category"

        # searchlight using RSA
        print "running RSA searchlight analysis..."
        print " - label =", label
        print " - radius =", radius
        print " - mask =", self.mask
        print " - cross val. folds =", folds
        print " - threads =", n_jobs

        kfold = KFold(self.imgs.shape[0], n_folds=folds)
        print "KFold done"
        searchlight = SearchLight(self.mask, radius=radius, verbose=2, n_jobs=n_jobs, estimator='svc', cv=kfold,  process_mask_img=self.mask)
        print "searchlight set-up done"
        print imgs.shape, len(labels)
        searchlight.fit(imgs, labels)
        self.searchlight = searchlight
        print "searchlight FIT done"

        scores = searchlight.scores_not_3D
        scores_3D = searchlight.scores_
        self.scores_3D = scores_3D
        self.scores = scores
        print "searchlight scores"

        coordinates = searchlight.scores_coordinates
        process_mask = searchlight.process_mask
        self.coordinates = coordinates
        self.process_mask = process_mask
        print "searchlight coordinates and process mask!"
        
        imgs_mean = mean_img(imgs)
        scores_img = new_img_like(imgs_mean, scores)
        scores_img.to_filename(self.vox_selection_scores)
        print "saving searchlight score image map"

        #threshold scores 
        # lets get the top spheres
        thresh = scores.take(scores.argsort()[::-1][nvox])
        feats = (scores>thresh) & (scores!=0)

        #extract spheres of thresholded scores
        select = np.hstack([np.array([0])]*len(scores))
        for i,ind in enumerate(coordinates):
            if feats[i]:
                sphere=map(int, np.array(coordinates[i]))
                select[sphere] = 1
            else:
                pass

        #print the number of features as a check.
        print select, sum(select)
        
        #put thresholded spheres back into 3D mask
        thresh_scores_3D=np.zeros(process_mask.shape)
        thresh_scores_3D[process_mask]=select
        thresh_scores_img=new_img_like(imgs_mean, thresh_scores_3D)
        thresh_scores_img.to_filename(self.vox_selection_mask)

        # flat features
        self.features = select==1

    def _feature_selection_classifier(self, nvox=500, method='svc', use_betas=None, label=category_label):

        if use_betas:
            labels = np.array(self.beta_labels['object_id'], dtype=np.str)
        else:
            labels = np.array(self.get_labels(label, 2), dtype=np.str)

        if method=='svc':
            meth = SVC(kernel='linear')
        elif method=='lda':
            meth = LDA()

        # threshold
        variance_threshold = VarianceThreshold(threshold=.01)
        # select features
        feature_select = SelectKBest(f_classif, k=nvox)
        # create anova pipeline
        anova_svc = Pipeline([('thresh', variance_threshold),
                              ('anova',  feature_select), 
                              (method,   meth)])
        # fit model
        anova_svc.fit(self.imgs, labels)
        # save mask of feature selection
        self.features = feature_select.get_support()
        print "saving backup voxel selection...".format(nvox,method)
        # save features to file
        np.save(self.vox_selection_backup, self.features)


    def get_imgs(self):
        # mask by features if available
        if self.features is not None:
            return self.imgs[:,self.features]

        # otherwise return all voxels
        return self.imgs

    def get_labels(self, label, duration):
        """ return specified metadata, shifted and aligned """
        select = self.label[label]
        duration=duration

        if select.dtype.type is np.string_:
            none = 'none'
        else:
            none = np.nan

        select = np.hstack([ [none]*self.mri_shift, select[:-self.mri_shift]])

        if none == 'none':
            base = np.where(select!=none)[0]
        else:
            base = np.where(~np.isnan(select))[0]

        for i in range(base[0], self.nframes):
            if select[i]!='none':
                obj = select[i]
                dur = duration

            if select[i]=='none' and dur!=0:
                select[i]=obj
                dur = dur-1
            else:
                pass
        return select

    def get_beta_labels(self):

        # dictionary for all volume-based labels
        labels = {}

        for i,row in enumerate(self.info):
            for col in self.info.dtype.names:
                if col not in labels:
                    labels[col]  = [np.nan for _ in range(480)]
                labels[col][i] = row[col]

        # data type correction
        for key,val in labels.iteritems():
            labels[key] = np.array(val)

        for column in [item_label, category_label]:
            try:
                # change object id to str (after casting to int)
                labels[column] = labels[column].astype(np.int).astype(np.str)
            except:
                # change object id to str
                labels[column] = labels[column].astype(np.str)

        self.beta_labels = labels

    def mean_pattern(self, **kwargs):
        """ props = dict(key,val) of labels/values for pattern """
        assert len(kwargs.keys()) > 0, "need arguments for pattern selection"

        select=None
        # find logical conjunction of labels whilst adjusting for shift
        for key, value in kwargs.iteritems():
            assert key in self.label, "key not found in label list"
            mp_labels=self.get_labels(key, duration=2)
            selectvals = mp_labels == value
            if select is None:
                select = selectvals
            else:
                select = select & selectvals

        # filtered image
        if self.features is not None:
            feat_imgs = self.imgs[:, self.features]
            spec_img = feat_imgs[select]
        else:
            spec_img = self.imgs[select]

        # return mean (spatial)
        return spec_img.mean(axis=0)

    def mean_beta_pattern(self, **kwargs):
        """ props = dict(key,val) of labels/values for pattern """
        assert len(kwargs.keys()) > 0, "need arguments for pattern selection"

        select=None
        # find logical conjunction of labels whilst adjusting for shift
        for key, value in kwargs.iteritems():
            assert key in self.beta_labels, "key not found in label list"
            
            selectvals = self.beta_labels[key] == value
            
            if select is None:
                select = selectvals
            else:
                select = select & selectvals

        # filtered image
        if self.features is not None:
            beta_feat_imgs = self.imgs[:, self.features]
            spec_img = beta_feat_imgs[select]
        else:
            spec_img = self.imgs[select, :]

        # return mean (spatial)
        return spec_img.mean(axis=0)

    def compare_category_betas(self):
        nids =  len(self.categories)
        pattern = np.zeros([nids, self.nfeatures])
        ind = 0
        for category in self.categories:
            pattern[ind] = self.mean_beta_pattern(category=category)
            ind += 1
        return pd.DataFrame(np.corrcoef(pattern), columns=self.categories)

    def compare_runs_betas(self):
        nrun =  nruns
        pattern = np.zeros([nrun, self.nfeatures])
        ind = 0
        for RUN in range(1,nrun+1):
            pattern[ind] = self.mean_beta_pattern(run=RUN)
            ind += 1
        return pd.DataFrame(np.corrcoef(pattern), columns=range(1,nrun+1))

    def compare_category_exposures(self):
        ncat = len(self.categories)
        nexp = len(self.exposures)
        if ncat==0 or nexp==0: return None

        pattern = np.zeros([ncat*nexp, self.nfeatures])
        label = []
        ind = 0
        for cat in self.categories:
            for exp in self.exposures:
                label.append('{}-{}'.format(cat,exp))
                #pattern[ind] = self.mean_category_pattern(cat, exp)
                pattern[ind] = self.mean_pattern(category=cat, exposure=exp)
                ind += 1

        return pd.DataFrame(np.corrcoef(pattern), columns=label)

    def compare_category_run(self):
        ncat = len(self.categories)
        nrun = len(self.runs)
        print 'number of runs', nrun
        if ncat==0 or nrun==0: return None

        pattern = np.zeros([ncat*nrun, self.nfeatures])
        label = []
        ind = 0
        for cat in self.categories:
            for run in range(1,nrun+1):
                label.append('{}-{}'.format(cat,run))
                #pattern[ind] = self.mean_category_pattern(cat, exp)
                pattern[ind] = self.mean_pattern(category=cat, run=run)
                ind += 1

        return pd.DataFrame(np.corrcoef(pattern), columns=label)

    def compare_category_runs_betas(self):
        ncat = len(self.categories)
        nrun = nruns
        print 'number of runs', nrun
        if ncat==0 or nrun==0: return None

        pattern = np.zeros([ncat*nrun, self.nfeatures])
        print "pattern dimensions are:", pattern.shape
        label = []
        ind = 0
        for cat in self.categories:
            for run in range(1,nrun+1):
                label.append('{}-{}'.format(cat,run))
                pattern[ind] = self.mean_beta_pattern(category=cat, run=run)
                ind += 1

        return pd.DataFrame(np.corrcoef(pattern), columns=label)


    def compare_objects(self):
        n_objs =  len(self.object_ids)
        n_vox  =  self.get_imgs().shape[1]

        # try to sort the object_ids, if possible
        try:
            obj_ids = map(str, sorted(map(int, self.object_ids)))
        except:
            obj_ids = self.object_ids

        # get patterns for each object
        pattern = np.zeros([n_objs, n_vox])
        for i,obj in enumerate(obj_ids):
            pattern[i,:] = self.mean_pattern(object_id=obj)

        # # break up matrix (solve numpy async issues)
        # chunk_size = 20
        # strides = zip(np.arange(0, n_objs, chunk_size),
        #               np.concatenate([ np.arange(chunk_size, n_objs, chunk_size), [n_objs] ]) )

        # # correlate neural patterns for subset of objects at a time
        # corr = np.zeros([n_objs, n_objs])
        # for i,j in strides:
        #     for k,l in strides:
        #         # correlations for chunk
        #         subi = np.concatenate([np.arange(i,j),np.arange(k,l)])
        #         pat = pattern[subi]
                # corr[i:j,k:l] = np.corrcoef(pat)[:chunk_size,chunk_size:]
        
        corr = np.corrcoef(pattern)

        # return dataframe of correlations
        return pd.DataFrame(corr, columns=obj_ids, index=obj_ids)

    def compare_objects_betas(self):
        n_objs =  len(self.object_ids)
        n_vox  =  self.get_imgs().shape[1]

        # try to sort the object_ids, if possible
        try:
            obj_ids = map(str, sorted(map(int, self.object_ids)))
        except:
            obj_ids = np.unique(self.object_ids)

        # get patterns for each object
        pattern = np.zeros([n_objs, n_vox])
        for i,obj in enumerate(obj_ids):
            pattern[i,:] = self.mean_beta_pattern(object_id=obj)

        # break up matrix (solve numpy async issues)
        chunk_size = 20
        strides = zip(np.arange(0, n_objs, chunk_size),
                      np.concatenate([ np.arange(chunk_size, n_objs, chunk_size), [n_objs] ]) )

        # correlate neural patterns for subset of objects at a time
        corr = np.zeros([n_objs, n_objs])
        for i,j in strides:
            for k,l in strides:
                # correlations for chunk
                subi = np.concatenate([np.arange(i,j),np.arange(k,l)])
                pat = pattern[subi]
                corr[i:j,k:l] = np.corrcoef(pat)[:chunk_size,chunk_size:]

        # return dataframe of correlations
        return pd.DataFrame(corr, columns=obj_ids, index=obj_ids)

    def compare_across_runs_activation(self):
        nrun = len(self.runs)
        if nrun==0: return None

        pattern = np.zeros([nrun, self.nfeatures])
        label = []
        ind = 0
        for run in range(1,nrun+1):
            label.append('{}'.format(run))
            #pattern[ind] = self.mean_category_pattern(cat, exp)
            pattern[ind] = self.mean_pattern(run=run)
            ind += 1

        return pd.DataFrame(pattern, index=label)

    def compare_category_run_activation(self):
        ncat = len(self.categories)
        nrun = len(self.runs)
        if ncat==0 or nrun==0: return None

        pattern = np.zeros([ncat*nrun, self.nfeatures])
        label = []
        ind = 0
        for cat in self.categories:
            for run in range(1,nrun+1):
                label.append('{}-{}'.format(cat,run))
                #pattern[ind] = self.mean_category_pattern(cat, exp)
                pattern[ind] = self.mean_pattern(category=cat, run=run)
                ind += 1

        return pd.DataFrame(pattern, index=label)


    def compare_category_run_activation(self):
        ncat = len(self.categories)
        nrun = len(self.runs)
        if ncat==0 or nrun==0: return None

        pattern = np.zeros([ncat*nrun, self.nfeatures])
        label = []
        ind = 0
        for cat in self.categories:
            for run in range(1,nrun+1):
                label.append('{}-{}'.format(cat,run))
                #pattern[ind] = self.mean_category_pattern(cat, exp)
                pattern[ind] = self.mean_pattern(category=cat, run=run)
                ind += 1

        return pd.DataFrame(pattern, index=label)
