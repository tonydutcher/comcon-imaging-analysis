import os
import argparse
import nibabel
import numpy as np
import pandas as pd
from glob import glob
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiMasker
from nilearn.signal import clean
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline

#hacks
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
FUNC_FILE  = 'bold_mcf_brain_hpass_dt_norm.nii.gz'
RUN_LEN    = 343
mri_shift  = 4

############################################################
#
# GET RETRIEVAL PHASE IMAGES
#
############################################################
#
# import the data from the retrieval trials
#   1. preprocess identical to localizer trials
#   2. get feature mask from localizer trails
#   3. function to call a mean pattern for specific scene and position cue
#
# find runs related to subject
#

class Retrieve(object):
    def __init__(self, subject_dir, run_prefix, mask, features=None, force_load=False):
        self.subject_dir = subject_dir
        self.mask = mask
        self.features = features
        self.backup = os.path.join(self.subject_dir, 'bold/retrieval.npy')

        run_dirs = self.find_runs(run_prefix)
        self.load_runs(run_dirs, force_load=force_load)

        self.set_fmri_properties()

    def set_fmri_properties(self, mri_shift=5, mri_nobvs=4):
        self.mri_shift = mri_shift
        self.mri_nobvs = mri_nobvs

    def find_runs(self, prefix):
        match_this = os.path.join(self.subject_dir, 'bold', "{}*".format(prefix))
        print "run search: ", match_this
        found = [d for d in glob(match_this) if os.path.isdir(d)]
        print "found: ", found
        return found

    def run_volume(self, run_dir):
        fname = os.path.join(run_dir, FUNC_FILE)
        if not os.path.exists(fname):
            raise Exception("Cannot find volume: {}".format(fname))
        return fname 
    
    def load_runs(self, run_dirs, force_load=False):
        self.runs = []

        # load all runs
        for run in sorted(run_dirs):
            fname = self.run_volume(run) 
            print "including vol: {}".format(fname)
            # load nifti
            img = nibabel.load(fname)
            self.runs.append(img)

        # check for backup (numpy)
        if os.path.exists(self.backup) and force_load:
            # load backup data
            print "loading backup: {}".format(self.backup)    
            self.imgs = np.load(self.backup)

        # otherwise, process nifti files
        else:
            # apply mask
            print "applying mask..."    
            self.masker = NiftiMasker(mask_img=self.mask, standardize=True, detrend=True, memory='nilearn_cache')

            # apply mask -> concatenate -> detrend, standardize
            self.imgs = clean(
                            np.concatenate(
                                [self.masker.fit_transform(r) for i,r in enumerate(self.runs)]
                            ),
                            detrend = True,
                            standardize = True
                        )

            print "saving (for later use)..."
            np.save(self.backup, self.imgs)

        # save properties
        self.nframes   = self.imgs.shape[0]
        self.nfeatures = self.imgs.shape[1]
    
    
    #load metadata - retrieval trial data
    def load_metadata(self, metadata_file, onset_cols):
        print "loading metadata..."

        self.info_file = metadata_file

        # load csv file containing metadata
        if not os.path.exists(metadata_file):
            raise Exception("Cannot find file: {}".format(metadata_file))
        self.info = np.recfromcsv(metadata_file, delimiter=',')

        # dictionary for all volume-based labels
        label = {}
        label['onset_type']  = [np.nan for _ in range(self.nframes)]

        # multiple onset types
        for onset_col in onset_cols:
            # align labels with volume shape
            for i,row in enumerate(self.info):
                # TODO: fix, hack. we need global onset column
                ind = int(round( (row['run']-1)*RUN_LEN + row[onset_col] ))

                # if timepoint is outside of the current scope, skip.
                if ind > self.nframes -1: continue 

                # save onset type
                label['onset_type'][ind] = onset_col

                # save all columns
                for col in self.info.dtype.names:
                    # initialize if col doesn't exist
                    if col not in label:
                        label[col]  = [np.nan for _ in range(self.nframes)]
                    # fill in timepoint with label value
                    label[col][ind] = row[col]

        # data type correction
        for key,val in label.iteritems():
            label[key] = np.array(val)

        self.label = label

    def get_scene_or_position_cues(self, cue, duration):
        """ return specified metadata, shifted and aligned """

        # specifies the upper limit of window for each
        if cue == 'scene' and duration>3:
            duration=3
        elif cue == 'position' and duration>6:
            duration=6
        elif cue == 'both' and duration>9:
            duration=9

        # sets the scenes variable to populate cues so we can match with family ids
        select = self.label['onset_type']
        SCENES = self.label['scene_id'].astype('string')
        select[select == 'nan'] = 'none'
        SCENES[SCENES == 'nan'] = 'none'

        # set variable to mark scene or position onset and shift data for fmri
        select = np.hstack([ ['none']*4, select[:-4] ])
        SCENES = np.hstack([ ['none']*4, SCENES[:-4] ])

        #vector of cues for specific scenes
        cues   = np.hstack([ ['none']*self.nframes ])

        #populate cues vector
        base   = np.where(select!='none')[0]
        if cue == 'both':
            cue='scene'

        if cue == 'scene':
            start = base[0]
        else:
            start = base[1]

        for i in range(start, self.nframes):
            if select[i] == '{}_onset'.format(cue):
                CUE = SCENES[i]
                cues[i] = CUE
                dur = duration-1
            elif dur != 0:
                cues[i] = CUE
                dur = dur-1
            else:
                pass 

        return cues

    def mean_pattern(self, cue, family_id, **kwargs):
        """ props = dict(key,val) of labels/values for pattern """
        
        if family_id not in np.unique(self.label['scene_id'].astype('string')):
            raise Exception("family/scene id not in retrieval phase")

        if cue == 'scene':
            cues = self.get_scene_or_position_cues('scene', 3)

        elif cue =='both':
            cues = self.get_scene_or_position_cues('both', 9)

        else:
            cues = self.get_scene_or_position_cues('position', 6)

        select = cues==str(family_id)

        if self.features is not None:
            feat_imgs = self.imgs[:, self.features]
            spec_imgs = feat_imgs[select]
        else:
            spec_imgs = self.imgs[select]

        return spec_imgs.mean(axis=0)

