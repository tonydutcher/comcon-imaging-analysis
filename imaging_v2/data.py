import os
import getpass
import numpy as np
import pandas as pd
import scipy.stats as stats
from glob import glob
from signaldetection import dPrime

subjects = [
    'comcon_201602171',
    'comcon_201602172',
    'comcon_201602173',
    'comcon_201602174',
    'comcon_201602191',
    'comcon_201602201',
    'comcon_201602202',
    'comcon_201602203',
    'comcon_201602221',
    #'comcon_201602222', # UNUSABLE - did retrieval w/o instructions
    #'comcon_201602223', # SUPER low hitrate
    'comcon_201602224',
    'comcon_201602241',
    'comcon_201602242',
    'comcon_201602261',
    'comcon_201602262',
    'comcon_201602271',
]

exclude_subjects = [
]

dir_behavioral = "/Users/{}/Dropbox (LewPeaLab)/DATA/comcon/behav-results/". \
                 format(getpass.getuser())

# image similarity
img_similarities = pd.read_table('image_similarity.csv', sep=',')

# grab all possible subjects
#subjects = np.loadtxt('subjects.txt', dtype=np.str)

# exclude unwanted
subjects = [s for s in subjects if s not in exclude_subjects]


#######################################
# Helper Functions
#######################################

# helper to determine object subtype (per row)
def object_subtype(r):
    if (r['novel']==1):
        return 'novel'
    else:
        if (r['target_type']=='cousin'):
            posdiff = np.abs(r['target']-r['object_position'])
            if posdiff==1:
                second='sibling1'
            else:
                second='sibling2'
        else:
            second=r['object_type']
        return '{}-{}'.format( r['target_type'], second)

# near/far from target
def object_dist_type(r):
    if r['novel']==1:
        return 'novel'
    elif r['practiced']==0:
        return 'nonprac'
    else:
        posdiff = np.abs(r['target']-r['object_position'])
        if posdiff==0:
            return 'target'
        elif posdiff==1:
            return 'close'
        elif posdiff==2:
            return 'far'
        else:
            return 'unknown'
    """
    else:
        if (r['target_type']=='cousin'):
            posdiff = np.abs(r['target']-r['object_position'])
            if posdiff==1:
                return 'close'
            else:
                return 'far'
        elif r['object_type']=='sibling':
            return 'close'
        else:
            return 'far'
    """

# join encoding and retrieval data to recognition
def make_joined(sid, rdf, rsuffix=''):
    if rsuffix=='':
        lsuffix = '_recog'
    else:
        lsuffix = ''
    #retr_sid = retr_sid[retr_sid.catch_family!=1]
    return recog[sid].reset_index().merge(
                            rdf,
                            #retr[sid][retr[sid].trial_repeat==1],
                            how='left',
                            on='scene_id',
                            suffixes=[lsuffix,rsuffix],
                            sort=False
                        ).set_index('trial').sort_index()

# read csv files
def grab_results(subjects, task, index_col=None):
    sdata = {}
    for s in sorted(subjects):
        d = pd.read_table(
               os.path.join(dir_behavioral, s, "{}_{}.csv".format(s,task)),
               sep = ',',
               index_col=index_col)
        d['subject'] = s
        sdata.update({ s: d})
    return pd.Panel(sdata)

# pairwise image similarity look-up
def get_similarity(img1, img2):
    img2loc = img_similarities.columns.get_loc(img2)
    return img_similarities[img1][img2loc]

# absolute time difference from target (seconds)
def object_time_diff(r):
    opos = int(r['object_position'])
    otype = r['object_type']
    ttype = r['target_type']
    tpos  = r['target']
    tdiff = None
    if ttype=='cousin':
        if opos==1 or opos==3:
            tdiff= 7.5
        elif opos==2:
            tdiff= 4.5
    elif ttype=='sibling':
        if otype=='sibling':
            tdiff= 1.5
        elif tpos==2:
            tdiff= 4.5
        else:
            tdiff= 7.5
    if opos < tpos:
        tdiff = -tdiff
    return tdiff


#######################################
# Data
#######################################

### ENCODING

# grab behavioral results for all tasks
print "loading encoding..."
enc   = grab_results(subjects, 'encoding', 0)
enc.items.name = 'sid'
enc.major_axis.name = 'trial'

### RETRIEVAL

print "loading retrieval..."
retr  = grab_results(subjects, 'retrieval')
retr.items.name = 'sid'
retr.major_axis.name = 'trial'

# some fixes
for s in subjects:
    # missing basename target file
    enc.loc[s,:,'target_filename'] = \
            enc.loc[s,:,'target_file'].map(os.path.basename, na_action='ignore')

    # normalize retrieval RT (z-score for subject)
    rt = retr.loc[s,:,'probe_resp_rt']
    retr.loc[s,:,'probe_resp_rt_z'] = (rt-rt.mean())/rt.std(ddof=0)

### RECOGNITION

print "loading recognition..."
recog = grab_results(subjects, 'recognition')
recog.items.name = 'sid'
recog.major_axis.name = 'trial'

# join recognition with encoding and retrieval data
recog = pd.Panel.from_dict({s:make_joined(s,enc[s]) for s in subjects})
#recog = pd.Panel.from_dict({s:make_joined(s,retr[s],'_retr') for s in subjects})

# fixes
for s in subjects:
    recog.loc[s,:,'subject'] = s

    seen = recog.loc[s,:,'old']==1
    is_practiced = recog.loc[s,:,'practiced']==1
    is_old = seen[seen & is_practiced].index

    # missing values
    recog.loc[s,:,'practiced'] = 0
    recog.loc[s,is_practiced,'practiced'] = 1

    sdata = recog.loc[s,is_old,:]
    recog.loc[s,is_old,'time_diff_from_target'] = sdata.apply(object_time_diff, axis=1)

    # object subtype 
    sdata = recog.loc[s,is_old,:]
    recog.loc[s,is_old,'object_subtype'] = sdata.apply(object_subtype, axis = 1)

    # object subtype 
    sdata = recog.loc[s,:,:]
    recog.loc[s,:,'object_dist_type'] = sdata.apply(object_dist_type, axis = 1)

    # object is target?
    recog.loc[s,:,'object_target'] = 0
    recog.loc[s,is_old,'object_target'] = \
            recog.loc[s,is_old,:].apply(
                    lambda r: r['obj{}_target'.format(r['object_position'])], axis=1)

    # object-target similarity
    recog.loc[s,is_old,'object2target_similarity'] = \
            recog.loc[s,is_old,:].apply(lambda r: get_similarity(
                                                r['object_filename'],
                                                r['target_filename']), axis=1)
    # target-scene similarity
    recog.loc[s,is_old,'target2scene_similarity'] = \
            recog.loc[s,is_old,:].apply(lambda r: get_similarity(
                                                r['target_filename'],
                                                r['scene_filename']), axis=1)

    # object-scene similarity (doesn't matter if practiced)
    is_old = seen[seen].index
    recog.loc[s,is_old,'object2scene_similarity'] = \
            recog.loc[s,is_old,:].apply(lambda r: get_similarity(
                                                r['object_filename'],
                                                r['scene_filename']), axis=1)
    # object-other objects similarity
    recog.loc[s,is_old,'object2objects_similarity'] = \
            recog.loc[s,is_old,:].apply(
                    lambda r: r['obj{}_to_objs'.format(r['object_position'])], axis=1)


# HIT/HCH
print "compute accuracies measures..."
is_old = recog.loc[:,:,'old']==1
# compute hit
recog.loc[:,:,'hit'] = (is_old.values & (recog.loc[:,:,'object_resp']<=2).values).astype(int).T
# compute high-confidence hit
recog.loc[:,:,'hch'] = (is_old.values & (recog.loc[:,:,'object_resp']==1).values).astype(int).T

# make 2D dataframes
enc_df= enc.transpose(2,0,1).to_frame(filter_observations=False)
retr_df= retr.transpose(2,0,1).to_frame(filter_observations=False)
recog_df= recog.transpose(2,0,1).to_frame(filter_observations=False)

# save to csv
enc_df.reset_index().to_csv('encoding.csv')
retr_df.reset_index().to_csv('retrieval.csv')
recog_df.reset_index().to_csv('recognition.csv')

