import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from test_reactivation import testing

DATA_DIR='/Users/amd5226/Dropbox (LewPeaLab)/STUDY/ComCon/fmri/'
group_results_dir = os.path.join(DATA_DIR, 'group_results_nov2016')

subject_ids = [#'comcon_201603141', # dan's data is fuckkkkked.
               'comcon_201603142',
               'comcon_201603161',
               'comcon_201604211',
               'comcon_201604261',
               'comcon_201604281',
               'comcon_201605021']

scene, pos, both, ranks, patterns = testing(subject_ids)


### DATA

scene_df = pd.Panel(scene)
pos_df   = pd.Panel(pos)
both_df  = pd.Panel(both)
# rank_df  = pd.Panel(ranks)



### PREPROCESS DATA ---------

# turn tensor data into stack/concatenate data
dfS  = pd.DataFrame()
dfP  = pd.DataFrame()
dfB  = pd.DataFrame()
dfS_cat = pd.DataFrame()
dfP_cat = pd.DataFrame()
dfB_cat = pd.DataFrame()

for i,s in enumerate(subject_ids):
	dfS  = scene_df.loc[s,:,:].transpose()
	dfP  = pos_df.loc[s,:,:].transpose()
	dfB  = both_df.loc[s,:,:].transpose()
	dfS_cat = pd.concat((dfS_cat, dfS))
	dfP_cat = pd.concat((dfP_cat, dfP))
	dfB_cat = pd.concat((dfB_cat, dfB))


# GET ABSOLUTE DISTANCE
dfP_cat['abs_dist'] = abs(dfP_cat.loc[:,'dist'])
dfS_cat['abs_dist'] = abs(dfS_cat.loc[:,'dist'])
dfB_cat['abs_dist'] = abs(dfB_cat.loc[:,'dist'])

mask='corr-20spheres'

### PREPROCESS DATA ---------



##### RESEARCH QUESTIONS

#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
## 1. Targets should increase in ranking from scene cue to position cue.
# WHAT I NEED: the target position during the scene and position cue.

dfS_cat_T = dfS_cat.loc[dfS_cat.dist==0,:]
dfP_cat_T = dfP_cat.loc[dfP_cat.dist==0,:]


# all trials
mean_tarPos_S  = dfS_cat_T.loc[:, 'tarPos'].mean()
mean_tarPos_P  = dfP_cat_T.loc[:, 'tarPos'].mean()
diff_tarS_tarP = mean_tarPos_S-mean_tarPos_P
mean_tarPos_P
diff_tarS_tarP


# yes trials
mean_tarPos_S  = dfS_cat_T.loc[dfS_cat_T.retr_resp=='yes', 'tarPos'].mean()
mean_tarPos_P  = dfP_cat_T.loc[dfP_cat_T.retr_resp=='yes', 'tarPos'].mean()
diff_tarS_tarP = mean_tarPos_S-mean_tarPos_P
mean_tarPos_P
diff_tarS_tarP


# no trials
mean_tarPos_S  = dfS_cat_T.loc[dfS_cat_T.retr_resp=='no', 'tarPos'].mean()
mean_tarPos_P  = dfP_cat_T.loc[dfP_cat_T.retr_resp=='no', 'tarPos'].mean()
diff_tarS_tarP = mean_tarPos_S-mean_tarPos_P
mean_tarPos_P
diff_tarS_tarP



#plotting all target positions
sns.distplot(dfP_cat_T.loc[:, 'tarPos'], bins=range(0, 121, 10), color='b', label='Order')
fname = 'betas_{}_TargetPos_OrderCue.png'.format(mask)
plt.savefig(os.path.join(group_results_dir, fname))



#plotting target position by response during retrieval trial
sns.distplot(dfP_cat_T.loc[dfP_cat_T.retr_resp=='yes', 'tarPos'], bins=range(0, 121, 10), color='b', label='Order')
sns.distplot(dfP_cat_T.loc[dfP_cat_T.retr_resp=='no', 'tarPos'], bins=range(0, 121, 10), color='r', label='Order')
fname = 'betas_{}_by-resp_TargetPos_OrderCue.png'.format(mask)
plt.savefig(os.path.join(group_results_dir, fname))




#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
## 2. Is there a negative relationship between temporal distance and rank during the position cue?
#
#

def object_time_diff(r):
    opos  = r['object_pos']
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
        elif int(tpos)==2:
            tdiff= 4.5
        else:
            tdiff= 7.5
    if opos < int(tpos):
        tdiff = -tdiff
    return tdiff

# # get short, medium, and long distances
sdataP = dfP_cat.loc[dfP_cat.loc[:,'dist']!=0, :]
sdataB = dfB_cat.loc[dfB_cat.loc[:,'dist']!=0, :]
sdataS = dfS_cat.loc[dfS_cat.loc[:,'dist']!=0, :]

dfP_cat['time_diff_from_target']=0
dfP_cat.loc[dfP_cat.loc[:,'dist']!=0,'time_diff_from_target'] = sdataP.apply(object_time_diff, axis=1)

dfP_cat_NT = dfP_cat.loc[dfP_cat.dist!=0,:]
dfP_cat_NT['weighted_memory_score']=((dfP_cat_NT.loc[:,'recog_resp']==1).values.astype(int) + (dfP_cat_NT.loc[:,'recog_resp']==2).values.astype(int) * 0.66 + (dfP_cat_NT.loc[:,'recog_resp']==3).values.astype(int) * 0.33).T
dfP_cat_NT['HCH'] = (dfP_cat_NT.loc[:,'recog_resp']==1).astype(int).T
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==-1.5, 'time_diff']='short'
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==1.5, 'time_diff']='short'
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==-4.5, 'time_diff']='medium'
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==4.5, 'time_diff']='medium'
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==-7.5, 'time_diff']='long'
dfP_cat_NT.loc[dfP_cat_NT.loc[:,'time_diff_from_target']==7.5, 'time_diff']='long'



dd = dfP_cat_NT
#dd_yes = dfP_cat_NT.loc[dfP_cat_NT.retr_resp=='yes', :]
title = '{} - absolute reactivation distances vs. encoding distances'.format(mask)
fname = '{}_reactivation_distances_vs_encoding_distances.png'.format(mask)
fig, ax = plt.subplots(figsize=(7,10))
ax = sns.barplot(data=dd, x='time_diff', order=['short', 'medium', 'long'], y='abs_dist', ax=ax, color='.6')
ax = sns.stripplot(data=dd, x='time_diff', order=['short', 'medium', 'long'], y='abs_dist', ax=ax, jitter=True, color=".3", alpha=0.6)
ax.set(xlabel=" ", ylabel=" ")
plt.title(title)
plt.savefig(os.path.join(group_results_dir, fname))



#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
#####################################################################################
## 3. Is the rank position (or difference from target) of non-target to position cue predictive of memory performance.
#
#### FIRST LOOK AT RELATIONSHIP BETWEEN MEMORY AND REACTIVATION DISTANCES BINNED
dd = dfP_cat_NT
dd.index = range(len(dd))

#GROUP REACITVATION SPLIT 6
one, two, three, four, five, six = np.array_split(np.argsort(dd['dist'].values), 6)
col='reactivation_distance_SIX'
dd[col] = "neg-far"
dd.loc[two,     col] = "neg-med"
dd.loc[three,   col] = "neg-near"
dd.loc[four,    col] = "pos-near"
dd.loc[five,    col] = "pos-med"
dd.loc[six,     col] = "pos-far"

#mean of each group
dd.loc[dd.reactivation_distance_SIX=='neg-far', 'dist'].mean()
dd.loc[dd.reactivation_distance_SIX=='neg-med', 'dist'].mean()
dd.loc[dd.reactivation_distance_SIX=='neg-near', 'dist'].mean()
dd.loc[dd.reactivation_distance_SIX=='pos-far', 'dist'].mean()
dd.loc[dd.reactivation_distance_SIX=='pos-med', 'dist'].mean()
dd.loc[dd.reactivation_distance_SIX=='pos-near', 'dist'].mean()

#histogram
dd.loc[dd.reactivation_distance_SIX=='neg-far', 'dist'].hist()
dd.loc[dd.reactivation_distance_SIX=='neg-med', 'dist'].hist()
dd.loc[dd.reactivation_distance_SIX=='neg-near', 'dist'].hist()
dd.loc[dd.reactivation_distance_SIX=='pos-far', 'dist'].hist()
dd.loc[dd.reactivation_distance_SIX=='pos-med', 'dist'].hist()
dd.loc[dd.reactivation_distance_SIX=='pos-near', 'dist'].hist()
plt.show()


#GROUP REACITVATION SPLIT 3
one, two, three = np.array_split(np.argsort(dd['abs_dist'].values), 3)
col='reactivation_distance_THREE'
dd[col] = "near"
dd.loc[two, col] = "medium"
dd.loc[three, col] = "far"

#sum in each group
dd.loc[dd.reactivation_distance_THREE=='near', 'abs_dist'].mean()
dd.loc[dd.reactivation_distance_THREE=='medium', 'abs_dist'].mean()
dd.loc[dd.reactivation_distance_THREE=='far', 'abs_dist'].mean()

#histogram
dd.loc[dd.reactivation_distance_THREE=='near', 'abs_dist'].hist()
dd.loc[dd.reactivation_distance_THREE=='medium', 'abs_dist'].hist()
dd.loc[dd.reactivation_distance_THREE=='far', 'abs_dist'].hist()
plt.show()


## PLOTTTING RELATIONSHIP BETWEEN MEMORY AND REACTIVATION DISTANCES BINNED
#
#
#
# PLOT GROUP REACTIVATION 6
title="{} - reactivation distance split 6 vs. memory score".format(mask)
col='reactivation_distance_SIX'
fig, ax = plt.subplots(figsize=(9,10))
# all retrieval trials
ax = sns.barplot(data=dd, x=col, order=['neg-far', 'neg-med', 'neg-near', 'pos-near', 'pos-med', 'pos-far'], y='HCH', ax=ax, color='.6')
# filtered by a 'yes' retrieval response
#ax = sns.barplot(data=dd.loc[dd.retr_resp=='yes', :], x=col, order=['neg-far', 'neg-med', 'neg-near', 'pos-near', 'neg-med', 'neg-far'], y='HCH', ax=ax, color='.6')
plt.title(title)
fname = '{}_reactivation_distance_split6_vs_memory_score.png'.format(mask)
fig.savefig(os.path.join(group_results_dir, fname))



# HCH_three1 = dd.loc[dd.HCH==1, 'reactivation_distance_THREE']
# HCH_three0 = dd.loc[dd.HCH==0, 'reactivation_distance_THREE']
# sum(HCH_three1=='near')
# sum(HCH_three1=='medium')
# sum(HCH_three1=='far')

# sum(HCH_three0=='near')
# sum(HCH_three0=='medium')
# sum(HCH_three0=='far')


# HCH_six1 = dd.loc[dd.HCH==1, 'reactivation_distance_SIX']
# HCH_six0 = dd.loc[dd.HCH==0, 'reactivation_distance_SIX']
# [, , , , , ]
# sum(HCH_six1=='neg-far')
# sum(HCH_six1=='neg-med')
# sum(HCH_six1=='neg-near')
# sum(HCH_six1=='pos-near')
# sum(HCH_six1=='neg-med')
# sum(HCH_six1=='neg-far')


# PLOT GROUP REACTIVATION 3
title="{} - reactivation distance split 3 vs. memory score".format(mask)
col='reactivation_distance_THREE'
fig, ax = plt.subplots(figsize=(9,10))

# all retrieval trials
ax = sns.barplot(data=dd, x=col, order=['near', 'medium', 'far'], y='HCH', ax=ax, color='.6')
ax.set(xlabel=" ", ylabel=" ")

# filtered by a 'yes' retrieval response
#ax = sns.barplot(data=dd.loc[dd.retr_resp=='yes', :], x=col, order=['near', 'medium', 'far'], y='HCH', ax=ax, color='.6')
plt.title(title)
fname = '{}_reactivation_distance_split3_vs_memory_score.png'.format(mask)
fig.savefig(os.path.join(group_results_dir, fname))

#### FIRST LOOK AT RELATIONSHIP BETWEEN MEMORY AND REACTIVATION DISTANCES BINNED










#### SECOND LOOK AT RELATIONSHIP BETWEEN MEMORY AND REACTIVATION DISTANCES
### --- LOGISTIC REGRESSION
## --- LINEAR


# make sure proper data type
df = dd.loc[:, ('HCH', 'abs_dist')]
np.asarray(df).dtype
df = df.astype(float)
np.asarray(df).dtype

# PLOTTING RELATIONSHIP
title = 'reactivation distance vs. memory score'
plt.figure(figsize=(8,10))
sns.set(style="darkgrid")
# all retrieval trials
ax = sns.jointplot(x="abs_dist", y="HCH", data=df, color="r", kind="reg", y_jitter=.03, logistic=True)
# filtered by a 'yes' retrieval response
#ax = sns.jointplot(x="abs_dist", y="HCH", data=df.loc[dd.retr_resp=='yes',:], color="r", kind="reg", y_jitter=.03, logistic=True)
plt.show()
fname='transect1_logistic_regression_reactivation_distance_vs_memory_score.png'
plt.savefig(os.path.join(group_results_dir, fname))


## --- NON-LINEAR
# make sure proper data type
df = dd.loc[:, ('HCH', 'dist')]
np.asarray(df).dtype
df = df.astype(float)
np.asarray(df).dtype


# PLOTTING RELATIONSHIP -> BELOW TARGET
dfrank = df.loc[df.dist<0, :]
title = 'reactivation distance vs. memory score'
plt.figure(figsize=(8,10))
sns.set(style="darkgrid")
ax = sns.jointplot(x="dist", y="HCH", data=dfrank, color="b", kind="reg", y_jitter=.03, logistic=True)
fname='transect1_logistic_regression_reactivation_distance_BELOW_target_vs_memory_score.png'
plt.savefig(os.path.join(group_results_dir, fname))


# PLOTTING RELATIONSHIP -> ABOVE TARGET
dfrank = df.loc[df.dist>0, :]
title = 'reactivation distance vs. memory score'
plt.figure(figsize=(8,10))
sns.set(style="darkgrid")
ax = sns.jointplot(x="dist", y="HCH", data=dfrank, color="r", kind="reg", y_jitter=.03, logistic=True)
fname='transect1_logistic_regression_reactivation_distance_ABOVE_target_vs_memory_score.png'
plt.savefig(os.path.join(group_results_dir, fname))













