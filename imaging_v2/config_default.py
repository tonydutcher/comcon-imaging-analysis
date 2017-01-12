
PROJECT='comcon'

# to find functional images
FUNC_FILE='bold_mcf_brain_hpass_dt_norm.nii.gz'
PAR_FILE='bold_mcf.nii.gz.par'

# MASK (found in subject_dir/masks/<MASK_VOL>)
#MASK_VOL='LOC_inf_25bin_ADJbin.nii.gz'
MASK_VOL='LOC_VTmask_ADJbin.nii.gz'
#MASK_VOL='LOC_VTmask_ADJbin_orig.nii.gz'
#MASK_VOL='tempoccfusi_phg_mni152_bin.nii.gz'
#MASK_VOL='vt.nii.gz'

# LOCALIZER
LOCALIZER_RUN_PREFIX     = 'comcon_localizer'
LOCALIZER_ITEM_LABEL     = 'object_id'
LOCALIZER_ITEM_NAME      = 'object_filename'
LOCALIZER_FAMILY_LABEL   = 'family_id'
LOCALIZER_CATEGORY_LABEL = 'category'
LOCALIZER_EXPOSURE_LABEL = 'exposure'

# find retrieval runs
RETRIEVAL_RUN_PREFIX='comcon_retrieval'