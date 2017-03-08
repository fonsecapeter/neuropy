import numpy as np
from util.tf_smri_data_set import TFsMRIDataSet

cannabis = TFsMRIDataSet(
    'sample_data/cannabis/sorted_participants.tsv',
    # 'sample_data/cannabis/sub-%s/ses-FU/anat/sub-%s_ses-FU_T1w.nii.gz',
    'sample_data/cannabis/fsl/struc/%s-sub-%s_ses-FU_T1w_struc_GM.nii.gz',  # % (class_label, p_id)
    'sample_data/cannabis/.numpies',
    (256, 256, 170),
    {'HC': np.asarray([1, 0]), 'CB': np.asarray([0, 1])}
)

ucla = TFsMRIDataSet(
    'sample_data/ucla/participants.tsv',
    'sample_data/ucla/%s/anat/%s_T1w.nii.gz',
    'sample_data/ucla/.numpies',
    (176, 256, 256),
    {'CONTROL': np.asarray([1, 0, 0, 0]), 'ADHD': np.asarray([0, 1, 0, 0]),
     'BIPOLAR': np.asarray([0, 0, 1, 0]), 'SCHZ': np.asarray([0, 0, 0, 1])}
)

oasis = TFsMRIDataSet(
    'sample_data/oasis/participants.tsv',
    'sample_data/oasis/%s/mwrc1%s_mpr_anon_fslswapdim_bet.nii.gz',
    'sample_data/oasis/.numpies',
    # (91, 109, 91),
    (46, 55, 46), # downsample 4x
    {'HC': np.asarray([1, 0]), 'AD': np.asarray([0, 1])}
)

