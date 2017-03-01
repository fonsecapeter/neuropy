import numpy as np
from util.tf_data import TFsMRIDataSet

cannabis = TFsMRIDataSet(
    'sample_data/cannabis/participants.tsv',
    'sample_data/cannabis/sub-%s/ses-FU/anat/sub-%s_ses-FU_T1w.nii.gz',
    'sample_data/cannabis/.numpies',
    (256, 256, 170),
    7828,
    {'HC': np.asarray([1, 0]), 'CB': np.asarray([0, 1])}
)

ucla = TFsMRIDataSet(
    'sample_data/ucla/participants.tsv',
    'sample_data/ucla/%s/anat/%s_T1w.nii.gz',
    'sample_data/ucla/.numpies',
    (176, 256, 256),
    7828,
    {'CONTROL': np.asarray([1, 0, 0, 0]), 'ADHD': np.asarray([0, 1, 0, 0]),
     'BIPOLAR': np.asarray([0, 0, 1, 0]), 'SCHZ': np.asarray([0, 0, 0, 1])}
)
