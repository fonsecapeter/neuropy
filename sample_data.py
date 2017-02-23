import numpy as np
import nibabel as nib
from tf_data import TFDataSet
from util.progress import print_progress

# one-hot class label
classes = {'HC': [1, 0], 'CB': [0, 1]}

# [[id, class]...]
participants = np.genfromtxt('cannabis_sample_data/participants.tsv', dtype=str, usecols=(0, 1), skip_header=1)

images = []
labels = []
total = len(participants) - 1
for idx, participant in enumerate(participants):
    p_id = participant[0]
    p_class = participant[1]

    for _ in range(5):
        # multiply data by 5 for kicks, don't try at home!
        # import just  followup b/c all are of the same shape (one bl is off)
        fu = nib.load('cannabis_sample_data/sub-%s/ses-FU/anat/sub-%s_ses-FU_T1w.nii.gz' % (p_id, p_id))
        images.append(fu.get_data().flatten())
        labels.append(classes[p_class])
    print_progress(idx, total, prefix='Loading images:', length=40)

cannabis = TFDataSet(np.array(images), np.array(labels))
