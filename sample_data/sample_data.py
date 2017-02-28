import numpy as np
import nibabel as nib
import os

from tf_data import TFDataSet
from util.progress import print_progress

DATA_MULTIPLIER = 1


def quiet_or_print(statement, quiet):
    if not quiet:
        print(statement)


def normalized(data, pixel_depth):
    return (data - pixel_depth / 2) / pixel_depth


def load_or_save_image(data_dir, np_dir, p_id, pixel_depth, quiet):
    if not os.path.exists(np_dir):
        os.makedirs(np_dir)
    try:
        img = np.load('%s/%s.npy' % (np_dir, p_id))
    except FileNotFoundError:
        try:
            img_file = nib.load(data_dir % (p_id, p_id))
            img = normalized(img_file.get_data().flatten(), pixel_depth)
            np.save('%s/%s.npy' % (np_dir, p_id), img)
        except FileNotFoundError:
            quiet_or_print('  !!! %s npy not found, skipping !!!' % p_id, quiet)
            img = None
    return img


def load_cannabis(data_multiplier=DATA_MULTIPLIER, quiet=False):
    CANNABIS_PIXEL_DEPTH = 7828
    data_dir = 'sample_data/cannabis/sub-%s/ses-FU/anat/sub-%s_ses-FU_T1w.nii.gz'
    np_dir = 'sample_data/cannabis/.numpies'

    # one-hot class label
    classes = {'HC': [1, 0], 'CB': [0, 1]}
    # [[id, class]...]
    participants = np.genfromtxt('sample_data/cannabis/participants.tsv', dtype=str, usecols=(0, 1), skip_header=1)

    try:
        images = np.load('%s/images_%sx.npy' % (np_dir, data_multiplier))
        labels = np.load('%s/labels_%sx.npy' % (np_dir, data_multiplier))
    except FileNotFoundError:
        images = []
        labels = []
        total = len(participants) - 1
        for idx, participant in enumerate(participants):
            try:
                p_id = participant[0]
                p_class = participant[1]
                for _ in range(data_multiplier):
                    # multiply data by data_multiplier for kicks, don't try at home!
                    # import just followup b/c all are of the same shape (one baseline is off)
                    fu = load_or_save_image(data_dir, np_dir, p_id, CANNABIS_PIXEL_DEPTH, quiet)
                    if fu is not None:
                        images.append(fu)
                        labels.append(classes[p_class])
                if not quiet:
                    print_progress(idx, total, prefix='Loading images:', length=40)
            except FileNotFoundError:
                quiet_or_print('  !!! sub-%s image not found, skipping !!!' % p_id, quiet)
        images = np.array(images)
        np.save('%s/images_%sx' % (np_dir, data_multiplier), images)
        labels = np.array(labels)
        np.save('%s/labels_%sx' % (np_dir, data_multiplier), labels)
    quiet_or_print('slicing data...', quiet)
    return TFDataSet(np.array(images), np.array(labels))


def load_ucla(quiet=False):
    UCLA_PIXEL_DEPTH = 7828
    data_dir = 'sample_data/ucla/%s/anat/%s_T1w.nii.gz'
    np_dir = 'sample_data/cannabis/.numpies'

    classes = {'CONTROL': [1, 0, 0, 0], 'ADHD': [0, 1, 0, 0], 'BIPOLAR': [0, 0, 1, 0], 'SCHZ': [0, 0, 0, 1]}
    participants = np.genfromtxt('sample_data/ucla/participants.tsv', dtype=str, usecols=(0, 1), skip_header=1,)
    # return participants

    try:
        images = np.load('%s/images' % np_dir)
        labels = np.load('%s/labels' % np_dir)
    except FileNotFoundError:
        images = []
        labels = []
        total = len(participants) - 1
        for idx, participant in enumerate(participants):
            try:
                p_id = participant[0]
                p_class = participant[1]

                image = load_or_save_image(data_dir, np_dir, p_id, UCLA_PIXEL_DEPTH, quiet)
                if image is not None:
                    images.append(image)
                    labels.append(classes[p_class])
                if not quiet:
                    print_progress(idx, total, prefix='Loading images:', length=40)
            except FileNotFoundError:
                quiet_or_print('  !!! %s image not found, skipping !!!' % p_id, quiet)
        images = np.array(images)
        np.save('%s/images' % np_dir, images)
        labels = np.array(labels)
        np.save('%s/labels' % np_dir, labels)
    quiet_or_print('slicing data...', quiet)
    return TFDataSet(images, labels)


cannabis = load_cannabis
ucla = load_ucla
