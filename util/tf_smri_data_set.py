from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random

from .subject import Subject
from .tf_smri_sub_group import TFsMRISubGroup
from util.progress import print_progress

t1_not_found_message = '  !!! %s T1 not found, skipping !!!'


class TFsMRIDataSet(object):
    """Splits data into train test and validation groups

    Notes:
        Usage:
        1.) run #load_data() to find subjects with available images
        2.) now valid subjects are loaded into memory and split into #train, #test, and #validation
    """

    def __init__(self, parts_dir, data_dir, np_dir, image_shape, label_map, downsample=None):
        self.parts_dir = parts_dir
        self.data_dir = data_dir
        self.np_dir = np_dir
        self.image_shape = image_shape
        self.flat_image_size = np.product(image_shape)
        self.label_map = label_map
        self.num_labels = len(label_map.keys())
        self.downsample = downsample
        self.quiet = False
        self.images = None
        self.labels = None
        self.subjects = None

    @property
    def length(self):
        return len(self.subjects)

    @property
    def description(self):
        return {
            'train': self.train.description,
            'test': self.test.description,
            'validation': self.validation.description
        }

    def quiet_or_print(self, statement):
        """Log to console if not in quiet mode"""
        if not self.quiet:
            print(statement)

    def split_subs(self):
        """Slice data into train, test, and validation

        Notes:
            May want to add custom proportions later, but for now breakdown is:
                - 75% train
                - 15% test
                - 15% validation
            Doesn't necessarily guarantee same proportion of classes in each group...
        """
        self.quiet_or_print('slicing subjects...')
        sub_train, sub_test = train_test_split(self.subjects, test_size=0.3)
        sub_test, sub_val = train_test_split(sub_test, test_size=0.5)
        self.train = TFsMRISubGroup(sub_train)
        self.test = TFsMRISubGroup(sub_test)
        self.validation = TFsMRISubGroup(sub_val)

    def load_data(self, quiet=False):
        """Load available images to memory

        Notes:
            This method will search through the participants sv and look for each subject's image.
            All valid subjects are saved to self.sibjects, and split into train, test, and validation
            groups
        """
        self.subjects = []
        self.quiet = quiet
        study_participants = np.genfromtxt(self.parts_dir, dtype=str, usecols=(0, 1), skip_header=1, delimiter='\t')
        total = len(study_participants) - 1
        for idx, participant in enumerate(study_participants):
            sub = Subject(
                participant[0], participant[1],
                self.data_dir, self.np_dir,
                self.image_shape, self.label_map, self.downsample
            )
            if sub.image is not None:
                self.subjects.append(sub)
                if not self.quiet:
                    print_progress(idx, total, prefix='loading subjects:', length=40, left_cap='', right_cap='')
        self.split_subs()

    def visualize(self, count, slice_num):
        """Visualize a set of images at a single slice"""
        indices = random.sample(range(0, self.length), count)
        fig = plt.figure()
        for idx in range(count):
            sub = self.subjects[indices[idx]]
            img = fig.add_subplot(1, count, idx + 1)
            plt.imshow(sub.image.reshape(self.image_shape)[slice_num])
            img.set_title(str(sub))
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def inspect_all(self, slice_cuts=[80, 100, 120, 140, 160, 180, 200]):
        """Inspects all images sequentially in dataset"""
        for sub in self.subjects:
            sub.inspect(slice_cuts)
