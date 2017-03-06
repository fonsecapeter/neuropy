from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import random

from .subject import Subject
from .tf_smri_data_group import TFsMRIDataGroup
from util.progress import print_progress

t1_not_found_message = '  !!! %s T1 not found, skipping !!!'


class TFsMRIDataSet(object):
    """Splits data into train test and validation groups

    Notes:
        Assumes all data can fit in memory. Usage:
        1.) run #load_data() to find available images
        2.) now images and labels are loaded into memory and split into #train, #test, and #validation
    """

    def __init__(self, parts_dir, data_dir, np_dir, image_shape, pixel_depth, label_map):
        self.parts_dir = parts_dir
        self.data_dir = data_dir
        self.np_dir = np_dir
        self.image_shape = image_shape
        self.flat_image_size = np.product(image_shape)
        self.pixel_depth = pixel_depth
        self.label_map = label_map
        self.rev_label_map = {str(one_hot): label for label, one_hot in label_map.items()}
        self.num_labels = len(label_map.keys())
        self.data_multiplier = 1
        self.quiet = False
        self.images = None
        self.labels = None
        self.subjects = None

    @property
    def length(self):
        return len(self.subjects)

    def quiet_or_print(self, statement):
        """Log to console if not in quiet mode"""
        if not self.quiet:
            print(statement)

    # def normalized(self, data):
    #     """Should be done in pre-processing, but just in case"""
    #     return (data - self.pixel_depth / 2) / self.pixel_depth

    def load_or_save_image(self, p_id, p_class):
        """Find np pickeled image data array or read mri and save one"""
        if not os.path.exists(self.np_dir):
            os.makedirs(self.np_dir)
        try:
            img = np.load('%s/%s.npy' % (self.np_dir, p_id))
        except FileNotFoundError:
            try:
                img_file = nib.load(self.data_dir % (p_id, p_id))
                # img_file = nib.load(self.data_dir % (p_class, p_id))
                img_data = img_file.get_data()
                if img_data.shape != self.image_shape:
                    return None
                # img = self.normalized(img_data.flatten())
                img = img_data.flatten()
                np.save('%s/%s.npy' % (self.np_dir, p_id), img)
            except FileNotFoundError:
                self.quiet_or_print(t1_not_found_message % p_id)
                return None
        return img

    def slice_data(self, images, labels, test_size=0.1):
        """Slice data into train, test, and validation

        Notes:
            May want to add custom proportions later, but for now breakdown is:
                - 75% train
                - 15% test
                - 15% validation
            Doesn't necessarily guarantee same proportion of classes in each group...
        """
        self.quiet_or_print('slicing data...')
        x_train, x_test, y_train, y_test, sub_train, sub_test = train_test_split(images, labels, self.subjects, test_size=0.3)
        del images
        del labels
        x_test, x_val, y_test, y_val, sub_test, sub_val = train_test_split(x_test, y_test, sub_test, test_size=0.5)
        self.train = TFsMRIDataGroup(x_train, y_train, sub_train)
        self.test = TFsMRIDataGroup(x_test, y_test, sub_test)
        self.validation = TFsMRIDataGroup(x_val, y_val, sub_val)

    def load_data(self, quiet=False, data_multiplier=1):
        """Load available images to memory

        Notes:
            assumes dataset can fit in memory. This method will search through the participants
            tsv and maintain a synced list of flattened images (self.images), labels (self.labels),
            and subjects (self.subjects) where images are found.

            for testing purposes, with smaller datasets, data_multiplier can be used to duplicate
            images and simulate a larger dataset, but will definitely cause overfit in any learning
        """
        self.subjects = []
        self.quiet = quiet
        self.data_multiplier = data_multiplier
        study_participants = np.genfromtxt(self.parts_dir, dtype=str, usecols=(0, 1), skip_header=1)
        try:
            images = np.load('%s/images_%sx.npy' % (self.np_dir, self.data_multiplier))
            labels = np.load('%s/labels_%sx.npy' % (self.np_dir, self.data_multiplier))
            self.subjects = np.load('%s/subjects_%sx.npy' % (self.np_dir, self.data_multiplier))
        except FileNotFoundError:
            images = []
            labels = []
            total = len(study_participants) - 1
            for idx, participant in enumerate(study_participants):
                try:
                    sub = Subject(participant[0], participant[0])
                    sub.p_id = participant[0]
                    sub.group = participant[1]
                    for _ in range(self.data_multiplier):
                        img = self.load_or_save_image(sub.p_id, sub.group)
                        if img is not None:
                            images.append(img)
                            labels.append(self.label_map[sub.group])
                            self.subjects.append(sub)
                    if not self.quiet:
                        print_progress(idx, total, prefix='loading images:', length=40)
                except FileNotFoundError:
                    self.quiet_or_print(t1_not_found_message % sub.p_id)
            self.quiet_or_print('building image set...')
            images = np.asarray(images)
            np.save('%s/images_%sx' % (self.np_dir, self.data_multiplier), images)
            labels = np.asarray(labels)
            np.save('%s/labels_%sx' % (self.np_dir, self.data_multiplier), labels)
            np.save('%s/subjects_%sx' % (self.np_dir, self.data_multiplier), self.subjects)
        self.slice_data(images, labels)

    def visualize(self, count, slice_num):
        """Visualize a set of images at a single slice"""
        indices = random.sample(range(0, self.length), count)
        fig = plt.figure()
        for idx in range(count):
            img = fig.add_subplot(1, count, idx + 1)
            plt.imshow(self.images[indices[idx]].reshape(self.image_shape)[slice_num])
            title = str(self.subjects[indices[idx]])
            img.set_title(title)
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def inspect(self, idx, slice_cuts=[80, 100, 120, 140, 160, 180, 200]):
        """Inspect a single image at multiplce slice"""
        fig = plt.figure()
        image = self.images[idx]
        sub = self.subjects[idx]
        for j, slice_num in enumerate(slice_cuts):
            img = fig.add_subplot(1, len(slice_cuts), j + 1)
            plt.imshow(image.reshape(self.image_shape)[slice_num])
            img.set_title(str(sub))
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def inspect_all(self, slice_cuts=[80, 100, 120, 140, 160, 180, 200]):
        """Inspects all images sequentially in dataset"""
        for idx in range(0, self.length):
            self.inspect(idx)
