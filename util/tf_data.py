from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import random

from util.progress import print_progress

t1_not_found_message = '  !!! %s T1 not found, skipping !!!'


class TFsMRIDataGroup(object):
    """Links images and labels"""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.slide = 0

    @property
    def length(self):
        return len(self.labels)

    @property
    def shape(self):
        return (self.images.shape, self.labels.shape)

    def next_batch(self, batch_size):
        """Gets next batch of random labelled data

        Returns:
            Tuple(images, labels)
        """
        idx = np.random.randint(0, self.length, size=batch_size)
        return self.images[idx, :], self.labels[idx, :]


class TFsMRIDataSet(TFsMRIDataGroup):
    """Splits data into train test and validation"""

    def __init__(self, parts_dir, data_dir, np_dir, image_shape, pixel_depth, label_map):
        self.parts_dir = parts_dir
        self.data_dir = data_dir
        self.np_dir = np_dir
        self.image_shape = image_shape
        self.pixel_depth = pixel_depth
        self.label_map = label_map
        self.rev_label_map = {str(one_hot): label for label, one_hot in label_map.items()}
        self.data_multiplier = 1
        self.quiet = False
        self.images = None
        self.labels = None
        self.participants = None

    def quiet_or_print(self, statement):
        if not self.quiet:
            print(statement)

    def normalized(self, data):
        return (data - self.pixel_depth / 2) / self.pixel_depth

    def load_or_save_image(self, p_id, p_class):
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
                img = self.normalized(img_data.flatten())
                np.save('%s/%s.npy' % (self.np_dir, p_id), img)
            except FileNotFoundError:
                self.quiet_or_print(t1_not_found_message % p_id)
                return None
        return img

    def slice_data(self, test_size=0.1):
        self.quiet_or_print('slicing data...')
        image_dataset, image_validation = train_test_split(self.images, test_size=test_size)
        image_train, image_test = train_test_split(image_dataset, test_size=test_size)

        label_dataset, label_validation = train_test_split(self.labels, test_size=test_size)
        label_train, label_test = train_test_split(label_dataset, test_size=test_size)

        self.validation = TFsMRIDataGroup(image_validation, label_validation)
        self.train = TFsMRIDataGroup(image_train, label_train)
        self.test = TFsMRIDataGroup(image_test, label_test)

    def load_data(self, quiet=False, data_multiplier=1):
        self.quiet = quiet
        self.data_multiplier = data_multiplier
        self.participants = np.genfromtxt(self.parts_dir, dtype=str, usecols=(0, 1), skip_header=1)
        try:
            self.images = np.load('%s/images_%sx.npy' % (self.np_dir, self.data_multiplier))
            self.labels = np.load('%s/labels_%sx.npy' % (self.np_dir, self.data_multiplier))
        except FileNotFoundError:
            images = []
            labels = []
            total = len(self.participants) - 1
            for idx, participant in enumerate(self.participants):
                try:
                    p_id = participant[0]
                    p_class = participant[1]
                    for _ in range(self.data_multiplier):
                        img = self.load_or_save_image(p_id, p_class)
                        if img is not None:
                            images.append(img)
                            labels.append(self.label_map[p_class])
                    if not self.quiet:
                        print_progress(idx, total, prefix='loading images:', length=40)
                except FileNotFoundError:
                    self.quiet_or_print(t1_not_found_message % p_id)
            self.quiet_or_print('building image set...')
            self.images = np.asarray(images)
            # np.save('%s/images_%sx' % (self.np_dir, self.data_multiplier), self.images)
            self.labels = np.asarray(labels)
            # np.save('%s/labels_%sx' % (self.np_dir, self.data_multiplier), self.labels)
        self.slice_data()

    def visualize(self, count, slice_num):
        indices = random.sample(range(0, self.length), count)
        fig = plt.figure()
        for idx in range(count):
            img = fig.add_subplot(1, count, idx + 1)
            plt.imshow(self.images[indices[idx]].reshape(self.image_shape)[slice_num])
            label = self.rev_label_map[str(self.labels[indices[idx]])]
            # for name, one_hot in self.label_map.items():
            #     if np.array_equal(one_hot, self.labels[indices[idx]]):
            #         label = name
            img.set_title(label)
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def inspect(self, idx, slice_cuts=[80, 100, 120, 140, 160, 180, 200]):
        fig = plt.figure()
        image = self.images[idx]
        p_id, p_class = self.participants[idx][0:2]
        for j, slice_num in enumerate(slice_cuts):
            img = fig.add_subplot(1, len(slice_cuts), j + 1)
            plt.imshow(image.reshape(self.image_shape)[slice_num])
            img.set_title('-'.join([p_class, p_id]))
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def inspect_all(self):
        for idx in range(0, self.length):
            self.inspect(idx)
