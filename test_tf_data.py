import numpy as np
from tf_data import TFDataGroup, TFDataSet
from unittest import TestCase

images = np.array([
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]],
    [[0, 1, 0],
     [1, 0, 1],
     [0, 1, 0]],
    [[1, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 1, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 1],
     [0, 0, 0],
     [0, 1, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
    [[0, 0, 0],
     [0, 0, 0],
     [0, 0, 0]],
])

labels = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
])


class TestTFDataGroup(TestCase):
    def setUp(self):
        self.data = TFDataGroup(images, labels)

    def test_images(self):
        assert np.array_equal(self.data.images, images)

    def test_labels(self):
        assert np.array_equal(self.data.labels, labels)


class TestTFDataSet(TestCase):
    """Ensure train_test_split

    trust it works, but make sure it's used in a good enough way
    """
    def setUp(self):
        self.dataset = TFDataSet(images, labels)
        self.train_image_shape = self.dataset.train.images.shape
        self.train_label_shape = self.dataset.train.labels.shape
        self.validation_label_shape = self.dataset.validation.labels.shape
        self.test_label_shape = self.dataset.test.labels.shape

    def test_train_sizes(self):
        assert self.train_image_shape[0] == self.train_label_shape[0]
        assert self.train_label_shape[1] == labels.shape[1]

    def test_proportions(self):
        assert self.test_label_shape[0] == self.validation_label_shape[0]
        assert self.train_label_shape[0] > self.test_label_shape[0]

    def test_overal(self):
        assert np.array_equal(self.dataset.labels, labels)
        assert np.array_equal(self.dataset.images, images)


    # rename to test_print to print
    def notest_print(self):
        print('all images:\n', self.dataset.images)
        print('all labels:\n', self.dataset.labels)
        print('validation images:\n', self.dataset.validation.images)
        print('validation labels:\n', self.dataset.validation.labels)
        print('train images:\n', self.dataset.train.images)
        print('train labels:\n', self.dataset.train.labels)
        print('test images:\n', self.dataset.test.images)
        print('test labels:\n', self.dataset.test.labels)


