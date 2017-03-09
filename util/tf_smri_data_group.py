"""DEPRECATED PLZ USE TFsMRISubGroup"""
import numpy as np


class TFsMRIDataGroup(object):
    """Links images and labels"""

    def __init__(self, images, labels, subjects):
        self.images = images
        self.labels = labels
        self.subjects = subjects
        self.dscrptn = None

    @property
    def length(self):
        return len(self.labels)

    @property
    def shape(self):
        return (self.images.shape, self.labels.shape)

    @property
    def description(self):
        if self.dscrptn:
            return self.dscrptn
        else:
            self.dscrptn = {}
            occurances = {}
            for sub in self.subjects:
                occurances.setdefault(sub.group, []).append(1)
            for group, occurances in occurances.items():
                self.dscrptn[group] = len(occurances)
            return self.dscrptn

    def next_batch(self, batch_size):
        """Gets next batch of random labelled data

        Returns:
            Tuple(images, labels)
        """
        idx = np.random.randint(0, self.length, size=batch_size)
        return self.images[idx, :], self.labels[idx, :]
