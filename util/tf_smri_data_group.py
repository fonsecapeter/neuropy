import numpy as np


class TFsMRIDataGroup(object):
    """Links images and labels"""

    def __init__(self, images, labels, subjects):
        self.images = images
        self.labels = labels
        self.subjects = subjects
        self.description = None

    @property
    def length(self):
        return len(self.labels)

    @property
    def shape(self):
        return (self.images.shape, self.labels.shape)

    @property
    def describe(self):
        if self.description:
            return self.description
        else:
            self.description = {}
            occurances = {}
            for sub in self.subjects:
                occurances.setdefault(sub.group, []).append(1)
            for group, occurances in occurances.items():
                self.description[group] = len(occurances)
            return self.description


    def next_batch(self, batch_size):
        """Gets next batch of random labelled data

        Returns:
            Tuple(images, labels)
        """
        idx = np.random.randint(0, self.length, size=batch_size)
        return self.images[idx, :], self.labels[idx, :]
