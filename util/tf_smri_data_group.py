import numpy as np


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
