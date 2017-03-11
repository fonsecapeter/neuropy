import numpy as np


class TFsMRISubGroup(object):
    """Links images and labels"""

    def __init__(self, subjects):
        self.subjects = np.asarray(subjects)
        self.lazy_images = None
        self.lazy_labels = None
        self.lazy_description = None
        self.slide = 0

    @property
    def shape(self):
        img_shape = self.subjects[0].flat_image_size
        lbl_shape = self.subjects[0].label.shape
        return ((self.length, img_shape), (self.length, lbl_shape))

    @property
    def images(self):
        if self.lazy_images is not None:
            return self.lazy_images
        self.lazy_images = np.asarray([sub.image for sub in self.subjects])
        return self.lazy_images

    @property
    def labels(self):
        if self.lazy_labels is not None:
            return self.lazy_labels
        self.lazy_labels = np.asarray([sub.label for sub in self.subjects])
        return self.lazy_labels

    @property
    def length(self):
        return len(self.subjects)

    @property
    def description(self):
        if self.lazy_description:
            return self.lazy_description
        else:
            self.lazy_description = {}
            occurances = {}
            for sub in self.subjects:
                occurances.setdefault(sub.group, []).append(1)
            for group, occurances in occurances.items():
                self.lazy_description[group] = len(occurances)
            return self.lazy_description

    def shuffle(self):
        np.random.shuffle(self.subjects)

    def next_batch(self, batch_size):
        """Gets the next batch of labelled data

        Notes:
            Should be used with evenly divisible batch_size and epochs,
            but will just truncate last batch if uneven
        Returns:
            Tuple(images, labels)
        """
        if self.slide + batch_size >= self.length:
            subs = self.subjects[self.slide:]
            self.slide = 0
        else:
            batch_end = self.slide + batch_size
            subs = self.subjects[self.slide:batch_end]
            self.slide = batch_end
        images = np.asarray([sub.image for sub in subs])
        labels = np.asarray([sub.label for sub in subs])
        return images, labels

    def random_batch(self, batch_size):
        """Gets next batch of random labelled data

        Returns:
            Tuple(images, labels)
        """
        idx = np.random.randint(0, self.length, size=batch_size)
        images = np.asarray([sub.image for sub in self.subjects[idx]])
        labels = np.asarray([sub.label for sub in self.subjects[idx]])
        return images, labels
