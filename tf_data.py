from sklearn.model_selection import train_test_split
# using image as generic subject of interest


class TFDataGroup(object):
    """Links images and labels"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.slide = 0

    @property
    def length(self):
        return len(self.labels)

    def next_batch(self, batch_size):
        """Gets next batch of labelled data

        Assumes:
            batch_size stays the same & used for at most one full pass through data
        Returns:
            Tuple(images, labels)
        """
        batch_end = self.slide + batch_size
        if batch_end >= self.length:
            raise IndexError('Batch out of range')
        images = self.images[self.slide:batch_end]
        labels = self.labels[self.slide:batch_end]
        self.slide += batch_size
        return (images, labels)


class TFDataSet(TFDataGroup):
    """Splits data into train test and validation"""
    def __init__(self, images, labels):
        super().__init__(images, labels)
        image_dataset, image_validation = train_test_split(images, test_size=0.1)
        image_train, image_test = train_test_split(image_dataset, test_size=0.1)

        label_dataset, label_validation = train_test_split(labels, test_size=0.1)
        label_train, label_test = train_test_split(label_dataset, test_size=0.1)

        self.validation = TFDataGroup(image_validation, label_validation)
        self.train = TFDataGroup(image_train, label_train)
        self.test = TFDataGroup(image_test, label_test)
