from sklearn.model_selection import train_test_split
# using image as generic subject of interest

class TFDataGroup(object):
    """Links images and labels"""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels


class TFDataSet(TFDataGroup):
    """Splits data into train test and validation"""
    def __init__(self, images, labels):
        super().__init__(images, labels)
        image_dataset, image_validation = train_test_split(images, test_size = 0.1)
        image_train, image_test = train_test_split(image_dataset, test_size = 0.1)

        label_dataset, label_validation = train_test_split(labels, test_size = 0.1)
        label_train, label_test = train_test_split(label_dataset, test_size = 0.1)

        self.validation = TFDataGroup(image_validation, label_validation)
        self.train = TFDataGroup(image_train, label_train)
        self.test = TFDataGroup(image_test, label_test)


