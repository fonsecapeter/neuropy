import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os

t1_not_found_message = '  !!! %s T1 not found, skipping !!!'


class Subject(object):
    def __init__(self, p_id, group, data_dir, np_dir, image_shape, label_map):
        self.p_id = p_id
        self.group = group
        self.data_dir = data_dir
        self.np_dir = np_dir
        self.image_shape = image_shape
        self.flat_image_size = np.product(image_shape)
        self.label_map = label_map
        self.rev_label_map = {str(one_hot): label for label, one_hot in label_map.items()}
        self.label = label_map[group]
        self.img = None

    @property
    def image(self):
        return self.load_or_save_image()

    def load_or_save_image(self):
        """Find np pickeled image data array or read mri and save one

        Notes:
            - first checks memory
            - then checks for numpy file
            - then check for image file with double p_id (raw)
            - then check for image file with group-p_id (fsl pre-processed)
        """
        if self.img is not None:
            return self.img

        if not os.path.exists(self.np_dir):
            os.makedirs(self.np_dir)
        try:
            img = np.load('%s/%s.npy' % (self.np_dir, self.p_id))
        except FileNotFoundError:
            try:
                img_file = nib.load(self.data_dir % (self.p_id, self.p_id))
            except FileNotFoundError:
                try:
                    img_file = nib.load(self.data_dir % (self.group, self.p_id))
                except FileNotFoundError:
                    print(t1_not_found_message % self.p_id)
                    return None
            img_data = img_file.get_data()
            if img_data.shape != self.image_shape:
                return None
            img = img_data.flatten()
            np.save('%s/%s.npy' % (self.np_dir, self.p_id), img)
        self.img = img
        return img

    def inspect(self, slice_cuts=[10, 20, 30, 40, 50, 60, 70, 80]):
        """Inspect a single image at multiplce slice"""
        fig = plt.figure()
        for j, slice_num in enumerate(slice_cuts):
            img = fig.add_subplot(1, len(slice_cuts), j + 1)
            plt.imshow(self.image.reshape(self.image_shape)[slice_num])
            img.set_title(str(self))
            img.axes.get_xaxis().set_visible(False)
            img.axes.get_yaxis().set_visible(False)
        plt.show()

    def __str__(self):
        return '%s-%s' % (self.group, self.p_id)

    def __repr__(self):
        return '<Subject %s>' % str(self)
