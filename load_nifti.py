# http://nipy.org/nibabel/nibabel_images.html
import nibabel as nib

img = nib.load('pf2015/t1.nii')
print('nifti shape:', img.shape)
print('datatype:', img.header.get_data_dtype())
print('voxel size:', img.header.get_zooms())
print('units:', img.header.get_xyzt_units(), 'but no time data for t1')
print('header:', img.header)
print('affine:', img.affine)

img_data = img.get_data()
print('image data shape == nifti shape:', img_data.shape == img.shape)
print('image data array:', img_data)
