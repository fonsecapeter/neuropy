# dicom value is 0-512

import dicom
ds = dicom.read_file('pf2015/IM-0005-0001.dcm')
slice = ds.pixel_array

print('slice 1, shape:', slice.shape)
print(slice)
