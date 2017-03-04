# Random collection of deep learning with MRIs

Set up environment `virtualenv -p /usr/local/bin/python3.6 venv`

Install dependencies `pip install -r requirements.txt`

Activiate the environment `. venv/bin/activate`

Run the (very few) tests `python -m unittest util.test_tf_data.py`

Grab oasis dataset from `nilearn.datasets.fetch_oasis_vbm(data_dir='sample_data/oasis')`

Move the contents of `sample_data/oasis/oasis1` into `sample_data/oasis`

I already added a column to `oasis_cross-sectional.csv` as `participants.tsv` for groups according to the [specs](http://www.oasis-brains.org/pdf/oasis_cross-sectional_facts.pdf):
  - `HC` for `CDR` nonexistent or 0
  - `AD` for `CDR` 0.5 (mild), 1 (moderate), or 2 (severe)
These images come preprocessed from fsl for vbm studies, ie they're already normalized and warped to dartel space. (My fsl flow is below for raw datasets)

`python naive_oasis_softmax.py` batch size: 50
```
step 0, training accuracy: 0.75
step 100, training accuracy: 0.6875
step 200, training accuracy: 0.875
step 300, training accuracy: 0.875
step 400, training accuracy: 0.75
step 500, training accuracy: 0.625
step 600, training accuracy: 0.875
step 700, training accuracy: 0.8125
step 800, training accuracy: 0.75
step 900, training accuracy: 0.625
step 1000, training accuracy: 0.9375
step 1100, training accuracy: 0.875
step 1200, training accuracy: 0.75
step 1300, training accuracy: 0.8125
step 1400, training accuracy: 0.75
step 1500, training accuracy: 0.75
step 1600, training accuracy: 0.5625
step 1700, training accuracy: 0.8125
step 1800, training accuracy: 0.875
step 1900, training accuracy: 0.875
test accuracy: 0.789474
```
`python naive_oasis_softmax.py` batch size: 100
```
step 0, training accuracy: 0.75
step 100, training accuracy: 0.7
step 200, training accuracy: 0.72
step 300, training accuracy: 0.67
step 400, training accuracy: 0.57
step 500, training accuracy: 0.65
step 600, training accuracy: 0.78
step 700, training accuracy: 0.73
step 800, training accuracy: 0.74
step 900, training accuracy: 0.73
step 1000, training accuracy: 0.79
step 1100, training accuracy: 0.73
step 1200, training accuracy: 0.82
step 1300, training accuracy: 0.71
step 1400, training accuracy: 0.76
step 1500, training accuracy: 0.75
step 1600, training accuracy: 0.8
step 1700, training accuracy: 0.68
step 1800, training accuracy: 0.71
step 1900, training accuracy: 0.76
test accuracy: 0.842105
```
Pretty bad but better than raw MRIs in smaller, less separated groups.

Deep cnn with the right setup could probably do better, especially with the more recent 3d support. Currently need more of either: computing power, performance design, or patience...

Insired by [Sarraf et al](http://biorxiv.org/content/biorxiv/early/2016/08/30/070441.full.pdf)

## FSL Flow
Just follow [the fsl-vbm guide](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLVBM/UserGuide) (do evertything in python2 environment):
  - make a `sample_data/<data-set>/fsl` dir
  - move all images into that dir
  - balance groups so that there are equal n each and note one filename each in `fsl/template_list`
    - prepend file names with class str (ex `CB-*.nii.gz` and `HC-*.nii.gz`)
  - create design.mat and design.con
    - start with design.txt (one col per class, one row per sub, in order of files (alphabetical by group then id))
    - convert to design.mat with `Text2Vest design.txt design.mat`
    - repeat with contrasts.txt -> design.con (one col per class, like the 1-hot, match design.mat cols)
  - extract brains with `fslvbm_1_bet -b`
    - inspect for quality and continue when all good
  - generate (linear affine) icbm-152 template from selected images with `fslvbm_2_template -a`
    - check in fslview as per directions from script output
  - final processing, run `fslvbm_3_proc`

