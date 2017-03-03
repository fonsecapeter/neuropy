`virtualenv -p /usr/local/bin/python3.6 venv`
`pip install -r requirements.txt`
`. venv/bin/activate`
`python -m unittest util.test_tf_data.py`

oases dataset from `nilearn.datasets.featch_oasis_vbm(data_dir='sample_data/oasis')`
from there the `oasis_cross-sectional.csv` has a group row added according to the dataset (specs)[http://www.oasis-brains.org/pdf/oasis_cross-sectional_facts.pdf]:
  - `HC` for `CDR` nonexistent or 0
  - `AD` for `CDR` 0.5 (mild), 1 (moderate), or 2 (severe)

data are preprocessed with fsl for vbm to dartel space

`python naive_oasis_softmax.py`
batch size: 16
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
pretty bad but better than 50/50
cnn with the right setup could do better


FSL preprocessing plan as used in (Sarraf et al)[http://biorxiv.org/content/biorxiv/early/2016/08/30/070441.full.pdf] (write script to do it)
  - extract brain using (fsl)[https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET/UserGuide] `bet`
  - generate study-specific gm template
    - segment brains to gm/wm/csf
    - register gm to icbm-152 space
    - create affine gm template from registered gm images
  - register gm images to new template to generate final study-specific template in standard space
  - bring images into study space

just follow (the fsl-vbm guide)[https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSLVBM/UserGuide] (do evertything in python2 environment)
  - make a `sample_data/<data-set>/fsl` dir
  - move all images into that dir
  - balance groups and note one filename each in `fsl/template_list`
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

