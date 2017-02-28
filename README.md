`virtualenv -p /usr/local/bin/python3.6 venv`
`pip install -r requirements.txt`

`python -m unittest test_tf_data.py`

Very inconsistent, but mostly just getting the data pipeline going
```python
(venv) pfonseca@mn-pfonseca:neuropy$ python naive_cannabis_softmax.py
Loading images: |████████████████████████████████████████| 100.0%
slicing data...

step 0, training accuracy: 0.48
step 1, training accuracy: 0.5
step 2, training accuracy: 0.52
test accuracy: 0.736842
```
