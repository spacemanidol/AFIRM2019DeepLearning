# AFRIRM Deep learning Track

```python
conda create -n afirm python=3.6
source activate afirm
pip install -U annoy
pip install -U torch torchvision
pip install -U sklearn
pip install -U pandas
pip install -U numpy
pip install -U spacy
pip install -U tqdm
python -m spacy download en
python -m spacy download en_core_web_lg
python -m spacy download en_vectors_web_lg
python preprocess.py
python train.py
python plot.py
```