# Environment Audio Classification


[https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YDEPUT]

# Configuration

First, create a conda env:

```sh
conda create -n Environment_Audio_Classification python=3.6 pip
conda activate Environment_Audio_Classification
```

Install the requirements:

```sh
pip install -r requirements.txt
```


# Training
python train.py -c config.json
