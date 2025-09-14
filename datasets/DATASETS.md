# Steps to install Datasets

## DUTS

Follow the below steps to download and unzip the dataset. 
```
wget http://saliencydetection.net/duts/download/DUTS-TE.zip

unzip DUTS-TE.zip
```

Python alternative to extract zip: 
```
from zipfile import ZipFile

with ZipFile("DUTS-TE.zip", 'r') as f:
  f.extractall()
```

Place all the dirs/files extracted in this directory. Verify that these are present:
- DUTS-TE/DUTS-TE-Mask
- DUTS-TE/DUTS-TE-Image

## CUB

Follow below steps to download and unzip the dataset.
```
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
wget https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1

tar -xzf CUB_200_2011.tgz
tar -xzf segmentations.tgz?download=1
```

Python alternative to extract tar:
```
import tarfile

file = tarfile.open("CUB_200_2011.tgz")
file.extractall()
file.close()

file = tarfile.open("segmentations.tgz?download=1")
file.extractall()
file.close()
```

Place all the dirs/files extracted in this directory. Verify that these are present:
- segmentations
- CUB_200_2011/images
- CUB_200_2011/train_test_split.txt
- CUB_200_2011/images.txt

## ECSSD

No manual loading required, directly loading from Deeplake library.


## CUSTOM

Within the datasets directory:
```
mkdir CUSTOM
cd CUSTOM
```
Add the gtMasks and images to the CUSTOM directory.
Then, at root directory:
```
python main.py --dataset CUSTOM
```
