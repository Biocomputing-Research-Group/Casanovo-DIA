# Casanovo-DIA

## Transformer-based de novo peptide sequencing for data-independent acquisition mass spectrometry 


## Installation

	conda env create -n casanovo-dia -f requirements_casanovo_dia.yml

Or install Casanovo [2] with the following command, then replace the files with the casanovo-dia files :

	pip install Casanovo 
Casanovo GitHub like: https://github.com/Noble-Lab/casanovo


## Dataset
We used the same data sets released by DeepNovo-DIA [1], download OC, Plasma, and UTI from google drive link. Each data set has two MS/MS spectra file with corresponding feature file, you can find more detailed here: https://github.com/nh2tran/DeepNovo-DIA/tree/master 

## Casnovo-DIA usage

### Step 0: Data Preparation 
- Split each dataset into train set, validation set, and test set
- Annotate the spectra file with the corresponding label extracted from feature file
- Create a new feature file 



### Step 1: Run de novo sequencing with a pre-trained model:

    Casanovo --mode=denovo —peak_path=path/to/predict/<test-spectrum file> —peak_feature=path/to/train/<feature-file> 

We have provided a pre-trained folder


### Step 2: Evaluate de novo sequencing results:

    Casanovo --mode=eval —model=path/to/pre-trained-model/<ckpt file> —peak_path=path/to/evalaute/<test-spectrum file> —peak_feature=path/to/train/<feature-file> 


### Step 3: Train a new model:

    Casanovo --mode=train —peak_path=path/to/train/<train-spectrum file> —peak_feature=path/to/train/<feature-file> —peak_path_val=path/to/validation/<validation-spectrum file> —peak_feature_val=path/to/validation/<feature-file>

## References:
 1. N.H.Tran,R.Qiao,L.Xin,X.Chen,C.Liu,X.Zhang,B.Shan,A.Gh- odsi, and M. Li, “Deep learning enables de novo peptide sequencing from data-independent-acquisition mass spectrometry,” Nature methods, vol. 16, no. 1, pp. 63–66, 2019. 

 2. Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S. & Noble, W. S. De novo mass spectrometry peptide sequencing with a transformer model. in Proceedings of the 39th International Conference on Machine Learning - ICML '22 vol. 162 25514–25522 (PMLR, 2022). https://proceedings.mlr.press/v162/yilmaz22a.html

