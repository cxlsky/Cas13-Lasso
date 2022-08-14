# Cas13_Lasso

A Lasso model for calculating Cas13 base preference. 

This repository is desinged for our paper *Intrinsic RNA targeting constrains the utility of CRISPR-Cas13 systems![image](https://user-images.githubusercontent.com/20998111/184535758-64dae8f5-df80-4ffa-8ec1-54246dafd783.png)
* [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.05.14.491940v1)

### Install dependencies ###

```
  pip install logomaker
  
  conda install -c conda-forge seaborn pandas numpy matplotlib
  
  conda install -c conda-forge scipy
  
  pip install biopython
  
  pip3 install -U scikit-learn

```

### Run Lasso Model ###

We provide a Jupyer Notebook script for user to run the model. In the script, we can:

	1. Generate features for the extended target sequence
	
	2. Train Lasso model
	
	3. Draw ROC curve (receiver operating characteristic curve) and calculate AUC score
	
	4. Plot base preference logo
