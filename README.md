# Cas13_Lasso

A Lasso model for Cas13 base preference. 

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
