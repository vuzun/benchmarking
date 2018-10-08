# Benchmarking

The purpose of this jupyter notebook and accompanying scripts was to compare various machine learning methods on the same dataset on time and accuracy as the number of features (variables) in the dataset grows.

The dataset was endometrial (UCEC) cancer's GISTIC scores obtained from [TCGA](http://firebrowse.org/).
The full processed data contains 19006 features for 507 endometrial cancer patient samples. The classes used for the classifiying methods were endometrioid/serous (endometrial cancer subtypes).

Methods used were random forests, Gaussian processes, support vector machines, PCA and k-means.

Python and R scripts were used to run models of various feature sizes and record time and accuracy.

The final comparisons are visualised in the notebook (.ipynb file).