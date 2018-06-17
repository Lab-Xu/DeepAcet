# DeepAcet
A deep learning framework to predict the lysine acetylation sites in protein
## Requirements
* Python>=3.6
* Matlab2016a
* Tensorflow =1.6.0

## File description
* In the "DeepLearning_method" folder, there are six python files. Each file is a implement of a prediction using one encoding method with deeplearning framework.
* In the "Encoding_method" folder, there are six matlab files. Each file is a implement of  one encoding method.
* In the "N_fold_cross_validation" folder, there are four python files. Each file is a n_fold cross validation of DeepAcet.

## Introduction of six encoding method
### One-hot encoding
The amino acids within a small range around the acetylation site are primary sequence features and have proven to be useful information for lysine acetylation sites prediction in previous studies.These features can be used to represent protein sequences [[1]](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0015411).
### BLOSUM62 matrix
BLOSUM matrices have belonged to the most common substitution matrix series for protein homology search and sequence alignments since their publication in 1992. Essential characters of protein evolution can be learned from analysis of aligned protein sequences[[2]](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-016-1060-3) [[3]](https://www.sciencedirect.com/science/article/pii/S0375960107016271).
### Composition of K-space amino acid pairs(CKSAAP)
The CKSAAP encoding scheme reflects the information of amino acid pairs in small range within the peptides.
### Information gain(IG)
Shannon Entropy was defined as a unique function that represents the average amount of information for a set of objects according to their probabilities. It can be used to measure the conservation of amino acids in fragments.
### Physicochemical and biochemical properties
AAindex is a database of numerical indices representing various physicochemical and biochemical properties of amino acids [[4]](https://academic.oup.com/nar/article/36/suppl_1/D202/2508449). There are 566 entries in Amino Acid Index Database.
### Position-specific scoring matrix(PSSM)
To get information about the sequential evolution, we can exploit the data of the position-specific scoring matrix [[5]](http://pubs.rsc.org/en/Content/ArticleLanding/2017/MB/C7MB00180K#!divAbstract).

## Algorithm flow
we combined a series of feature extraction methods with deep learning framework to predict lysine acetylation sites and got better results. Two ways were adopted. One way was training the model by different coding schemes respectively. Another was combining six types of encoding schemes with F-score to train the model. The flow as shown below:
<img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig1.png"> 
## DeepLearing Framework
We constructed a feedforward neural network of six layers (including input and output layers).
<img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig2.png"> 

## Results
* Comparisons of fragments information between lysine acetylation and non-acetylation sites. (A) The percentage of amino acids in the lysine acetylation and the non-acetylation fragments. (B) A pLogo of compositional bias around the lysine acetylation and non-acetylation sites
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig3A.png"> 
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig3B.png"> 
* Performance measures of different features. (A)the Accuracy, Specificity, Sensitivity, AUC values of different features. (B)ROC curves and their AUC values of different features.
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig4A.png"> 
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig4B.png"> 
* The distribution of the number of each type of features and their corresponding F-score sums in the optimized feature set. (A) The distribution of the number of each type of features. (B) The distribution of F-score sums of each type of features.
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig5A.png"> 
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig5B.png"> 
* Performance measures of the optimized selected predictors. (A) the Accuracy, Specificity, Sensitivity, AUC values in 4-,6-,8-,10-fold cross-validation. (B)ROC curves in and their AUC values in 4-,6-,8-,10-fold cross-validation.
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig6A.png"> 
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig6B.png"> 
* The ROC curves for the independent test set.
  * <img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig7.png"> 
