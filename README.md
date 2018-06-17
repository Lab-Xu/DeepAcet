# DeepAcet
A deep learning framework to predict the lysine acetylation sites in protein
## File description
* In the "DeepLearning_method" folder, there are six python files. Each file is a implement of a prediction using one encoding method with deeplearning framework.
* In the "Encoding_method" folder, there are six matlab files. Each file is a implement of  one encoding method.
* In the "N_fold_cross_validation" folder, there are four python files. Each file is a n_fold cross validation of DeepAcet.

## Introduction of six encoding method
### One-hot encoding
The amino acids within a small range around the acetylation site are primary sequence features and have proven to be useful information for lysine acetylation sites prediction in previous studies.These features can be used to represent protein sequences.
### BLOSUM62 matrix
BLOSUM matrices have belonged to the most common substitution matrix series for protein homology search and sequence alignments since their publication in 1992. Essential characters of protein evolution can be learned from analysis of aligned protein sequences.
### Composition of K-space amino acid pairs(CKSAAP)
The CKSAAP encoding scheme reflects the information of amino acid pairs in small range within the peptides.
### Information gain(IG)
Shannon Entropy was defined as a unique function that represents the average amount of information for a set of objects according to their probabilities. It can be used to measure the conservation of amino acids in fragments.
### Physicochemical and biochemical properties
AAindex is a database of numerical indices representing various physicochemical and biochemical properties of amino acids. There are 566 entries in Amino Acid Index Database.
### Position-specific scoring matrix(PSSM)
To get information about the sequential evolution, we can exploit the data of the position-specific scoring matrix.

## Algorithm flow
we combined a series of feature extraction methods with deep learning framework to predict lysine acetylation sites and got better results. Two ways were adopted. One way was training the model by different coding schemes respectively. Another was combining six types of encoding schemes with F-score to train the model. The flow as shown below:
<img src="https://github.com/Sunmile/DeepAcet/blob/master/Picture/Fig1.png"> 



