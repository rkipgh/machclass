# projectreport

*August 2015*

*This document reports on my project analysis for the Coursera--Johns Hopkins "Machine Learning" class.*


## Introduction

Project data are from here:  <http://groupware.les.inf.puc-rio.br/har>

Participants in the study were asked to perform weight-lifting exercises either correctly or incorrectly for a total of 5 movement patterns/activities (reported in the "classe" variable in the data set).  The goal of this project is to predict the participants' activities based on data from motion sensors that they were wearing.


## Data exploration and preprocessing



Besides the outcome variable (the 'classe' variable), there are 159 other variables in the data set.  The first few variables (e.g., timestamps, row numbers) were not relevant to predicting the outcome variable based on motion sensor data, so they were removed (the arguable exception is the participants' names, which might allow detection of motion patterns that vary among the participants).

The remaining variables divided into two categories:  those with no missing data and those that were composed almost entirely of missing data, as shown in the histogram:

![](projectreport_files/figure-html/unnamed-chunk-2-1.png) 

Thus, all variables that were composed of more than 95% missing data were discarded.



To further reduce the number of variables, the functions 'nearZeroVar' and 'findLinearCombos' from the 'caret' package were used to identify variables with little variance or that composed linear combinations, but neither of these methods identified variables for further elimination.  That left 52 potential predictors.  



Four groups of variables were found that showed high correlations (>0.85) within each group.  All of these variables except for one in each group could be eliminated with probably little effect on prediction error.  This process would eliminate 8 variables.  However, the correlations in one of the groups were driven by a single outlier.  Elimination of this outlier would mean that only 6 variables could be eliminated.  Subsequent analyses were performed with and without these variables and with and without the outlier, but the effects on the final analysis were negligible.  For brevity, these analysis variations will not be discussed further.



Principal components analysis was used to further simplify the data.  To retain at least 80% of the variance, 12 principal components were generated and used in subsequent analysis.


## Machine learning model training



The original plan was to run several machine learning algorithms on 60% of the cases as a training set with cross-validation with K-folds.  However, running the algorithms (with the 'train' function from the 'caret' package) on the large data set kept producing memory allocation errors, even when only very small portions of the data set were used for training (e.g., 50 - 200 cases).  Therefore, I manually set up a random sample of the data and settled on 1500 cases, which was small enough to avoid memory errors and for the algorithms to run in a reasonable amount of time.  


```r
ctrl = trainControl(method = "cv", p = 0.60, number = 10)
```





I ran random forests, generalized boosted regression, naive Bayes, linear discriminant analysis, and classification tree ('rpart') models with cross-validation (settings shown above). This set of models was run for both the principal components (PCA) variables and for the non-PCA-transformed variables as separate data sets.  The in-sample accuracy measurements are below:

0.9313417 - non-PCA, Random forests

0.9032203 - non-PCA, Generalized boosted regression

0.6974288 - non-PCA, Naive Bayes

0.7026829 - non-PCA, Linear discriminant analysis

0.5089165 - non-PCA, Classification tree


0.7859805 - PCA, Random forests

0.6727186 - PCA, Generalized boosted regression

0.5261418 - PCA, Naive Bayes

0.4663647 - PCA, Linear discriminant analysis

0.3725405 - PCA, Classification tree

For both the PCA and non-PCA data sets, random forests and generalized boosted regression produced the highest accuracy in-sample.  Notably, the accuracies for all the models in the PCA data set were substantially lower than in the non-PCA data set.  Simplifying the data with PCA clearly diminished the prediction accuracies of the models.  But perhaps much of the difference is due to overfitting to the non-PCA data; perhaps the PCA data eliminated a lot of noise so that the models will generalize well out-of-sample.


## Machine learning model testing

The out-of-sample accuracies for the two best in-sample models (random forests and generalized boosted regression) were calculated on the 18122 cases that were not part of the 1500 cases used for training.  The out-of-sample accuracies are below:

0.9391899 - non-PCA, Random forests

0.9186072 - non-PCA, Generalized boosted regression

0.7837987 - PCA, Random forests

0.6750359 - PCA, Generalized boosted regression

The models still performed substantially better on the non-PCA data than on the PCA data.



Finally, a combined, or ensembled, model was created by taking the majority vote of the random forests, generalized boosted regression, and naive Bayes models.  This ensembled model was tested on the non-PCA data with 18122 cases.  Its accuracy is below:

0.9285399 - non-PCA, Ensembled model

This model performed about the same as the corresponding random forests and generalized boosted regression models on the same data.  I had intended to validate the ensembled model on the assignment-provided testing data (from the file "pml-testing.csv"), but I did not realize that the data are composed of only 20 cases.  Thus, it remains to be seen whether the ensembled model might perform better than the random forests or generalized boosted regression models in a new set of data.

## Conclusions

1. Random forests and generalized boosted regression models produced the highest accuracies on this data set
2. Simplifying the data with PCA substantially reduced the prediction accuracies compared with the non-PCA data.  This report did not examine whether using PCA provided a benefit by substantially reducing computational time for the models.
3. Removing variables that were highly correlated with other variables in the data set did not substantially change the prediction accuracies obtained by the models (results not shown).  Such variable elimination might provide a benefit by reducing computational time, but this was not examined.
4. Removing one outlier that accounted for several high correlations among variables did not substantially change the prediction accuracies obtained by the models (results not shown).  However, outliers were not subject to an exhaustive search effort, so this might be a useful avenue for improving the models, especially the ones that make stronger assumptions about the data distributions.
