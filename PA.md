# Coursera Practical Machine Learning Course Project
Kolachalam Manish Sharma  

The objective of this analysis was to use data from the Groupware@LES Human Activity Recognition Weight Lifting Exercises Dataset(reference at the end of the report) to predict the kind of activity being performed.        

The dataset consisted of data from multiple sensors attached to six participants performing one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).       

Random forests classification was used to create a model which was then used on a subset of the data for crossvalidation. Using this method, classification of activity type in the crossvalidation set was predicted with accuracy greater than 99%. The final accuracy of the model as judged by the test set provided was 100%.     

###Data Processing

The data consisted of 19622 observations of 160 variables. These data were processed to remove annotative columns and impute missing values. While this processing was being performed on the test set, it was noticed that the test dataset contained only 52 variable columns whose sum was not zero i.e. those that could be used for imputation. Furthermore, none of these colums contained missing values in either the test set or the training set. Hence, these columns were used to create a dataset to train the model and the scheme for imputing missing values (using "knnImpute") was discarded. 


```r
training <- read.csv("pml-training.csv", stringsAsFactors = F)
testing <- read.csv("pml-testing.csv", stringsAsFactors = F)
classe <- training[,160]

final <- testing[,-c(1:7)]
final<- data.frame(lapply(final[,-153], as.numeric))
final <- final[,!(colSums(final, na.rm = T) == 0)]
sum(is.na(final))
```

```
## [1] 0
```

```r
subset <- training[,-c(1:7)]
subset<- data.frame(lapply(subset[,-153], as.numeric))
subset <- subset[,colnames(final)]
subset <- cbind(classe, subset)
sum(is.na(subset))
```

```
## [1] 0
```


###Model training

This reduced training dataset was then partitioned into training and crossvalidation datasets and the training dataset used to train the random forests model.


```r
library(caret)
intrain <- createDataPartition(subset$classe, p = 0.7, list = F)
train <- subset[intrain, ]
test <- subset[-intrain, ]
```

To train the model, the "randomForest" function was used with 500 trees and 7 variables sampled at each split (using "train" from the caret package was prohibitively slow).


```r
library(randomForest)
model <- randomForest(classe~., data = train)
model
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.61%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3901    4    0    0    1    0.001280
## B   16 2635    7    0    0    0.008653
## C    0   15 2378    3    0    0.007513
## D    0    0   27 2224    1    0.012433
## E    0    0    2    8 2515    0.003960
```

###Crossvalidation and results

Although the randomForest function returns a value for out of bag(OOB) error for the model (which was 0.54%), to ensure that its predictions were accurate, the model was used to predict activity classes using the crossvalidation set. The confusion matrix for this prediction is shown below the code. This is followed by the predictions for the test set.


```r
pred <- predict(model, newdata = test[,-1])
require("e1071")
confusionMatrix(pred, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    7    0    0    0
##          B    1 1130    7    0    0
##          C    0    2 1019    4    0
##          D    0    0    0  958    2
##          E    0    0    0    2 1080
## 
## Overall Statistics
##                                         
##                Accuracy : 0.996         
##                  95% CI : (0.994, 0.997)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.995         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.993    0.994    0.998
## Specificity             0.998    0.998    0.999    1.000    1.000
## Pos Pred Value          0.996    0.993    0.994    0.998    0.998
## Neg Pred Value          1.000    0.998    0.999    0.999    1.000
## Prevalence              0.284    0.194    0.174    0.164    0.184
## Detection Rate          0.284    0.192    0.173    0.163    0.184
## Detection Prevalence    0.285    0.193    0.174    0.163    0.184
## Balanced Accuracy       0.999    0.995    0.996    0.997    0.999
```

```r
finalanswer <- predict(model, newdata = final)
finalanswer
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

#####Reference

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
