# Machine Learning Predictive Model from Monitored Exercise

author: Marcos Medeiros

date: 08/09/2022 

## Synopsis

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways to predict the manner in which participants will perform a barbell lift. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har 

## Possible Outcomes

The outcome variable is `classe`, a factor variable  with 5 levels of precision in a set of 10 repetitions of Unilateral Dumbbell Curl:

`class A` exactly according to the specification;

`class B` throwing the elbows to the front;

`class C` lifting the dumbbell only halfway;

`class D` lowering the dumbbell only halfway;

`class E` throwing the hips to the front.

## Data Loading and preparing analisys 
```{r}
library(lattice)
library(ggplot2)
library(caret)
library(corrplot)
library(gbm)
library(rpart)
library(rpart.plot)
library(rattle)

trainURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL <- "http://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

training <- read.csv(url(trainURL))
testing <- read.csv(url(testURL))


## Creating a partition

label <- createDataPartition(training$classe, p = 0.7, list = FALSE)
train <- training[label, ]
test <- training[-label, ]


## Filtering data 
### Excluding variables with nearly zero variance

NZV <- nearZeroVar(train)
train <- train[ ,-NZV]
test <- test[ ,-NZV]

### Excluding variables with more than 90% NAs 

label <- apply(train, 2, function(x) mean(is.na(x))) > 0.90
train <- train[, -which(label, label == FALSE)]
test <- test[, -which(label, label == FALSE)]

### Excluding identification variables

train <- train[ , -(1:5)]
test <- test[ , -(1:5)]

dim(train)
dim(test)
```

```{r}
## [1] 13737    54
## [1] 5885   54
```

We reduced the dataset from 160 to 54 variables.

## Exploratory Analysis

```{r}

## Making a correlation plot to look the dependence intensity

depend <- cor(train[,-54])
corrplot(depend, method = "color", type = "lower", tl.cex = 0.5, tl.col = rgb(0,0,0))
```
![Figure1](https://github.com/msrcos3s/practical_machine_learning/blob/main/Figure_1.jpg)

## Predictive Model Selection

To choose what method provides the best accuracy in the predictive model, we will perform Random Forest, Generalized Boosted Model and Decision Tree. A confusion matrix at the end of each model will help to compare them. 


### Random Forest

```{r}
set.seed(14518)
control <- trainControl(method = "cv", number = 4, verboseIter=FALSE)
modelRF <- train(classe ~ ., data = train, method = "rf", trControl = control)
modelRF$finalModel

predictRF <- predict(modelRF, test)
confMatRF <- confusionMatrix(predictRF, as.factor(test$classe))
confMatRF
```

```{r}
## Output

Call:
 randomForest(x = x, y = y, mtry = min(param$mtry, ncol(x))) 
               Type of random forest: classification
                     Number of trees: 500
No. of variables tried at each split: 27

        OOB estimate of  error rate: 0.23%
Confusion matrix:
     A    B    C    D    E  class.error
A 3905    0    0    0    1 0.0002560164
B    7 2648    3    0    0 0.0037622272
C    0    8 2388    0    0 0.0033388982
D    0    0   10 2241    1 0.0048845471
E    0    0    0    2 2523 0.0007920792
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1673    2    0    0    0
         B    1 1137    0    0    0
         C    0    0 1026    4    0
         D    0    0    0  960    2
         E    0    0    0    0 1080

Overall Statistics
                                          
               Accuracy : 0.9985          
                 95% CI : (0.9971, 0.9993)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.9981          
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9994   0.9982   1.0000   0.9959   0.9982
Specificity            0.9995   0.9998   0.9992   0.9996   1.0000
Pos Pred Value         0.9988   0.9991   0.9961   0.9979   1.0000
Neg Pred Value         0.9998   0.9996   1.0000   0.9992   0.9996
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2843   0.1932   0.1743   0.1631   0.1835
Detection Prevalence   0.2846   0.1934   0.1750   0.1635   0.1835
Balanced Accuracy      0.9995   0.9990   0.9996   0.9977   0.9991
```

### Generalized Boosted Model 

```{r}
set.seed(14518)
control <- trainControl(method = "repeatedcv", number = 5, repeats = 1, verboseIter = FALSE)
modelGBM <- train(classe ~ ., data = train, trControl = control, method = "gbm", verbose = FALSE)
modelGBM$finalModel

predictGBM <- predict(modelGBM, test)
confMatGBM <- confusionMatrix(predictGBM, as.factor(test$classe))
confMatGBM
```
```{r}
## Output

A gradient boosted model with multinomial loss function.
150 iterations were performed.
There were 53 predictors of which 53 had non-zero influence.
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1671   11    0    2    0
         B    3 1112    9    4    3
         C    0   15 1015   14    2
         D    0    1    1  943   12
         E    0    0    1    1 1065

Overall Statistics
                                          
               Accuracy : 0.9866          
                 95% CI : (0.9833, 0.9894)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.983           
                                          
 Mcnemar's Test P-Value : NA              

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.9982   0.9763   0.9893   0.9782   0.9843
Specificity            0.9969   0.9960   0.9936   0.9972   0.9996
Pos Pred Value         0.9923   0.9832   0.9704   0.9854   0.9981
Neg Pred Value         0.9993   0.9943   0.9977   0.9957   0.9965
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2839   0.1890   0.1725   0.1602   0.1810
Detection Prevalence   0.2862   0.1922   0.1777   0.1626   0.1813
Balanced Accuracy      0.9976   0.9861   0.9914   0.9877   0.9919
```

### Decision Tree

```{r}
set.seed(14518)
modelDT <- rpart(classe ~ ., data = train, method = "class")
fancyRpartPlot(modelDT)
```
![Figure1](https://github.com/msrcos3s/practical_machine_learning/blob/main/Figure_2.jpg)

```{r}
predictDT <- predict(modelDT, test, type = "class")
confMatDT <- confusionMatrix(predictDT, as.factor(test$classe))
confMatDT
```
```{r}
## Output
Confusion Matrix and Statistics

          Reference
Prediction    A    B    C    D    E
         A 1464  247   35  106   62
         B   53  636   72   45  106
         C   20   82  836  160   80
         D   85  129   58  613  125
         E   52   45   25   40  709

Overall Statistics
                                          
               Accuracy : 0.7235          
                 95% CI : (0.7119, 0.7349)
    No Information Rate : 0.2845          
    P-Value [Acc > NIR] : < 2.2e-16       
                                          
                  Kappa : 0.6488          
                                          
 Mcnemar's Test P-Value : < 2.2e-16       

Statistics by Class:

                     Class: A Class: B Class: C Class: D Class: E
Sensitivity            0.8746   0.5584   0.8148   0.6359   0.6553
Specificity            0.8931   0.9418   0.9296   0.9193   0.9663
Pos Pred Value         0.7649   0.6974   0.7097   0.6069   0.8140
Neg Pred Value         0.9471   0.8989   0.9596   0.9280   0.9256
Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
Detection Rate         0.2488   0.1081   0.1421   0.1042   0.1205
Detection Prevalence   0.3252   0.1550   0.2002   0.1716   0.1480
Balanced Accuracy      0.8838   0.7501   0.8722   0.7776   0.8108
```

## Conclusion
Random Forest Model offers the best accuracy, with 0.9968 95%CI (0.9950, 0.9981) 


## Predicting Results

```{r}
predictRF <- predict(modelRF, testing)
predictRF
```
```{r}
## Output

[1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E
```
