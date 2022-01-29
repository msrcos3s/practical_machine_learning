# Machine Learning Predictive Model from Monitored Exercise

author: Marcos Medeiros

date: 1/28/2022

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

```
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
