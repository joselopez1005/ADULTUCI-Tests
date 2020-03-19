
library(ggplot2)
library(sampling)
library(scatterplot3d)
library(Matrix)
library(arules)
library(mlbench)
library(rpart)
library(rpart.plot)
library(ROCR)
library(foreach)

AdultUCI <- read.csv("C:/Users/User/Downloads/AdultUCI.dat")

#Remove any object with NAs and/or duplicated
clean_x <- unique(AdultUCI[complete.cases(AdultUCI),])

#Create a default tree for income
tree_Adult <- rpart(income~., data = clean_x)

#WIll plot the actual tree
rpart.plot(tree_Adult)

#WIll predict a class for each object in the tree
pred_Adult <- predict(tree_Adult, clean_x, type="class")

#Building a contengency table to find necessary information about predictions
confusion_table <- table(clean_x$income, pred_Adult)
confusion_table

#WIll find true positive rate for the prediction
TPR_pred <- confusion_table[1,1]/sum(confusion_table[1,1:2])
TPR_pred

#Will find true negative rate for the prediction
TNR_pred <- confusion_table[2,2]/sum(confusion_table[2,1:2])
TNR_pred

#WIll find False positive rate for the prediction
FPR_pred <- 1 - TNR_pred
FPR_pred

#Will find F negative rate for the prediction
FNR_pred <- 1 - TPR_pred
FNR_pred

#WIll find the accuracy of the predictions
correct <- sum(diag(confusion_table))
error <- sum(confusion_table) - correct
accuracy <- correct/(correct+error)
accuracy

#Will create a contengency table over the first 10 predicted and their actual value
confusion_ten <- table(clean_x$income[1:10], pred_Adult[1:10])
confusion_ten

#To find error rate
correct_ten <- sum(diag(confusion_ten))
error_ten <- sum(confusion_ten) - correct_ten
error_rateTen <- error_ten/(correct_ten+error_ten)
error_rateTen

#To find training error on whole tree.
error_rate <- error/(correct+error)

#Optimisitc generalization error will be equal to this
error_rate


#To find pessimistic generalization error
amount_leaves <- sum(tree_Adult$frame$var == "<leaf>")
generalization_error <- (error + (amount_leaves * .5))/ nrow(clean_x)
generalization_error

#Creating the full grown decision tree
tree_fullAdult <- rpart(income~., data = clean_x, control = rpart.control(minsplit = 2, cp = 0))

#Predicting the income using the full grown decision tree
predFull_Adult <- predict(tree_fullAdult, data = clean_x, type="class")

#Need to create a contegency table to find necessary information about predictions
confusion_table_full <- table(clean_x$income, predFull_Adult)
confusion_table_full

#WIll find true positive rate for the prediction
TPR_pred_full <- confusion_table_full[1,1]/sum(confusion_table_full[1,1:2])
TPR_pred_full

#Will find true negative rate for the prediction
TNR_pred_full <- confusion_table_full[2,2]/sum(confusion_table_full[2,1:2])
TNR_pred_full

#WIll find False positive rate for the prediction
FPR_pred_full <- 1 - TNR_pred_full
FPR_pred_full

#Will find F negative rate for the prediction
FNR_pred_full <- 1 - TPR_pred_full
FNR_pred_full

#Finding the accuracy of the full decision tree
correct_full <- sum(diag(confusion_table_full))
error_full <- sum(confusion_table_full) - correct_full
accuracy_full <- correct_full / (correct_full+error_full)
accuracy_full

#Finding generalization error
#Training error will be generalization error in optmistic approach
error_rate_full <- error_full/(correct_full+error_full)
error_rate_full


#Using pessimistic approach
amount_leaves_full <- sum(tree_fullAdult$frame$var == "<leaf>")
generalization_error <- (error_full + (amount_leaves_full * .5))/ nrow(clean_x)
generalization_error

#Need to only use a fraction of the objects. Will use 1/3rd of the data
n_train <- as.integer(nrow(clean_x) * .33)
n_train

#Select n_train objects from the whole data frame
train_id <- sample(1:nrow(clean_x), n_train)
train <- clean_x[train_id,]

#Creating the tree with 1/3rd of data objects
third_tree <- rpart(income~., data = train)
rpart.plot(third_tree)


#Create prediction for whole objects using 1/3 object tree
pred_Adult_third <- predict(third_tree, clean_x, type="class")

#Create contengency table to find detail information about predictions
confusion_table_third <- table(clean_x$income, pred_Adult)
confusion_table_third

#Finding training error. Should be the highest out of all previous trees used
correct_third <- sum(diag(confusion_table_third))
error_third <- sum(confusion_table_third) - correct_third
accuracy_third <- correct_third/(correct_third+error_third)
error_rate_third <- 1 - accuracy_third
error_rate_third

object_predict <- data.frame(c1 = c(24, 38),
                             c2 = c("Private", "Self-emp-not-inc"),
                             c3 = c(43323,120985),
                             c4 = c("HS-grad","HS-grad"),
                             c5 = c(9,9),
                             c6 = c("Never-married", "Married-civ-spouse"),
                             c7 = c("Other-service", "Craft-repair"),
                             c8 = c("Not-in-family", "Husband"),
                             c9 = c("White", "White"),
                             c10 = c("Female","Male"),
                             c11 = c(0,4386),
                             c12 = c(1762,0),
                             c13 = c(40,35),
                             c14 = c("United-States","United-States"),
                             c15 = c(NA,NA))
colnames(object_predict) <- make.names(colnames(clean_x))
#Predict using default tree
predict(tree_Adult, object_predict, type="class")

#Predict using full tree
predict(tree_fullAdult, object_predict, type="class")

#Splitting data into 2/3rd training and 1/3 test set
#Obtaining training set
train_amount <- as.integer(nrow(clean_x)*.66)
training_id <- sample(1:nrow(clean_x), train_amount)
training_set <- clean_x[training_id,]

#Obtaining testing set
test <- clean_x[-training_id, colnames(clean_x) != "income"]
test_type <- clean_x[-training_id, "income"]

#Creating the tree using the training set
training_tree <- rpart(income~., data = training_set)

#Training error for training set
accuracy <- function(truth, prediction) {
  tbl <- table(truth, prediction)
  sum(diag(tbl))/sum(tbl)
}
1-accuracy(training_set$income, predict(training_tree, training_set, type="class"))

#Training error on testing set
1- accuracy(test_type, predict(training_tree, test, type="class"))

#Shulffling objects
index <- 1:nrow(clean_x)
index <- sample(index)

#Creating 10 folds
fold <- rep(1:10, each = nrow(clean_x) / 10)[1 : nrow(clean_x)]
folds <- split(index,fold)

#Each testing and training set will have 3013 objects
summary(folds)

#Use training set of every row minus i and then apply it to row i
accs <- vector(mode="numeric")
error_rate_fold <- vector(mode="numeric")
for(i in 1:length(folds)) {
  tree_fold <- rpart(income ~., data=clean_x[-folds[[i]],], control=rpart.control(minsplit=2))
  error_rate_fold[i] <- accuracy(clean_x[-folds[[i]],]$income, predict(tree_fold, clean_x[-folds[[i]],], type="class"))
  accs[i] <- accuracy(clean_x[folds[[i]],]$income, predict(tree_fold, clean_x[folds[[i]],], type="class"))
}
#Training error
error_rate_fold

#Testing error
accs

#average testing error
mean(1-accs)
