#tests for LM MODEL

library(car)

#converting TRAIN and TEST categorical features in factors
df.train$driv_nationality = as.factor(df.train$driv_nationality)
df.train$fullname = as.factor(df.train$fullname)
df.train$const_name = as.factor(df.train$const_name)
df.train$name = as.factor(df.train$name)
str(df.train)

df.test$driv_nationality = as.factor(df.test$driv_nationality)
df.test$fullname = as.factor(df.test$fullname)
df.test$const_name = as.factor(df.test$const_name)
df.test$name = as.factor(df.test$name)
str(df.test)

#fixing FACTOR LEVELS in the testing dataframe
#In this way we set factor levels of ALL categorical columns of df.test to be exactly the same as df.train
levels(df.test$status) <- levels(df.train$status)
levels(df.test$driv_nationality) <- levels(df.train$driv_nationality)
levels(df.test$fullname) <- levels(df.train$fullname)
levels(df.test$const_name) <- levels(df.train$const_name)
levels(df.test$name) <- levels(df.train$name)


### REGRESSION MODELS

#linear regression
linearmodel = lm(positionOrder ~ driverId + grid + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms +wins, data = df.train)
predict_lm = predict(linearmodel, df.test)
score_regression(df.test, predict_lm) #58%
summary(linearmodel)
avPlots(linearmodel)


#REGRESSION TREE
predict_rt = predict(tree(positionOrder ~ + grid + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms + wins
                               , df.train), newdata = df.test)
score_regression(df.test, predict_rt) #47%




### KNN REGRESSION
#library(FNN)
#library(KernelKnn)
#knn_regressor = knnreg(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms + wins, data = df.train, k = 10) 
#prediction_knn <- predict(knn_regressor, newdata = df.test, )
#score_regression(df.test, prediction_knn) #37%


### CLASSIFICATION TECHNIQUES
library(caret)
library(e1071)
#We want to predict the probability of class=1 (winner) or class=0 (not winner)
#Then we are going to sort the probabilities and pick the greater probability of class=1, hence the driver with 
#the greatest probability of being a winner. 

#We are removing categorical features with a lot of levels since they are not providing useful informations to the model. 

# BAYES CLASSIFIER
df.nb <- naiveBayes(winner ~ . -number -positionOrder -resultId -points -status -fullname -const_name, data = df.train)
prediction_nb <- predict(df.nb, newdata = df.test, type = 'raw')
score_classification(df.test, prediction_nb) #50%
 
#DECISION TREE.
df.dt = rpart(winner ~ . -number -positionOrder -resultId -points -status -fullname -const_name, data = df.train, method="class")
prediction_dt = predict(df.dt, newdata = df.test, method="prob")
score_classification(df.test, prediction_dt) #58%
rpart.plot(df.dt,faclen = 2)


#SVM CLASSIFICATION.
df.lsvm = svm(winner ~ . , data = df.train[, -c(5,8,9,12,15)], kernel = 'radial', fitted = FALSE, probability = TRUE)   #sigmoid kernel working best
prediction_svm <- predict(df.lsvm, newdata = df.test[,-c(5,8,9,12,15)], fitted = FALSE, probability = TRUE)
SVM_class = data.frame(attributes(prediction_svm)$probabilities)
score_classification(df.test, SVM_class) #56%
#50% with linear, 48% with polynomial, 56% with sigmoid, 56% with radial basis. 

#RANDOM FOREST
df.rf <- randomForest(winner ~ . -number -points -positionOrder -resultId -const_name -name -fullname -status, data = df.train, ntree = 200)
prediction.rf <- predict(df.rf, df.test, type = 'prob')
prediction.rf[is.na(prediction.rf)] <- 0
score_classification(df.test, prediction.rf) #60%




#####################
### TRYING CLASSIFICATION BY REMOVING CATEGORICAL FEATURE ISSUE using NAs substitution
#we are using df.train.clean and df.test.clean (copies of df.train and df.clean but without categorical features converted in factors)
#also, in df.test.clean we substituted categories not present in df.train.clean with NAS
df.test.clean$status[which(!(df.test.clean$status %in% unique(df.train.clean$status)))] <- NA
df.test.clean$driv_nationality[which(!(df.test.clean$driv_nationality %in% unique(df.train.clean$driv_nationality)))] <- NA
df.test.clean$fullname[which(!(df.test.clean$fullname %in% unique(df.train.clean$fullname)))] <- NA
df.test.clean$const_name[which(!(df.test.clean$const_name %in% unique(df.train.clean$const_name)))] <- NA
df.test.clean$name[which(!(df.test.clean$name %in% unique(df.train.clean$name)))] <- NA

#DECISION TREE
df.dt.clean = rpart(winner ~ . -positionOrder -resultId -points -status, data = df.train.clean, method="class")
prediction_dt_clean = predict(df.dt.clean, newdata = df.test.clean, method="prob")
score_classification(df.test.clean, prediction_dt_clean) #54% (without this technique was 50%)
plot(df.dt.clean)
text(df.dt.clean)


#BAYES CLASSIFIER (+8%)
df.nb.clean <- naiveBayes(winner ~ . -positionOrder -resultId -points, data = df.train.clean)
prediction_nb_clean <- predict(df.nb, newdata = df.test.clean, type = 'raw')
score_classification(df.test.clean, prediction_nb_clean) #52%







