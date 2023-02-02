#tests for LM MODEL

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
linearmodel = lm(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms, data = df.train)
summary(linearmodel)

predict_lm = predict(linearmodel, df.test)
predict_lm = data.frame(predict_lm)
score_regression(df.test, predict_lm) #performed 54%

####alternative model
linearmodel.new = lm(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms + wins, data = df.train.new)
predict_lm_new = predict(linearmodel.new, df.test.new)
predict_lm_new = data.frame(predict_lm_new)
score_regression(df.test.new, predict_lm_new) #performed actually 2% better, 56%


#REGRESSION TREE
predict_rt = predict(tree(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms
                               , df.train), newdata = df.test)
score_regression(df.test, predict_rt) #34%

#SVM REGRESSIONS
library(e1071)
test_scaled_train = scale(df.train[, c(6,7,8,10,11,16,20,21,22,23)], center = TRUE, scale = TRUE)
df.lsvm.regres = svm(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms, data = test_scaled_train, kernel = 'linear')
prediction_svm_regres <- predict(df.lsvm, newdata = df.test)
score_regression(df.test, prediction_svm_regres)


### CLASSIFICATION TECHNIQUES
library(caret)
library(e1071)
#We want to predict the probability of class=1 (winner) or class=0 (not winner)
#Then we are going to sort the probabilities and pick the greater probability of class=1, hence the driver with 
#the greatest probability of being a winner. 

# BAYES CLASSIFIER
df.nb <- naiveBayes(winner ~ . -positionOrder -resultId -points, data = df.train)
prediction_nb <- predict(df.nb, newdata = df.test, type = 'raw')
score_classification(df.test, prediction_nb) #41%
 
#DECISION TREE.
#using df.test 
df.dt = rpart(winner ~ . -positionOrder -resultId -points, data = df.train, method="class")
prediction_dt = predict(df.dt, newdata = df.test, method="prob")
score_classification(df.test, prediction_dt) #50%

#SVM CLASSIFICATION (using classic df.test)
df.lsvm = svm(winner ~ . , data = df.train[, -c(8,9)], kernel = 'radial', fitted = FALSE, probability = TRUE)   #sigmoid kernel working best
prediction_svm <- predict(df.lsvm, newdata = df.test[,-c(8,9)], fitted = FALSE, probability = TRUE)
SVM_class = data.frame(attributes(prediction_svm)$probabilities)
score_classification(df.test, SVM_class) #52%. (when removing factors was just 32%)
#32% with linear, 50.7% with polynomial, 46% with sigmoid, 52% with radial basis. 


### TRYING CLASSIFICATION BY REMOVING CATEGORICAL FEATURE ISSUE using NAs substitution
#we are using df.train.clean and df.test.clean (copies of df.train and df.clean but without categorical features converted in factors)
#also, in df.test.clean we substituted categories not present in df.train.clean with NAS
df.test.clean$status[which(!(df.test.clean$status %in% unique(df.train.clean$status)))] <- NA
df.test.clean$driv_nationality[which(!(df.test.clean$driv_nationality %in% unique(df.train.clean$driv_nationality)))] <- NA
df.test.clean$fullname[which(!(df.test.clean$fullname %in% unique(df.train.clean$fullname)))] <- NA
df.test.clean$const_name[which(!(df.test.clean$const_name %in% unique(df.train.clean$const_name)))] <- NA
df.test.clean$name[which(!(df.test.clean$name %in% unique(df.train.clean$name)))] <- NA

#DECISION TREE (+8%)
df.dt.clean = rpart(winner ~ . -positionOrder -resultId -points, data = df.train.clean, method="class")
prediction_dt_clean = predict(df.dt.clean, newdata = df.test.clean, method="prob")
score_classification(df.test.clean, prediction_dt_clean) #58 (without this technique was 50%)

#BAYES CLASSIFIER (+10%)
df.nb <- naiveBayes(winner ~ . -positionOrder -resultId -points, data = df.train.clean)
prediction_nb <- predict(df.nb, newdata = df.test.clean, type = 'raw')
score_classification(df.test.clean, prediction_nb) #51%

#RANDOM FOREST
library(randomForest)
df.rf <- randomForest(winner ~ . -points -positionOrder -resultId, data = df.train, ntree = 200, na.action = na.exclude)
prediction.rf <- predict(df.rf, df.test, type = 'prob')
score_classification(df.test, prediction.rf)


