#tests for LM MODELS

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


# BAYES CLASSIFIER
library(caret)
library(e1071)

df.nb <- naiveBayes(winner ~ . -positionOrder -resultId -points, data = df.train)
prediction_nb <- predict(df.nb, newdata = df.test, type = 'raw')
score_classification(df.test, prediction_nb)

#CLASSIFICATION (dec tree not working for factors)
#train model, decision tree for predicting class winner:{0,1}
df.dt = rpart(winner ~ . -positionOrder -resultId -points -status -driv_nationality -fullname -const_name -name, data = df.train, method="class")

# predict probabilities of class winner
prediction_dt = predict(df.dt, newdata = df.test, method="prob")
score_classification(df.test, prediction_dt)

# we want to predict the probability of class=1 (winner) or class=0 (not winner)
#Then we are going to sort the probabilities and pick the greater probability of class=1, hence the driver with 
#the greatest probability of being a winner. 


# SVM
library(e1071)
df.lsvm = svm(winner ~ ., data = df.train, kernel = 'linear')
prediction_svm <- predict(df.lsvm, newdata = df.test)
score_classification(df.test, prediction_svm)
