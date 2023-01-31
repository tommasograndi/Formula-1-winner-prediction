#tests for LM MODELS

#linear regression
model = lm(positionOrder ~ grid + number + points + laps + fastestLapSpeed + round + const_points + driver_age + fastestLap_ms, data = df)
summary(model)

prova = predict(model, df)
prova = data.frame(prova)
df_test_lm = data.frame(c(df, prova)) #creating df with predictions attached to original df
df_test_lm = df_test_lm[, c(1,4,7,8,24,15)]

# BAYES CLASSIFIER (need to test if it works)
library(caret)
library(e1071)
for (i in 1:nrow(df.results.merged)) {
  if (df.results.merged[i, "positionText"] == '1'){
    df.results.merged[i, 'winner'] <- 1
  }else{
    df.results.merged[i, 'winner'] <- 0
  }
}
df.idx <- sample(nrow(df), 19178)
df.train <- df.results.merged[df.idx,]
df.test <- df.results.merged[-df.idx,]

df.nb <- naiveBayes(winner ~ ., data = df.results.merged)
df.nb
prediction <- predict(df.nb, df.test)
table(df.test$winner, prediction)

#test with less features 
df.nb.2 <- naiveBayes(winner ~ number + grid + points + laps + fastestLapSpeed + status + driv_nationality + fullname + round + const_name + name + const_points + const_wins + driver_age + fastestLap_ms, data = df.results.merged)
prediction2 <- predict(df.nb.2, df.test)
table(df.test$winner, prediction2)
### comments: performed worse. we could use it as a benchmark for the decision tree (expected to perform better)


#CLASSIFICATION (not yet runned, just written to be tested later)
# train model, decision tree for predicting class winner:{0,1}
#df.decision.train = rpart(formula, data=df_train, method="class")

# predict probabilities of class winner
#df.decision.test = predict(model.rpart, newdata=df_test, method="prob")

# we want to predict the probability of class=1 (winner) or class=0 (not winner)
#Then we are going to sort the probabilities and pick the greater probability of class=1, hence the driver with 
#the greatest probability of being a winner. 

