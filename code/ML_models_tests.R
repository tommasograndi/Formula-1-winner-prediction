#tests for LM MODELS

#linear regression
model = lm(positionOrder ~ grid + number + points + laps + fastestLapSpeed + round + const_points + driver_age + fastestLap_ms, data = df)
summary(model)

prova = predict(model, df)
prova = data.frame(prova)
df_test_lm = data.frame(c(df, prova)) #creating df with predictions attached to original df
df_test_lm = df_test_lm[, c(1,4,7,8,24,15)]







#CLASSIFICATION (not yet runned, just written to be tested later)

# train model
#df.decision.train = rpart(formula, data=df_train, method="class")

# predict probabilities of class
#df.decision.test = predict(model.rpart, newdata=df_test, method="prob")

# we want to predict the probability of class=1 (winner) or class=0 (not winner)
#Then we are going to sort the probabilities and pick the greater probability of class=1, hence the driver with 
#the greatest probability of being a winner. 

