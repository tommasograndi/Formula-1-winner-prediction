#tests for LM MODELS

#linear regression
model = lm(positionOrder ~ grid + number + points + laps + fastestLapSpeed + round + const_points + driver_age + fastestLap_ms, data = df)
summary(model)

prova = predict(model, df)
prova = data.frame(prova)
df_test_lm = data.frame(c(df, prova)) #creating df with predictions attached to original df
df_test_lm = df_test_lm[, c(1,4,7,8,24,15)]
