
#ALTERNATIVE DATA TO TEST MODELS

#DF.NEW definition
df.new = merge(df, driver_standings[, c(2,3,7)], by = c('raceId', 'driverId'))
df.test.new = df.new[df.new$raceId %in% races_test_list, ]
df.train.new = df.new[df.new$raceId %in% races_train_list, ]

#DF.REDUCED definition
df.reduced = merge(df.new, pit_stops[,c(1,2,3,6)], by = c('raceId', 'driverId'))

fanculo = unique(df.reduced$raceId)
idx = sample(length(unique(df.reduced$raceId)), length(unique(df.reduced$raceId))*0.8)
train_list = fanculo[idx]
test_list = fanculo[-idx]


df.train.reduced = df.reduced[df.reduced$raceId %in% train_list, ]
df.test.reduced = df.reduced[df.reduced$raceId %in% test_list, ]


predict_lm_reduced = predict(
  lm(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms + wins + stop + duration, data = df.train.reduced)
  , df.test.reduced)
predict_lm_reduced = data.frame(predict_lm_reduced)

score_regression(df.test.reduced, predict_lm_reduced)  #4% better than original dataframe (58%)

bayes_reduced <- predict(naiveBayes(winner ~ . -positionOrder -resultId -points, data = df.train.reduced),
                         newdata = df.test.reduced, type = 'raw')

score_classification(df.test.reduced, bayes_reduced)    #52%

#test with regression tree
library(tree)
regression_tree = predict(tree(positionOrder ~ grid + number + laps + fastestLapSpeed + round + const_points + const_wins + driver_age + fastestLap_ms + wins + stop + duration
                               , df.train.reduced), newdata = df.test.reduced)
score_regression(df.test, regression_tree)



