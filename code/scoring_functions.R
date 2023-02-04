

### SCORING FUNCTION FOR CLASSIFICATION (Bayes, Decision trees, Random Forests)

score_classification <- function(test, prediction){
  
  df.score.class = data.frame(test, prediction)
  colnames(df.score.class)[c(26,27)] = c('pred_0', 'pred_1')  
  
  racelist = unique(df.score.class$raceId)
  
  score = 0
  for (i in 1:length(racelist)){
    race = df.score.class[df.score.class$raceId %in% racelist[i], ]
    pred_winner_idx = which.max(race$pred_1)
    if (race$positionOrder[pred_winner_idx] == 1){
      score = score + 1
    }
  }
  
  score_ratio = score/length(racelist)
  return(score_ratio)
} 

score_classification(df.test, prediction)


### SCORING FUNCTION FOR REGRESSION

score_regression <- function(test, prediction){
  
  df.score.regres = data.frame(test, prediction)
  colnames(df.score.regres)[c(26)] = c('prediction') #substitute with 26
  
  racelist = unique(df.score.regres$raceId)
  
  score = 0
  for (i in 1:length(racelist)){
    race = df.score.regres[df.score.regres$raceId %in% racelist[i], ]
    pred_winner_idx = which.min(race$prediction)
    if (race$positionOrder[pred_winner_idx] == 1){
      score = score + 1
    }
  }
  
  score_ratio = score/length(racelist)
  return(score_ratio)
} 

score_regression(df.test, predict_lm)






