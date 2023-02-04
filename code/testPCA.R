
pca = prcomp(scale(df[,c(6,7,10,11,16,20,21,22,23,25)], scale=TRUE))
summary(pca) #first 7 PCAs explain 94% most of the model
pca$x[,1:7]

df.pca = data.frame(df, pca$x[,1:10])
colnames(df.pca)[26:35] = c('PCA1', 'PCA2','PCA3','PCA4','PCA5','PCA6','PCA7','PCA8','PCA9','PCA10')

df.train.pca = df.pca[df.pca$raceId %in% races_train_list, ]
df.test.pca = df.pca[df.pca$raceId %in% races_test_list, ]

#score PCA
score_PCA <- function(test, prediction){
  
  df.score.regres = data.frame(test, prediction)
  colnames(df.score.regres)[c(36)] = c('prediction') #substitute with 26
  
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

#regression with PCA
summary(lm(positionOrder ~ PCA1+PCA2+PCA3+PCA4+PCA5+PCA6+PCA7, data = df.train.pca))
predict_lm_pc = predict(lm(positionOrder ~ PCA1+PCA2+PCA3+PCA4+PCA5+PCA6+PCA7, data = df.train.pca), df.test.pca)
score_PCA(df.test.pca, predict_lm_pc) #60%

