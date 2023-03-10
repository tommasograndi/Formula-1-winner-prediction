
#CORRELATION MATRIX for numerical features
corr_matrix = cor(df.results.merged[c(7,8,9,10,11,16,20,21,22,23)])
corrplot(corr_matrix)


### WINNER FOR EACH YEAR
#create dataframe for driver points
df_points = merge(driver_standings, races[, c(1,2)], by = 'raceId') #merging races info
df_points = merge(df_points, df[, c(1, 2, 17, 24)], by = c('raceId', 'driverId')) 

#Writing a function fo find winner of F1 for each year (extracting age, points, name, driveId)
winner_Age = function(){
  df_year_winner = matrix(nrow = 1, ncol=5)
  years = seq(1958, 2022, 1)
  for (y in years){
    
    df_year = df_points[df_points$year == y, c(2,4,9,10)]
    id_winner = df_year[which.max(df_year$points), ]
    df_year_winner <- rbind(df_year_winner, c(y, id_winner$driverId, id_winner$points, id_winner$fullname, id_winner$driver_age))
    
  }
  df_year_winner = data.frame(df_year_winner)
  df_year_winner = df_year_winner[-1,]
  return(df_year_winner)
}
df_year_winner = winner_Age()
df_year_winner

#plot for winners age during time
names(df_year_winner) = c('year', 'driverId', 'final_points', 'fullname', 'age')
ggplot(df_year_winner, aes(x = year, y = age)) + 
  labs(x = 'Years', y = 'Age') +
  ggtitle('Winning driver age for every year') +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  geom_abline(slope = 0, intercept = mean(df_year_winner$year), color = 'red', size = 1.5) + theme_bw() + 
  geom_text(aes(label = year), vjust = -1, size=5)

  
#convert fastest lap time in number of milliseconds 
df$fastestLap_ms = as.numeric(lubridate::ms(as.character(df$fastestLapTime)))*1000
df$fastestLap_ms[is.na(df$fastestLap_ms)] <- 0


# how important is the pole position (ratio between how many times a racer qualified first and win the race)
df_pole <- merge(df, qualifying[, c(2:4, 6)], by = c('raceId', 'driverId', 'constructorId'))
colnames(df_pole)[25] = 'qual_position' #renaming column position in qual_position
df_pole <- subset(df_pole, qual_position == 1)
pole_ratio <- matrix(ncol = 2)
circuits_list = circuits$name #assign circuits names to a list
for (i in circuits_list) {
  df_temp <- subset(df_pole, df_pole$name == i)
  pole_ratio <- rbind(pole_ratio, c(i, nrow(subset(df_temp, df_temp$positionOrder == 1))/nrow(df_temp)))
}
pole_ratio = data.frame(pole_ratio)
pole_ratio = pole_ratio[-1,]
pole_ratio$X2 = as.numeric(pole_ratio$X2)
pole_ratio <- na.omit(pole_ratio)
pole_ratio <- subset(pole_ratio, X2 != 0)

ggplot(pole_ratio, aes(x = X1, y = X2)) + 
  labs(x = 'Circuit', y = 'Ratio') +
  ggtitle('How important is the pole position') +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))



for (i in 2:length(circuits)) {
  df_temp1 <- subset(df, df$name == circuits[i])
  fast_speed <- max(df_temp1$fastestLapSpeed)
  row_t <- data.frame(Circuit = circuits[i], Max_Speed = fast_speed)
  df_speed <- rbind(df_speed, row_t)
}
df_speed <- subset(df_speed, df_speed$Max_Speed > 0)

conta = 0
for (i in (1:23693)){
  if (df$positionOrder[i] == 1) {
    if (df$positionOrder[i] == df$grid[i]){
      conta = conta + 1
    }
  }
}
print(conta/1012)


