#creating df with only numerical features
merged_numeric = df.results.merged[, c(1:6,7,8,9,10,12,19,23,24,25)]

# adding 0s when there is NA in fastestLapSpeed
df["fastestLapSpeed"][is.na(df["fastestLapSpeed"])] <- 0


#CORRELATION MATRIX for numerical features
cm_numeric = cor(merged_numeric[c(7:15)])
corrplot(cm_numeric)


#create dataframe for driver points
df_points = merge(driver_standings, races[, c(1,2)], by = 'raceId') #merging races info
df_points = merge(df_points, df[, c(1, 3, 17, 24)], by = c('raceId', 'driverId')) 

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
ggplot(df_year_winner, aes(x = X1, y = X5)) + 
  labs(x = 'Years', y = 'Age') +
  ggtitle('Winning driver age for every year') +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  geom_abline(slope = 0, intercept = mean(df_year_winner$X5), color = 'red', size = 1.5) + theme_bw() + 
  geom_text(aes(label = X5), vjust = -1, size=5)


  
#convert fastest lap time in number of milliseconds 
df$fastestLap_ms = as.numeric(lubridate::ms(as.character(df$fastestLapTime)))*1000
df$fastestLap_ms[is.na(df$fastestLap_ms)] <- 0
df = df[, -c(11)]  #dropping column





