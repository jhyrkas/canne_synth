library(ggplot2)

df <- read.csv('analysis_files/timing.csv')

ggplot(data=df, aes(x=num_seconds, y=average_time, group=Method)) + 
  geom_line(aes(color=Method)) + geom_point(aes(shape=Method)) +
  ggtitle('Average generation time by method') +
  theme(plot.title = element_text(hjust = 0.5))
