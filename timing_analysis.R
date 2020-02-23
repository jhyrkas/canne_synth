library(ggplot2)

df <- read.csv('analysis_files/timing.csv')

# annotate line should be adjusted per data
ggplot(data=df, aes(x=num_seconds, y=average_time, group=Method)) + 
  geom_line(aes(color=Method)) + geom_point(aes(shape=Method)) +
  geom_abline(colour='red', linetype=2) +
  annotate(geom="text", x=2.75, y=3, label="real-time", colour="red",
           size=3, family="Courier", fontface="bold", angle=65) + 
  xlab('Seconds of audio') + ylab('Generation time') +
  ggtitle('Average generation time by method') +
  theme(plot.title = element_text(hjust = 0.5), 
        panel.background = element_rect(fill = 'white', colour = 'white'),
        panel.grid.major = element_line(size = 0.5, linetype = 'solid', colour = "grey"),
        panel.grid.minor = element_line(size = 0.25, linetype = 'solid', colour = "grey"))  +
  scale_colour_discrete(name="Generational Method",
                      breaks=c("network_mod_changing", "network_mod_static", "vanilla_changing", "vanilla_static"),
                      labels=c("Network Mod. Variable Bias", "Network Mod. Static Bias", "Decoder Variable Encoding", "Decoder Static Encoding")) +
  scale_shape_discrete(name="Generational Method",
                        breaks=c("network_mod_changing", "network_mod_static", "vanilla_changing", "vanilla_static"),
                        labels=c("Network Mod. Variable Bias", "Network Mod. Static Bias", "Decoder Variable Encoding", "Decoder Static Encoding"))
