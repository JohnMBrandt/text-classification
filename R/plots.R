log <- read.csv("log.csv")

library(ggplot2)

ggplot(data = log, aes(x= epoch, y = top_3_accuracy*100))+
  geom_smooth(se=F, aes(color = "Train"))+
  geom_smooth(aes(y=val_top_3_accuracy*100, color = "Validation"), se = F)+
  theme_bw()+
  ylab("Top 3 accuracy")+
  xlab("Epoch")+
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(legend.title = element_blank(),
        legend.position=c(0.1,0.93))
  
