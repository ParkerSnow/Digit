# Title     : TODO
# Objective : TODO
# Created by: Parker
# Created on: 7/31/2021

train <- read.csv("train.csv")
install.packages("ggcorrplot")
library(ggplot2)
library(ggcorrplot)

cor <- round(cor(train),1)

ggcorrplot(corr)