library(lars)
library(dplyr)
library(microbenchmark)
data("diabetes")
par(mfrow=c(2,2))
attach(diabetes)
object <- lars(x,y)
plot(object)
object2 <- lars(x,y,type="lar")
plot(object2)
object3 <- lars(x,y,type="forward.stagewise") # Can use abbreviations
plot(object3)
microbenchmark(
  LAR = lars(x,y,type="lar"),
  LASSO = lars(x,y),
  STAGEWISE = lars(x,y,type="forward.stagewise")
)
detach(diabetes)
data <- read.csv("data/PRSA_Data_Aotizhongxin_20130301-20170228.csv")
data <- data %>% na.omit()
x_ <- data %>% select(-c(No, year, month, day, hour, SO2, station, wd)) %>% as.matrix() 

microbenchmark(
  LAR = lars(x_, data$SO2, type="lar"),
  LASSO = lars(x_, data$SO2,  type="lasso"),
  STAGEWISE = lars(x_, data$SO2,type="forward.stagewise")
)

par(mfrow=c(2,2))
obj4 <- lars(x_, data$SO2, type="lar")
plot(obj4)
obj5 <- lars(x_, data$SO2, type="lasso")
plot(obj5)
obj6 <- lars(x_, data$SO2, type="forward.stagewise")
plot(obj6)

data <- read.csv("data/trainingData.csv")
data <- data %>% na.omit()
x_ <- data %>% select(-c(LATITUDE, LONGITUDE, TIMESTAMP)) %>% as.matrix()
ind = sample(1:500, 100)
x_ <- x_[,ind]
par(mfrow=c(2,2))
obj7 <- lars(x_, data$LONGITUDE, type="lar")
plot(obj7)
obj8 <- lars(x_, data$LONGITUDE, type="lasso")
plot(obj8)
obj9 <- lars(x_, data$LONGITUDE, type="forward.stagewise")
plot(obj9)


microbenchmark(
  LAR = lars(x_, data$LONGITUDE, type="lar"),
  LASSO = lars(x_, data$LONGITUDE,  type="lasso"),
  STAGEWISE = lars(x_, data$LONGITUDE,type="forward.stagewise"),
  times=10
)



data <- read.table("data/YearPredictionMSD.txt",sep=',')
data <- data %>% na.omit() %>% as.matrix()
x_ <- data[,2:90]
y_ <- data[,91]
lapply(colnames(x_), function(x) is.numeric(x_[,x]))
par(mfrow=c(2,2))
obj10 <- lars(x_, y_, type="lar")
plot(obj10)
obj11 <- lars(x_, y_, type="lasso")
plot(obj11)
obj12 <- lars(x_, y_, type="forward.stagewise")
plot(obj12)

obj10$beta
microbenchmark(
  LAR = lars(x_, y_, type="lar"),
  LASSO = lars(x_, y_,  type="lasso"),
  STAGEWISE = lars(x_, y_,type="forward.stagewise"),
  times=1
)
