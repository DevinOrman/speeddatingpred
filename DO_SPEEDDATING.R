#Devin Orman - LIS4805 Final Project
#Speed Dating Analysis



#LIBRARIES
require(dataQualityR) #checking missing values
require(boot) #for k fold cross validation
require(ggplot2) #for plotting
require(caret) #confusion matrix

#add dataset
S_D<-read.csv("Final/Speed Dating Data.csv")

View(S_D)



#####################  CLEAN DATA #####################

#check data

checkDataQuality(data= S_D, out.file.num="Final/DQR_num.csv" , out.file.cat="Final/DQR_cat.csv" )
dqr.num <- read.csv("Final/DQR_num.csv")
dqr.cat  <- read.csv("Final/DQR_cat.csv")
View(dqr.num)
View(dqr.cat)

#trim dataset to identifying and predictor variables

var_predict <- c("iid", "id", "pid", "gender", "order", "wave", "position", "round", "partner", "match", "samerace",
                 "age", "age_o", "field_cd", "imprace", "imprelig", "date",
                 "go_out", "goal","career_c", "exphappy", "attr", "attr_o", "sinc",
                 "sinc_o", "intel", "intel_o", "fun", "fun_o","shar","shar_o", "dec",
                 "dec_o", "like", "like_o", "prob", "prob_o", "met", "met_o")


S_D <- subset(S_D, select = var_predict)

#find and impute missing id and pid

fixid_iid <- S_D[is.na(S_D$id),]$iid
fixid_iid
S_D[is.na(S_D$id),]$id <- 22

fixpid_iid <- S_D[is.na(S_D$pid),]$iid
fixpid_iid
S_D[is.na(S_D$pid),]$pid <- 111

#impute missing values with variable means

S_D[is.na(S_D$imprace),]$imprace <- mean(S_D$imprace, na.rm = T)
S_D[is.na(S_D$imprelig),]$imprelig <- mean(S_D$imprelig, na.rm = T)
S_D[is.na(S_D$goal),]$goal <- mean(S_D$goal, na.rm = T)
S_D[is.na(S_D$go_out),]$go_out <- mean(S_D$go_out, na.rm = T)
S_D[is.na(S_D$field_cd),]$field_cd <- mean(S_D$field_cd, na.rm = T)
S_D[is.na(S_D$date),]$date <- mean(S_D$date, na.rm = T)
S_D[is.na(S_D$exphappy),]$exphappy <- mean(S_D$exphappy, na.rm = T)
S_D[is.na(S_D$career_c),]$career_c <- mean(S_D$career_c, na.rm = T)
S_D[is.na(S_D$age),]$age <- mean(S_D$age, na.rm = T)
S_D[is.na(S_D$age_o),]$age_o <- mean(S_D$age_o, na.rm = T)
S_D[is.na(S_D$attr),]$attr <- mean(S_D$attr, na.rm = T)
S_D[is.na(S_D$attr_o),]$attr_o <- mean(S_D$attr_o, na.rm = T)
S_D[is.na(S_D$like),]$like <- mean(S_D$like, na.rm = T)
S_D[is.na(S_D$like_o),]$like_o <- mean(S_D$like_o, na.rm = T)
S_D[is.na(S_D$sinc),]$sinc <- mean(S_D$sinc, na.rm = T)
S_D[is.na(S_D$sinc_o),]$sinc_o <- mean(S_D$sinc_o, na.rm = T)
S_D[is.na(S_D$intel),]$intel <- mean(S_D$intel, na.rm = T)
S_D[is.na(S_D$intel_o),]$intel_o <- mean(S_D$intel_o, na.rm = T)
S_D[is.na(S_D$prob),]$prob <- mean(S_D$prob, na.rm = T)
S_D[is.na(S_D$prob_o),]$prob_o <- mean(S_D$prob_o, na.rm = T)
S_D[is.na(S_D$fun),]$fun <- mean(S_D$fun, na.rm = T)
S_D[is.na(S_D$fun_o),]$fun_o <- mean(S_D$fun_o, na.rm = T)
S_D[is.na(S_D$met),]$met <- mean(S_D$met, na.rm = T)
S_D[is.na(S_D$met_o),]$met_o <- mean(S_D$met_o, na.rm = T)
S_D[is.na(S_D$shar),]$shar <- mean(S_D$shar, na.rm = T)
S_D[is.na(S_D$shar_o),]$shar_o <- mean(S_D$shar_o, na.rm = T)


View(S_D)
attach(S_D)
#####################   LOGISTIC REGRESSION   #####################

#training

set.seed(1)
sample<-sample.int(n = nrow(S_D), size = floor(.70*nrow(S_D)), replace = F)
train<-S_D[sample,]
test<-S_D[-sample,]


#plotting variables examples

ggplot(train, aes(factor(match), attr)) + geom_boxplot() #possible positive association with match
ggplot(train, aes(factor(match), goal)) + geom_boxplot() #no sign of correlation

#build models

glm1.fit <- glm(match ~+attr+attr_o+fun+fun_o+shar+shar_o+prob+prob_o,family=binomial, data=train)
summary(glm1.fit)
glm2.fit <- glm(match ~+attr+fun+shar,family=binomial, data=train)
summary(glm2.fit)
glm3.fit <- glm(match ~attr+fun+shar+prob,family=binomial, data=train)
summary(glm3.fit)

#resampling models
cv.err.glm1.fit <- cv.glm(train, glm1.fit, K=10)
cv.err.glm2.fit <- cv.glm(train, glm2.fit, K=10)
cv.err.glm3.fit <- cv.glm(train, glm3.fit, K=10)

cv.err.glm1.fit$delta[1]
cv.err.glm2.fit$delta[1]
cv.err.glm3.fit$delta[1]

#glm1.fit showed the least variance

#predict test models

glm1.prob<-predict.glm(glm1.fit, test, type = "response")
glm1.prob[1:5]

# 0 = no match 1 = yes match .5 = classifier cut off
glm1.pred <- rep(0, 2514)
glm1.pred[glm1.prob>0.5] <- 1
summary(glm1.pred)


###################   TEST PREDICTED MODEL   #######################
confusionMatrix(data=glm1.pred, test$match)

error.perc<-377/2514
error.perc

