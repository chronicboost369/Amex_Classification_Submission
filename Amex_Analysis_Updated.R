# Due to misunderstanding the data on Kaggle, this is the updated version

setwd("E:/jhk/R/AMEX_Competition")

library(tidyverse)
library(data.table)
library("RDRToolbox")
library("h2o")
library(kernlab)
library("ICS")
library("randomForest") ##for random forests (and bagging)
library(gbm) 
library("e1071")
library(rstatix)

data_cmbn <- merge(data,data_res, by="customer_ID") #combining label and train data.




chunks <-seq(1,nrow(data_cmbn), length= 40)



#write.csv(data_cmbn[1:chunks[2],],"E:/jhk/R/AMEX_Competition/train_1.csv", row.names = T)
data_c1 <- fread("train_1.csv") 


#EDA
many_na <- names(colSums(is.na(data_c1))[colSums(is.na(data_c1))>nrow(data_c1)*.1]) #removing variables with missing values for more than 10%

data_c1 <- (data_c1%>%select(-many_na)) #now, dim = 141833 x 153

colSums(is.na(data_c1)) # it looks like certain observations are missing many variables
## Intuitively, let's try removing all observations with at least 1 variables.

dim(na.omit(data_c1)) #122190 x 153

narows <- rowSums(is.na(data_c1)) #tracking narows for later purpose
data_c1 <- na.omit(data_c1)
data_c1 <- data_c1%>%select(-V1)

table(data_c1$B_31) # B_31 can be removed.
data_c1 <- data_c1 %>% select(-B_31)


## Selecting only numeric column
numcol <- which(sapply(data_c1,is.numeric) == TRUE)
data_c1_n <- data_c1%>%select(numcol) # 122190 x 147



#View(data_c1_n) # quick eyeballing everything looks numeric

sum(abs(cor(data_c1_n)) >= 0.9) - ncol(data_c1_n) # there are 28/2 variables that have high cor with each other


## Identifying highly correlated variables
highcor <- which(abs(cor(data_c1_n)) >= 0.9&lower.tri(abs(cor(data_c1_n))), arr.ind=T, useNames = F)
colnames(data_c1_n)[highcor[,2]] # decided to remove the 2nd column.
## Intuitively assumed that, the order of importance doesn't matter because correlation is >0.9.
## I think correlated variables can explain the response variable equally well.

data_c1_n <- data_c1_n%>%select(-highcor[,2]) #122190 x 136


correlations <- cor(data_c1_n$target, data_c1_n%>%select(-target))


meaningcor <- which(correlations> 0.4 | correlations < -0.4)

numcol <- names(data_c1_n%>%select(meaningcor))

data_set_cat <- data_c1%>%select(which(sapply(data_c1,is.numeric) != TRUE)) #just need D_63 & D_64

# Verifying D_63
table(data_c1[which(data_c1$target==1),"D_63"])/sum((data_c1$target==1)) * 100
table(data_c1[which(data_c1$target!=1),"D_63"])/sum((data_c1$target!=1)) * 100

# There isn't clear difference between the counts of each levels in D_63 to verify its usefulness on predicting the default rate.
# So, logistic regression is used.

logit_d_63 <- glm(as.factor(data_c1$target) ~ data_c1$D_63, family = "binomial")
summary(logit_d_63) 

1-pchisq(logit_d_63$null.deviance -logit_d_63$deviance,5) # p-value, so D_63 may be used.

# Verifying D_64
table(data_c1[which(data_c1$target==1),"D_64"])/sum((data_c1$target==1)) * 100
table(data_c1[which(data_c1$target!=1),"D_64"])/sum((data_c1$target!=1)) * 100

# D_64 is weird because it has blank value that is not recognized and -1.
# However, D_64 = U may be significant in predicting the default rate because there are higher portion of people who have D_64= U in default group
# than non-default group.
logit_d_64 <- glm(as.factor(data_c1$target) ~ data_c1$D_64, family = "binomial")
summary(logit_d_64) 

# Because of unrecognized vlaues in d_64, it's not used.
data_c1 <- data_c1%>%select(-D_64)




# Splitting Training & Testing
set.seed(12345)
index <- sample(1:nrow(data_c1), 0.7*nrow(data_c1))

data_c1_train <- data_c1[index,]
data_c1_test <- data_c1[-index,]




# Removing outliers
data_c1_3sdv <- remove_outliers(data_c1_train,3) #removing outliers that are greater than 3sds

dim(data_c1_3sdv) #32088 x 150


# Selecting predictors mainly b/c of computation limitation

data_c1_3sdv <- data_c1_3sdv%>%select(numcol,target,D_63) #32088 x 15


# Modelling
data_c1_3sdv$target <- as.factor(data_c1_3sdv$tar)
rf.train <- randomForest(data_c1_3sdv$target~.,mtry=sqrt(ncol(data_c1_3sdv)-1), importance = T, ntree=600, data=data_c1_3sdv)
rf.pred <- predict(rf.train,newdata=data_c1[-index,], type="prob")
mean(ifelse(rf.pred[,2]>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 86.14%


# What if removing the outliers were set to 2sd?

#data_c1_2sdv <- remove_outliers(data_c1,2)
#data_c1_2sdv$target <- as.factor(data_c1_2sdv$target)

#rf.train_2sd <- randomForest(data_c1_2sdv$target~.,mtry=sqrt(ncol(data_c1_2sdv)-1), importance = T, ntree=600, data=data_c1_2sdv )
#rf.pred_2sd <- predict(rf.train_2sd,newdata=data_c1[-index,], type="prob")
#mean(ifelse(rf.pred_2sd[,2]>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 81.41%



## What if not subsetting predictors even if it may take longer?

data_c1_noout <- remove_outliers(data_c1_train,3)
data_c1_noout$target <- as.factor(data_c1_noout$target)
rf.train_noout <- randomForest(data_c1_noout$target~.,mtry=sqrt(ncol(data_c1_noout)-1), importance = T, ntree=600, data=data_c1_noout%>%select(-customer_ID))
rf.pred_noout <- predict(rf.train_noout,newdata=data_c1[-index,], type="prob")
mean(ifelse(rf.pred_noout[,2]>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 86.85%

top10 <- names(sort(importance(rf.train_noout)[,3],decreasing = T))[1:10] # top 10 variables.
varImpPlot(rf.train_noout)

colnames(data_c1_3sdv)

data_c1_3sdv%>%
  select(-D_63)%>%
  group_by(target)%>%
  summarize_all(mean)

data_c1_3sdv%>%
  select(-D_63)%>%
  group_by(target)%>%
  summarize_all(median)

# The difference in mean and median between the two groups suggest that these variables may explain the defaults, which aligns with the variable selection
# done via correlation.

# The model contains 14 dimensions, which may lead to the curse of dimensionality. Before reducing the dimension, let's check if randomforest is still the 
# best method.

# Logistic
logistic <- glm(target ~., data=data_c1_3sdv, family= "binomial")
summary(logistic)


1-pchisq(logistic$null.deviance-logistic$deviance,ncol(data_c1_3sdv)-1) # p-value of 0. The model "logistic" is better than the null model.

logistic_pred <- predict(logistic,newdata=data_c1[-index,], type="response")
logistic_pred_convert <- ifelse(logistic_pred>0.5,1,0)
sum(logistic_pred_convert == data_c1[-index,"target"]) / nrow(data_c1[-index,]) #85.8%


# Boosting
#new_data_set_training_boosting <- data_c1_3sdv
#new_data_set_training_boosting$target <- as.numeric(new_data_set_training_boosting$target) -1
#new_data_set_training_boosting$D_63 <- as.factor(new_data_set_training_boosting$D_63) 
#boost <- gbm(target~., data=new_data_set_training_boosting, distribution="bernoulli", n.trees=300)
#summary(boost)

#boost_pred <- predict(boost,data_c1[-index,], n.trees = 300, typ="response")
#mean(ifelse(boost_pred>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 85.4%


# Radial SVM
#svm_model <- svm(target~.,data=data_c1_3sdv, kernel="radial", probability=T)
#svm_pred <- predict(svm_model,newdata=data_c1[-index,])
#mean(ifelse(svm_pred==1,1,0) == data_c1[-index,"target"]) # accuracy = 83.79%



# PCA 

pca <- prcomp(data_c1_3sdv%>%select(-c(D_63,target)),scale.=T)

plot(cumsum(pca$sdev)/sum(pca$sdev))

# PCA didn't yield a deseriable result. Let's check kernel PCA.

# Kernel PCA
# Not working due to computational limitation.


# Improving randomforest

top10rf <- randomForest(data_c1_noout$target~.,mtry=sqrt(ncol(data_c1_noout%>%select(top10))-1), importance = T, ntree=600, data=data_c1_noout%>%select(top10))
rf.pred_top10<- predict(top10rf,newdata=data_c1[-index,], type="prob")
mean(ifelse(rf.pred_top10[,2]>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 87.06%


# So far, top10rf performed the best. Tuning parameter.

top10rf <- randomForest(data_c1_noout$target~.,mtry=sqrt(ncol(data_c1_noout%>%select(top10))-1), 
                        importance = T, ntree=1000, nodesize=1,  
                        data=data_c1_noout%>%select(top10))
rf.pred_top10<- predict(top10rf,newdata=data_c1[-index,], type="prob")
mean(ifelse(rf.pred_top10[,2]>0.5,1,0) == data_c1[-index,"target"]) # Accuracy is 87.06%





# on all dataset

data_cmbn_top10 <- data_cmbn%>%select(top10,customer_ID,target)

# Exploring missing observations
colSums(is.na(data_cmbn_top10))

# Imputing missing observations with knn
# To avoid, the curse of dimensionality, identify correlations to find appropriate variables.

sapply(data_cmbn_top10,is.numeric) # all numeric except customer_id, and of course target
#lower.tri(abs(cor(data_cmbn_top10%>%select(-c(target,customer_ID))%>%na.omit())), arr.ind=T, useNames = F)
abs(cor(data_cmbn_top10%>%select(-c(target,customer_ID))%>%na.omit()))



# To compute missing variables for P_2, D_55 and D_44 are used.
# To computing missving variables for D_52, P_2 is used.
# For D_55, P_2, and D_44 are used
# For R_27, notsure.
# D_45, D_47 is used.
# D_44, P-2, and D_55 are used. 
# D_124, unsure.
# D_115, D_47 is used

# However, they don't account much of the total dataset so just remove and proceed.

data_cmbn_top10 <- na.omit(data_cmbn_top10)

set.seed(4543)
index <- sample(1:nrow(data_cmbn_top10), nrow(data_cmbn_top10)* 0.03)

data_cmbn_top10_train <- data_cmbn_top10[index,] #
data_cmbn_top10_train <- remove_outliers(data_cmbn_top10_train,3) #3187905x 12
data_cmbn_top10_train$target <- as.factor(data_cmbn_top10_train$target)

rf.all <- randomForest(target~.,mtry=sqrt(length(top10)), importance = T, ntree=600, data=data_cmbn_top10_train%>%select(-customer_ID)) #can't handle the size.
rf.pred_all<- predict(rf.all,newdata=data_cmbn_top10[-index,], type="prob")
mean(ifelse(rf.pred_all[,2]>0.5,1,0) == data_cmbn_top10[-index,"target"]) # Accuracy is 85.64%

#given that test data was much larger than the training dataset due to the memory limitation, the model is performed at 85.64% accuracy.
# It is not that big compared to earlier with just 1/20th of the original for both training and test data combined.



# Test on the test data provided by AMEX
# Because the file is too large to read...

s <- seq(50000,40000000, length= 30)

amex_data <- list()

for(i in 1:length(s)){
  if (i ==1){
    amex_data[[i]] <- read_csv("test_data.csv", n_max=s[i])
  }
  else{
    amex_data[[i]] <- read_csv("test_data.csv", n_max=s[i], skip=s[i-1])
  }
}

a
