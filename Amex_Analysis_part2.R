library("ICS")
library("randomForest") ##for random forests (and bagging)
library(gbm) 
library("e1071")
# Goal

# Previously, dimension reduction wasn't successful. So in order to overcome the problem related to dimensions,
# variables that are meaningfully different between people's default status are selected.
# For numeric variables, the correlation is checked.
# For the categorical variables, the counts of two groups(default) are compared. 






# repreparing data

data_res <- fread("train_labels.csv") # 458913 X 2

data <- fread("train_data.csv")

# creating smaller dataset with the first 22946 rows for EDA
#write.csv(data[1:chunks[2],],"E:/jhk/R/AMEX_Competition/train_1.csv", row.names = T)


data_c1 <- fread("train_1.csv")


many_na <- names(colSums(is.na(data_c1))[colSums(is.na(data_c1))>1000]) 

data_c1 <- (data_c1%>%select(-many_na)) 



narows <- rowSums(is.na(data_c1)) #tracking narows for later purpose
data_c1 <- na.omit(data_c1)

data_c1 <- data_c1%>%select(-V1)


data_set <- merge(data_res,data_c1,by="customer_ID")




data_set_num <- data_set%>%select(which(sapply(data_set,is.numeric) == TRUE))


correlations <- cor(data_set_num$target, data_set_num[,2:ncol(data_set_num)]) # correlations between "target" and other numeric variables.


meaningcor <- which(correlations> 0.4 | correlations < -0.4)

numcol <- names(data_set_num%>%select(meaningcor))

data_set_cat <- data_set%>%select(which(sapply(data_set,is.numeric) != TRUE)) #just need D_63 & D_64


# Verifying D_63
table(data_set[which(data_set$target==1),"D_63"])/sum((data_set$target==1)) * 100
table(data_set[which(data_set$target!=1),"D_63"])/sum((data_set$target!=1)) * 100

# There isn't clear difference between the counts of each levels in D_63 to verify its usefulness on predicting the default rate.
# So, logistic regression is used.

logit_d_63 <- glm(as.factor(data_set$target) ~ data_set$D_63, family = "binomial")
summary(logit_d_63) 

# In the tables of D_63, D_63 == CR may have some effectiveness on the probability of declaring default because there are higher portion of people
# who have D_63 = CR in the tables, and it's only only predictor that is significant in logistic regression(informal)

# Verifying D_64
table(data_set[which(data_set$target==1),"D_64"])/sum((data_set$target==1)) * 100
table(data_set[which(data_set$target!=1),"D_64"])/sum((data_set$target!=1)) * 100

# D_64 is weird because it has blank value that is not recognized and -1.
# However, D_64 = U may be significant in predicting the default rate because there are higher portion of people who have D_64= U in default group
# than non-default group.
logit_d_64 <- glm(as.factor(data_set$target) ~ data_set$D_64, family = "binomial")
summary(logit_d_64) 

# However, logistic regression's z-test doesn't say anything about this. Hence, one can conclude that it may not be useful informally.

# So for now, the useful predictors are D_63 & other numeric columns found earlier with correlations.


data_set_training <- data_set%>%select(customer_ID,numcol,D_64)
dim(data_set_training) # Now the dataset is 20687 x 20, should be managable with the current computing power.


###################################################################################################
# Training and Test Data
data_set_training$target <- as.factor(data_set_training$target)

set.seed(4545)
index <- sample(1:nrow(data_set_training), nrow(data_set_training)*.7)

# Classification Algorithm

## Logistic Regression:
#logistic <- glm(target ~., data=data_set_training[index,-"customer_ID"], family= "binomial")
#summary(logistic)
# Error: (converted from warning) glm.fit: fitted probabilities numerically 0 or 1 occurred
# The error is caused probably because of outliers.

## Removing outliers


numcols <- which(sapply(data_set_training,is.numeric) == T)

boxplot(data_set_training%>%select(numcols)) # The rough sketch of boxplot shows presence of outliers.


## Because the dataset is large, I think we don't have to worry about removing outliers.
## To expedite the process, first begin with removing outliers that are more than 3 sd away from the mean.




remove_3sdvoutliers <- function(data){
  
  numcols <- which(sapply(data,is.numeric) == T)
  means <- apply(data%>%select(numcols),2,mean)
  sdv <- apply(data%>%select(numcols),2,sd)
  print(sdv)
  rows <- NULL
  for(i in 1:length(numcols)){
    index <- which(data%>%select(numcols[i]) >= (means[i]+ (2*sdv[i])) |  data%>%select(numcols[i]) <= (means[i] - (2*sdv[i])))
    rows <- append(rows,index)
      
    }
  rows <- unique(rows)
  data[-rows,]
  
  }

new_data_set_training <- remove_3sdvoutliers(data_set_training)

new_data_set_training$D_64 <- as.factor(new_data_set_training$D_64)
  
### Redoing Logistic Regression


set.seed(123456)
index <- sample(1:nrow(new_data_set_training), nrow(new_data_set_training)*.7)
logistic <- glm(target ~., data=new_data_set_training[index,-"customer_ID"], family= "binomial")
summary(logistic)


1-pchisq(logistic$null.deviance-logistic$deviance,ncol(new_data_set_training[index,-"customer_ID"])-1) # p-value of 0. The model "logistic" is better than the null model.

logistic_pred <- predict(logistic, newdata=new_data_set_training[-index,-"customer_ID"], type="response")

logistic_pred_convert <- ifelse(logistic_pred>0.5,1,0)

sum(logistic_pred_convert == new_data_set_training[-index,"target"]) / nrow(new_data_set_training[-index,]) #86.41%

# Logistic Regression Model has 53.89% accuracy rate.

### Logistic Regression improvement

## based on the output of summary, there are some predictors that had high p-values so want to remove them to see if improves the model.
## Theoratically, one should remove them one by one, but this is only about the accuracy, I'm going to remove both of them at once. Also,
## They have extremely high p-value so I think removing them at once won't affect individual's z-test much.

# Removing D_65, B_16, B_18, D_64
logistic.v2 <- glm(target ~., data=new_data_set_training[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64")], family= "binomial")
summary(logistic.v2)
1-pchisq(logistic.v2$null.deviance-logistic.v2$deviance,ncol(new_data_set_training[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64")])-1) # pvalue=0


logistic.v2_pred <- predict(logistic.v2, newdata=new_data_set_training[-index,], type="response")
logistic.v2_pred_convert <- ifelse(logistic.v2_pred>0.5,1,0)
sum(logistic.v2_pred_convert == new_data_set_training[-index,"target"]) / nrow(new_data_set_training[-index,]) #accuracy is now 86.58%


# Because the previous method improved the accuracy on the test data so trying one more time.
# Removing R_21
logistic.v3 <- glm(target ~., data=new_data_set_training[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64", "R_21")], family= "binomial")
summary(logistic.v3)
1-pchisq(logistic.v3$null.deviance-logistic.v3$deviance,ncol(new_data_set_training[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64")])-1) # pvalue=0

logistic.v3_pred <- predict(logistic.v3, newdata=new_data_set_training[-index,], type="response")
logistic.v3_pred_convert <- ifelse(logistic.v3_pred>0.5,1,0)
sum(logistic.v3_pred_convert == new_data_set_training[-index,"target"]) / nrow(new_data_set_training[-index,]) #accuracy is now 86.62%

# Removing R_21 improved the accuracy on the test data very slightly so let's disregard it.
# Thus, for the logistic regression the best model is now logistic.v2


# Trying higher degrees of polynomial on logistic.v2

summary(logistic.v2)

# The predictor with the highest absolute value of magnitude is B_6. Could transforming B_6 to higher degrees improve the model
# since the model is giving more weight to the one with the most impact?

B_6poly <- poly(new_data_set_training$B_6,3) #setting up polynomial for B_6.

new_data_set_training_poly <- new_data_set_training


# subsetting the predictors
new_data_set_training_poly$B_6 <- B_6poly[,1]
new_data_set_training_poly <- cbind(new_data_set_training_poly, B_6poly[,2:3])
colnames(new_data_set_training_poly)[21:22] <- c("B_6_poly2", "B_6_poly3") # Changing the colnames of B_6 polynomials


# Now, trying logistic regression again.
logistic.v2poly <- glm(target ~., data=new_data_set_training_poly[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64")], family= "binomial")
summary(logistic.v2poly)
1-pchisq(logistic.v2poly$null.deviance-logistic.v2poly$deviance,ncol(new_data_set_training[index,-c("customer_ID", "D_65", "B_16", "B_18", "D_64")])-1) # pvalue=0


logistic.v2poly_pred <- predict(logistic.v2poly, newdata=new_data_set_training_poly[-index,], type="response")
logistic.v2poly_pred_convert <- ifelse(logistic.v2poly_pred>0.5,1,0)
sum(logistic.v2poly_pred_convert == new_data_set_training_poly[-index,"target"]) / nrow(new_data_set_training_poly[-index,]) #accuracy is now 86.62%

## using polynomial on B_6 didn't improve much so continuing with "logistic.v2".

## LDA & QDA
# For both discrminant analyses, one critical assumption is that variables are multivariately normally distrubted.

ICS::mvnorm.kur.test(new_data_set_training[which(new_data_set_training$target==1),]%>%select(numcols)) # not multivariately normally distubted for kurtosis test
ICS::mvnorm.skew.test(new_data_set_training[which(new_data_set_training$target==1),]%>%select(numcols)) # not multivariately normally distubted for kurtosis test

# Because the dataset doesn't meet the multivariate normality assumption. LDA or QDA won't guarantee boundaries that differentiate people who are likely
# to declare defaults or not. Since there are still other algorithms left and logistic regression model is currently yielding 86% accuracy rate, LDA/QDA
# are not tested.

## Tree based methods
rf <- randomForest(target~., data=new_data_set_training[index,-"customer_ID"], mtry=sqrt(ncol(new_data_set_training)-1), importance = T)

## checking variable importance
importance(rf)
varImpPlot(rf)

# RandomForest's importance plot suggest some variables to remove like "R_21" and "D_65".
# Before removing variables, checking the performance of RandomForest is needed.

rf.pred <- predict(rf,newdata=new_data_set_training[-index,], type="prob")
mean(ifelse(rf.pred[,2]>0.5,1,0) == new_data_set_training[-index,"target"]) # Accuracy is 92.02%

# RandomForest substantially outperformed logistic regression.


# Improving Random Forest by finding optimal number of trees

tree <- 500
accuracy <- 0.89
treesize <- 500
accu.para <- 0.89
diff <- 0.01

while( diff > 0.0001){
  tree <- tree + 50
  set.seed(12345)
  rf <- randomForest(target~., data=new_data_set_training[index,-"customer_ID"], 
                     mtry=sqrt(ncol(new_data_set_training)-1), importance = T, ntree=tree) #1000 trees now
  rf.pred <- predict(rf,newdata=new_data_set_training[-index,], type="prob")
  accu <- mean(ifelse(rf.pred[,2]>0.5,1,0) == new_data_set_training[-index,"target"])
  
  diff <- accu- accu.para
  accu.para <- accu
  accuracy <- append(accuracy,accu.para)
  treesize <- append(treesize,tree)
  
  
}

accuracy
treesize

# randomforest with ntrees=600 performed the best, but only slightly.




## Boosting
new_data_set_training_boosting <- new_data_set_training
new_data_set_training_boosting$target <- as.numeric(new_data_set_training_boosting$target) -1
new_data_set_training_boosting$D_64 <- as.factor(new_data_set_training_boosting$D_64) 
boost <- gbm(target~., data=new_data_set_training_boosting[index,-"customer_ID"], distribution="bernoulli", n.trees=300)
summary(boost)

boost_pred <- predict(boost,newdata=new_data_set_training_boosting[-index,], n.trees = 300, typ="response")

mean(ifelse(boost_pred>0.5,1,0) == new_data_set_training[-index,"target"]) # Accuracy is 87.33%


## SVM

# Linear
svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="linear", probability=T)

svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"],)

mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) # accuracy = 86.83%

# polynomial
svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="polynomial", probability=T)

svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"],)

mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) # accuracy = 88.28%

# radial
svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="radial", probability=T)

svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"],)

mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) # accuracy = 88.72%

# sigmoid
svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="sigmoid", probability=T)

svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"],)

mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) # accuracy = 81.675%


# radial is the best performing svm classfier. Tuning radial svm.
set.seed(45454)
svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="radial", probability=TRUE, cost= 0.5)
svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"])
base_accu <- mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) # accuracy = 88.33%

cost <- 0.5
diff <- 10
cost_set <- 0.5
accu_set <- 0.8833

while(diff>0.005){
  cost = 0.5*1.05
  svm_model <- svm(target~.,data=new_data_set_training[index,-"customer_ID"], kernel="radial", probability=TRUE, cost= cost)
  svm_pred <- predict(svm_model,newdata=new_data_set_training[-index,-"customer_ID"])
  base_accu2 <- mean(ifelse(svm_pred==1,1,0) == new_data_set_training[-index,"target"]) 
  diff<- base_accu2 - base_accu
  base_accu <- base_accu2
  cost_set <- append(cost_set,cost)
  accu_set <- append(accu_set,base_accu2)
}

cost_set
accu_set



## Xgboost

xgboost_train <- xgb.DMatrix(data=data.matrix(new_data_set_training[index,-"customer_ID"]), 
                             label= data.matrix((new_data_set_training[index,"target"]))-1 )
xgboost_test <- xgb.DMatrix(data=data.matrix(new_data_set_training[-index,-"customer_ID"]), 
                             label= data.matrix(new_data_set_training[-index,"target"])-1 )

xgboost_fit <- xgboost(data=xgboost_train, max.depth=10,nrounds=50,objective= "binary:logistic")

xgboost_pred <- predict(xgboost_fit,xgboost_test, type="prob")

mean(ifelse(1 / (1+exp(xgboost_pred)) >0.5,1,0) == new_data_set_training[-index,"target"]) #81.21%

