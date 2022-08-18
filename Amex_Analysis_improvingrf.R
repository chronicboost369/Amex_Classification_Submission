rf <- randomForest(target~., data=new_data_set_training[index,-"customer_ID"], mtry=sqrt(ncol(new_data_set_training)-1), importance = T, ntree=600)

## checking variable importance
importance(rf)
varImpPlot(rf)

# RandomForest's importance plot suggest some variables to remove like "R_21" and "D_65".
# Before removing variables, checking the performance of RandomForest is needed.

rf.pred <- predict(rf,newdata=new_data_set_training[-index,], type="prob")
mean(ifelse(rf.pred[,2]>0.5,1,0) == new_data_set_training[-index,"target"]) # Accuracy is 93.19%




# Based on the importance plot, removing poorly performing models may improve the model to ease overfitting.

rf.2 <- randomForest(target~., data=new_data_set_training[index,-c("customer_ID","R_21","D_65")], mtry=sqrt(ncol(new_data_set_training)-1), importance = T, ntree=600)
rf2.pred <- predict(rf,newdata=new_data_set_training[-index,], type="prob")
mean(ifelse(rf2.pred[,2]>0.5,1,0) == new_data_set_training[-index,"target"]) # Accuracy is 92.14%

# No significant difference between them.


# ======================================================================================================================================================

# Putting Everything Together

# Due to computation power limitation, the model is built based on a fraction of the original dataset.
# If the test performance is similiar on other fractional dataset, it's safe to state that the current model is the best performing model.


("E:/jhk/R/AMEX_Competition/train_allsamp.csv", row.names = T)

for(i in 1:10){
  print(paste(paste("E:/jhk/R/AMEX_Competition/train_",i, sep=""),".csv",sep=""))
}

chunks
length(chunks)
dim(data)

nrow(data_res) == nrow(data)
