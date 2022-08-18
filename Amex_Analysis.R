setwd("E:/jhk/R/AMEX_Competition")

library(tidyverse)
library(data.table)
library("RDRToolbox")
library("h2o")
data_res <- fread("train_labels.csv") # 458913 X 2

dim(data_res)
# The orginal data is large so trying to load them in chunks
chunks <- seq(1,nrow(data_res), by=22945)
chunks[21] <- 458913

data <- fread("train_data.csv")




# creating smaller dataset with the first 22946 rows for EDA
#write.csv(data[1:chunks[2],],"E:/jhk/R/AMEX_Competition/train_1.csv", row.names = T)

set.seed(4545)
samp.index <- sample(1:nrow(data), nrow(data)*.3)

#write.csv(data[samp.index,],"E:/jhk/R/AMEX_Competition/train_allsamp.csv", row.names = T) taking too long
data_c1 <- fread("train_1.csv") 

# EDA
## removing variables with more than 1000 
many_na <- names(colSums(is.na(data_c1))[colSums(is.na(data_c1))>1000]) 

data_c1 <- (data_c1%>%select(-many_na)) #now, dim = 22946 x 149

colSums(is.na(data_c1)) # it looks like certain observations are missing many variables
## Intuitively, let's try removing all observations with at least 1 variables.

dim(na.omit(data_c1)) #20687 x 149


narows <- rowSums(is.na(data_c1)) #tracking narows for later purpose
data_c1 <- na.omit(data_c1)


table(data_c1$B_31) # B_31 can be removed.
data_c1 <- data_c1 %>% select(-B_31)


## Selecting only numeric column
numcol <- which(sapply(data_c1,is.numeric) == TRUE)
data_c1_n <- data_c1%>%select(numcol) # 20687 x 144

## Verifying if all columns in data_c1
max <- apply(data_c1_n,2,max) 
min <- apply(data_c1_n,2,min)

as.numeric(max) - as.numeric(min) # nothing abnormal that signalizes nonnumeracy 

#View(data_c1_n) # quick eyeballing everything looks numeric

sum(abs(cor(data_c1_n)) >= 0.9) - ncol(data_c1_n) # there are 32/2 variables that have high cor with each other


## Identifying highly correlated variables
highcor <- which(abs(cor(data_c1_n)) >= 0.9&lower.tri(abs(cor(data_c1_n))), arr.ind=T, useNames = F)
colnames(data_c1_n)[highcor[,2]] # decided to remove the 2nd column.
## Intuitively assumed that, the order of importance doesn't matter because correlation is >0.9.
## I think correlated variables can explain the response variable equally well.

data_c1_n <- data_c1_n%>%select(-highcor[,2]) #20687 x 131




# Machine Learning

## Subsetting rows equivalent to data_c1_n
data_res_1 <- data_res[1:chunks[4]] 
data_res_1 <- data_res_1[-which(narows>0),]

dim(data_res_1);dim(data_c1_n)
data_1_cb <- cbind(data_res_1,data_c1_n) #combining data_res_1 and data_c1_n
data_1_cb$target <- as.factor(data_1_cb$target)

#log_reg <- glm(target~., family = "binomial", data=data_1_cb) # logreg taking forever due to the datasize
# This is only 1/20th size of the original but already taking long time.
# Dimension reduction is needed.

# Dimensionality Reduction 
## PCA
pca <- prcomp(data_c1_n, scale. = T, center=T)

plot(y=cumsum(pca$sdev)/ sum(pca$sdev), x=1:ncol(data_c1_n))

# The cumulative sum of components show the explanation of variance. The plot looks like square root-like function, which makes it
# difficult to select certain amount of components. There isn't really a hard line where the increase in the explanation of variance
# per component decreases substantially. In this case, it's safe to assume that either the relationship between variables is nonlinear.
# Therefore, PCA may not be appropriate for dimension reduction.

# https://www.aanda.org/articles/aa/pdf/2015/04/aa24194-14.pdf states that locally linear embedding outperforms PCA for dimensionality reduction.

## LLE
### source: https://cs.nyu.edu/~roweis/lle/papers/lleintro.pdf
### source: https://cs.nyu.edu/~roweis/lle/algorithm.html

#lle_compression  <- LLE(as.matrix(data_c1_n), dim=30, k=100)

# However, LLE wasn't sucessful because the computation was taking long time even with the dataset size of 1/20th of the original.
# So, autoencoder is tried to reduce the dataset.

## Autoencoder
# Contrary to PCA, autoencoder can reduce the dimension nonlinearly with appropriate functions. 
# In the simplest form, autoencoder has decoder and encoder. Encoder reduces the dimension then reconstructs the data with decoder.
# If reduced x dimensions can reconstruct the original dataset, one can assume that such reduced dimension is good to represent
# the original dataset.

# Such dimension reduction is done by constraining the dimension of inner layer less than the dimension of the input data, which
# is known as undercomplete autoencoder. This forces the autoencoder to capture most dominant features of the data.
# Therefore, the goal of autoencoder is to minimize the loss function, which is really just comparision between the original and the 
# reconstructed dataset. 

auto_data <- as.h2o(data_c1_n[,2:ncol(data_c1_n)])


auto_train_tanh <- h2o.deeplearning(
  x = seq_along(auto_data),
  training_frame = auto_data,
  autoencoder = TRUE,
  hidden = 50,
  activation = 'Tanh',
  seed=20220808
)


auto_pred_tanh <- predict(auto_train_tanh, as.h2o(data_c1_n[,2:ncol(data_c1_n)]))

tanh_rmse <- apply((as.data.frame(data_c1_n[,2:ncol(data_c1_n)]) - as.data.frame(auto_pred_tanh))^2,2,function(x){sqrt(mean(x))})


## Because the data is not scaled for now, calculating the total rmse isn't applicable because each variables have different magnitude
# of values. For example, 1 is 10% of 10 but 1% of 100. The size of 1 is dependent on the magnitude of the values it represent.

# Solution is to compare them with the median of each variables.

tanh_rmse / apply(as.data.frame(data_c1_n[,2:ncol(data_c1_n)]), 2, median) * 100

# The result shows that autoencoder with 50 layers and tanh function didn't work well. Overall, RMSE is too large compared to medians.
# So, checking other activation is needed.


auto_train_tanhdo <- h2o.deeplearning(
  x = seq_along(auto_data),
  training_frame = auto_data,
  autoencoder = TRUE,
  hidden = 50,
  activation = 'TanhWithDropout',
  seed=20220808
)


auto_pred_tanhdo <- predict(auto_train_tanhdo,as.h2o(data_c1_n[,2:ncol(data_c1_n)]))
tanhdo_rmse <- apply((as.data.frame(data_c1_n[,2:ncol(data_c1_n)]) - as.data.frame(auto_pred_tanhdo))^2,2,function(x){sqrt(mean(x))})

tanhdo_rmse / apply(as.data.frame(data_c1_n[,2:ncol(data_c1_n)]), 2, median) * 100

# Tanhwithdropout wasn't successful as well.
# Problem: Out of available activation functions in h2o, only tanh and tanhwithdropout work to fit the data.

# Another solution: use stacked autoencoder.

# Stacked autoencoder is autoencoder with multiple layers.
# For example, if a dataset contains 1000 variables, a stacked autoencoder would look like endcoding to 500 to 250 to 50 dimensions.
# For reconstruction, it follows the reverse order.
# Adding additional depth can allow to fit more complex or nonlinear characteristics in the dataset.

grid <- list( hidden = list(c(50), c(100,50,100), c(100,75,25,75,100)))

auto_grid <- h2o.grid(
  algorithm = 'deeplearning',
  x = seq_along(auto_data),
  training_frame = auto_data,
  grid_id = 'autoencoder_grid',
  autoencoder = TRUE,
  activation = 'Tanh',
  hyper_params = grid,
  ignore_const_cols = FALSE,
  seed = 20220808
)

h2o.getGrid('autoencoder_grid', sort_by = 'mse', decreasing = FALSE)

auto_train_tanh_mult <- h2o.deeplearning(
  x = seq_along(auto_data),
  training_frame = auto_data,
  autoencoder = TRUE,
  hidden = c(100,75,25,75,100),
  activation = 'Tanh',
  seed=20220808
)

auto_pred_tanhmult <- predict(auto_train_tanh_mult,as.h2o(data_c1_n[,2:ncol(data_c1_n)]))
tanh_mult__rmse <- apply((as.data.frame(data_c1_n[,2:ncol(data_c1_n)]) - as.data.frame(auto_pred_tanhmult))^2,2,function(x){sqrt(mean(x))})
tanh_mult__rmse / apply(as.data.frame(data_c1_n[,2:ncol(data_c1_n)]), 2, median) * 100

# The result is still not good for dimensionality reduction.

# Current status Overview

# The original dataset has about about 400,000 observations and 149 variables after removing certain variables with many missing values.
# The main problem with this dataset is that working memory on the computer I'm using cannot handle the dataset.
# The initial solution for this was to reduce the dimensionality while capturing the dominant directions of variance.
# To accomplish this, PCA and autoencoder were used; however, they both failed to reduce the dimensionality.

# Picking dominant variables that can distinguish people who are more likely to declare defaults.

## Going back to data_c1



