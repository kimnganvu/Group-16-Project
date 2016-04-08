#setwd("~/Downloads/Parkinson_Multiple_Sound_Recording_Data")

# Loading libraries
library("kknn")
library("caret")
library("plyr")
library("ggplot2")

#read data from train_date.csv
train_data <- read.table("train_data.csv", sep = ',', header = T)

# Description of 26 samples - corresponding voice samples 
# 1: sustained vowel (aaa??????????????) 
# 2: sustained vowel (ooo???????...) 
# 3: sustained vowel (uuu???????...) 
# 4-13: numbers from 1 to 10 
# 14-17: short sentences 
# 18-26: words 

# For plotting the last graph
# Only take n (2-26) first voice sample of each subject_id (starting from 2 to be able to generate standard deviation)
# n are prone to change accordingly when generate the graph with different n
n = 26
train_data <- ddply(train_data, "subject_id", function(z) head(z,n))

# UPDRS is not included in the training attributes according to papers
train_data$UPDRS<-NULL

# SUMMARIZE DATA FOR EACH SUBJECT USING CENTRAL TENDENCY AND DISPERSION METRICS

# s-LOO(1-3): mean-sd 
mean_sum <- aggregate(train_data[, 2:28], list(train_data$subject_id), mean) #summarize using mean
sd_sum <- aggregate(train_data[, 2:27], list(train_data$subject_id), sd) #summarize using standard deviation

colnames(mean_sum) <- paste("mean", colnames(mean_sum), sep = "_")
colnames(sd_sum) <- paste("sd", colnames(sd_sum), sep = "_")
colnames(mean_sum)[colnames(mean_sum) == 'mean_Group.1'] <- 'subject_id'
colnames(sd_sum)[colnames(sd_sum) == 'sd_Group.1'] <- 'subject_id'
summ<- merge(sd_sum,mean_sum,by="subject_id")

summ1 <- data.frame(scale(summ[,-54])) # normalized
summ1$class <- summ[,54]

# Checking normalization
# colMeans(scaled.summ)
# library(matrixStats)
# colSds(data.matrix(scaled.summ))

# k-NN 
train <- summ1[,2:53]
cl <- summ1[,54]
out<-knn.cv(train, cl, k = 7, prob = TRUE)
result_knn <- out[0:40]
confusionMatrix(result_knn,cl)

# SVM
library(e1071)
tc <- tune.control(cross = 40)
#linear kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "linear")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out<-predict(model,summ1,probability=TRUE)
result_svm_linear <- out[0:40]
confusionMatrix(result_svm_linear,cl)


#RBF kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "radial")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out<-predict(model,summ1,probability=TRUE)
result_svm_rbf <- out[0:40]
confusionMatrix(result_svm_rbf,cl)

# s-LOO(2-4): median - mad
median_sum <- aggregate(train_data[, 2:28], list(train_data$subject_id), median) #summarize using median
mad_sum <- aggregate(train_data[, 2:27], list(train_data$subject_id), mad) #summarize using mean absolute deviation

colnames(median_sum) <- paste("median", colnames(median_sum), sep = "_")
colnames(mad_sum) <- paste("mad", colnames(mad_sum), sep = "_")
colnames(median_sum)[colnames(median_sum) == 'median_Group.1'] <- 'subject_id'
colnames(mad_sum)[colnames(mad_sum) == 'mad_Group.1'] <- 'subject_id'
summ<- merge(mad_sum,median_sum,by="subject_id")

summ1 <- data.frame(scale(summ[,-54])) # normalized
summ1$class <- summ[,54]
# colMeans(scaled.summ)
# library(matrixStats)
# colSds(data.matrix(scaled.summ))

# k-NN 
train <- summ1[,2:53]
cl <- summ1[,54]
out<-knn.cv(train, cl, k = 7, prob = TRUE)
result_knn <- out[0:40]
confusionMatrix(result_knn,cl)

# SVM
library(e1071)
tc <- tune.control(cross = 40)
#linear kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "linear")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out<- predict(model,summ1,probability=TRUE)
result_svm_linear <- out[0:40]
confusionMatrix(result_svm_linear,cl)

#RBF kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "radial")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out<-predict(model,summ1,probability=TRUE)
result_svm_rbf <- out[0:40]
confusionMatrix(result_svm_rbf,cl)

# s-LOO(3-6): (mean,trim = 0.25), IQR:interquartile range
meantr_sum <- aggregate(train_data[, 2:28], list(train_data$subject_id), mean, trim = 0.25) #summarize using trimmed mean (25% removed)
iqr_sum <- aggregate(train_data[, 2:27], list(train_data$subject_id), IQR) #summarize using interquartile range

colnames(meantr_sum) <- paste("meantr", colnames(meantr_sum), sep = "_")
colnames(iqr_sum) <- paste("iqr", colnames(iqr_sum), sep = "_")
colnames(meantr_sum)[colnames(meantr_sum) == 'meantr_Group.1'] <- 'subject_id'
colnames(iqr_sum)[colnames(iqr_sum) == 'iqr_Group.1'] <- 'subject_id'
summ<- merge(iqr_sum,meantr_sum,by="subject_id")

summ1 <- data.frame(scale(summ[,-54])) # normalized
summ1$class <- summ[,54]
# colMeans(scaled.summ)
# library(matrixStats)
# colSds(data.matrix(scaled.summ))

# k-NN 
train <- summ1[,2:53]
cl <- summ1[,54]
out<-knn.cv(train, cl, k = 7, prob = TRUE)
result_knn <- out[0:40]
confusionMatrix(result_knn,cl)

# SVM
library(e1071)
tc <- tune.control(cross = 40)
#linear kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "linear")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out <- predict(model,summ1,probability=TRUE)
result_svm_linear <- out[0:40]
confusionMatrix(result_svm_linear,cl)

#RBF kernel
obj <- best.tune(svm,class~., data = summ1, tunecontrol = tc, kernel = "radial")
summary(obj)
model <- svm(class~., data=summ1, type = "C-classification", cost = 1, gamma = 0.01886792 , epsilon = 0.1, probability=TRUE)
out<-predict(model,summ1,probability=TRUE)
result_svm_rbf <- out[0:40]
confusionMatrix(result_svm_rbf,cl)

# Reading table of 
accuracy <- read.csv("accuracy.csv", sep = ',', header = T)

ggplot(accuracy, aes(no_of_sample, y = accuracy, color = variable)) + 
  geom_line(aes(y = knn_k_7, col = "k-NN k=7")) +
  geom_line(aes(y = svm_linear, col = "SVM Linear")) +
  geom_line(aes(y = svm_rbf, col = "SVM RBF"))

