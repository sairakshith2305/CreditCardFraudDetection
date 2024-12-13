library(ranger)
library(caret)
library(data.table)
library(rpart)
library(caTools)
library(pROC)
library(rpart.plot)
library(gbm, quietly=True)
library(class)
library(e1071)
library(gplots, warn.conflicts = FALSE)
library("ROCR")
library("PRROC")
library(ROSE)
library(smotefamily)
library(dplyr)

#load the dataset
creditcard_data <- read.csv("D:/RAKSHITH/NOTES/NOTES_5SEM/FDS/Activity/archive/creditcard.csv")

#glance the dataset
str(creditcard_data)

#to get the total number of rows and columns in the dataset
dim(creditcard_data)

#to find the missing values
sum(is.na(creditcard_data))

#to display first 6 rows
head(creditcard_data,6)

#to display last 6 rows
tail(creditcard_data,6)

#list of columns
names(creditcard_data)

#convert class to a factor variable
creditcard_data$Class <- factor(creditcard_data$Class,levels=c(0,1))

# Total number of Fraud and Non-Fraud Rows in data
rowsTotal <- nrow(creditcard_data)
fraudRowsTotal <- nrow(creditcard_data[creditcard_data$Class == 1,])
nonFraudRowsTotal <- rowsTotal - fraudRowsTotal

fraudRowsTotal
nonFraudRowsTotal
rowsTotal

#number of fraud and legit transactions
table(creditcard_data$Class)

#percentage of fraud and legit transactions in the dataset
prop.table(table(creditcard_data$Class))

#pie chart of credit card transactions
labels<-c("Legit","Fraud")
labels<-paste(labels, round(100*prop.table(table(creditcard_data$Class)),2))
labels <- paste0(labels, "%")

pie(table(creditcard_data$Class), labels, col=c("grey","red"),
    main ="Pie Chart")

#--------------------------------------
#No model prediction
p <-rep.int(0, nrow(creditcard_data))
p <- factor(p, levels=c(1,0))
confusionMatrix(data= p ,reference = creditcard_data$Class)

#----------------------------------------
set.seed(1)
creditcard_data <- creditcard_data %>% sample_frac(0.1)
table(creditcard_data$Class)
ggplot(data = creditcard_data, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red')) 


#divide the dataset in training and testing datasets
set.seed(123)
data_sample = sample.split(creditcard_data$Class,SplitRatio=0.80)
train_data = subset(creditcard_data,data_sample==TRUE)
test_data = subset(creditcard_data,data_sample==FALSE)
dim(train_data)
dim(test_data)

#Random Over-Sampling
n_legit <- 22750
new_frac_legit <- 0.50
new_n_local <- n_legit/new_frac_legit 
oversampling_result <-ovun.sample(Class~., data = train_data,method = "over",
                                   N = new_n_local, seed = 2019)
oversampled_credit <- oversampling_result$data
table(oversampled_credit$Class)
ggplot(data = oversampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red')) 

#Random Under-Sampling
table(train_data$Class)
n_fraud <- 35
new_frac_fraud <- 0.50
new_n_total <- n_fraud/new_frac_fraud

undersampling_result <- ovun.sample(Class~., data = train_data,method="under",
                                    N = new_n_total, seed = 2019)

undersampled_credit <-undersampling_result$data
table(undersampled_credit$Class)
ggplot(data = undersampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red')) 

#Random Over-Sampling and Random Under - Sampling
new <- nrow(train_data)
fraction_fraud_new<- 0.50
sampling_result <- ovun.sample(Class~., data = train_data,method ="both",
                               N = new, p = fraction_fraud_new,seed = 2019)
sampled_credit <- sampling_result$data
table(sampled_credit$Class)

ggplot(data = sampled_credit, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red')) 

#----------------------------------------
#Using SMOTE to balance the dataset
n0 <- 22750
n1 <- 35
r0 <- 0.6

#calcuate the parameter for the dup_size value 
n <- ((1 - r0)/r0) * (n0 / n1) - 1

smote_output <- SMOTE(X = train_data[ , -c(1,31)],
                      target = train_data$Class,
                      K = 5, dup_size = n)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))

#Class distribution for original dataset
ggplot(data = train_data, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red'))

#class distribution for smote data
ggplot(data = credit_smote, aes(x=V1, y=V2, col=Class))+
  geom_point()+
  theme_bw()+
  scale_color_manual(values = c('blue', 'red'))

#----------------------------------------
#decision tree

decisionTree_model1 <- rpart(Class ~ . , credit_smote)
predicted_val1 <- predict(decisionTree_model1, test_data, type = 'class')
probability1 <- predict(decisionTree_model1, test_data, type = 'prob')
rpart.plot(decisionTree_model1)
confusionMatrix(predicted_val1,test_data$Class)


predicted_val<- predict(decisionTree_model1, creditcard_data[-1], type = 'class')
confusionMatrix(predicted_val,as.factor(creditcard_data$Class))

#without using smote technique
#decision tree

decisionTree_model2 <- rpart(Class ~ . , train_data[-1])
predicted_val2 <- predict(decisionTree_model2, test_data, type = 'class')
probability2 <- predict(decisionTree_model2, test_data, type = 'prob')
rpart.plot(decisionTree_model2)
confusionMatrix(predicted_val2,test_data$Class)

predicted_val3<- predict(decisionTree_model2, creditcard_data, type = 'class')
confusionMatrix(predicted_val3,as.factor(creditcard_data$Class))
