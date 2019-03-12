#Problem focuses on predicting probablity of customer to close his/her loan before time.
#Data Set avaialble Customer, LMS,train,test,email data.
#Source:https://www.hackerearth.com/challenges/competitive/machine-learning-Edelweiss/problems/

options(scipen = 999)
library(randomForest)
library(data.table)
library(tidyr)
library(dplyr)
#EDA part
library(DataExplorer)
#generating report for data profiling and distribution plot for each variable
require(lubridate)
#loading train and test data frame
lms=fread("C:/Users/C25060/Documents/EDW/lms31Jan.csv",na.strings = c("","NA","null"))
cust=fread("C:/Users/C25060/Documents/EDW/CSV Customers_31JAN2019.csv",na.strings = c("","NA","null"))
train=fread("C:/Users/C25060/Documents/EDW/train_foreclosure.csv")
test=fread("C:/Users/C25060/Documents/EDW/test_foreclosure.csv")
#mean(train$FORECLOSURE)=0.08969618 says 9% :91% is mixture of 1 and 0 respectively
#,suggesting a skewed data

#observations
#1. Data from customer table has 10000 unique ID.
#2. Data from LMS table has 33000 unique cust ID, so technically there is 23000 customer info
# is not available.
#3. Used K-Means method to cluster Customer data to find how customer data is scattered across it's Key variables.
#
sink("EDA on cust and LMS data.txt",append = T)
describe(lms)
describe(cust)
glimpse(lms)
glimpse(cust)
sink()



library(Hmisc)
#install.packages("missForest")
library(missForest)
library(missRanger)
#Based on EDA deleting the col's which has a corelated variable present and It has lot of Missing values

cust$QUALIFICATION=NULL
cust$PRE_JOBYEARS=NULL
cust$BRANCH_PINCODE=NULL
cust$NO_OF_DEPENDENT=NULL
cust$POSITION=NULL

cust=missRanger(cust, pmm.k =4, seed = NULL,num.trees=100,verbose = 1, returnOOB = FALSE)



cust$SEX=ifelse(cust$SEX%in% "M",1,0) # converting gender to numeric.
cust$MARITAL_STATUS=ifelse(cust$MARITAL_STATUS%in% "M",1,0)# converting martial status to numeric
cust$MARITAL_STATUS=as.numeric(cust$MARITAL_STATUS)


class(cust$SEX)

cust=cust[complete.cases(cust),]
#capturing data without Cust id and binary Numeric vars before scaling the data
inputdata_final=cust[,-c(1,6,5)]

a=cust[,c(6,5)]
inputdata_final=scale(inputdata_final)
inputdata_final=cbind(inputdata_final,a)

library(cluster) # Needed for silhouette function


# Setup for k-means loop to identify best suitable K value between 3-12
km.out <- list()
sil.out <- list()
x <- vector()
y <- vector()

minClust <- 3      # Hypothesized minimum number of segments
maxClust <- 12     # Hypothesized maximum number of segments

# Compute k-means clustering over various clusters, k, from minClust to maxClust
for (centr in minClust:maxClust) {
  i <- centr-(minClust-1) # relevels start as 1, and increases with centr
  set.seed(11)            # For reproducibility
  km.out[i] <- list(kmeans(inputdata_final, centers = centr))
  sil.out[i] <- list(silhouette(km.out[[i]][[1]], dist(inputdata_final)))
  # Used for plotting silhouette average widths
  x[i] = centr  # value of k
  y[i] = summary(sil.out[[i]])[[4]]  # Silhouette average width
}
library(ggplot2)
ggplot(data = data.frame(x, y), aes(x, y)) + 
  geom_point(size=3) + 
  geom_line() +
  xlab("Number of Cluster Centers") +
  ylab("Silhouette Average Width") +
  ggtitle("Silhouette Average Width as Cluster Center Varies")

cluster_ten <- kmeans(inputdata_final,10)

cust1_new<-cbind(cust,km_clust_10=cluster_ten$cluster)

#Graph based on k-means - how clsuter are separated


#Converting into factors
cust1_new$km_clust_10=factor(cust1_new$km_clust_10)

#install.packages("tables")
require(tables)
names(cust1_new)
profile<-tabular(1+CUST_CONSTTYPE_ID+CUST_CATEGORYID +AGE+SEX+MARITAL_STATUS+GROSS_INCOME
                 +NETTAKEHOMEINCOME~
                   mean+(mean*km_clust_10),
                 data=cust1_new)

View(profile)

profile1<-as.matrix(profile)
profile1<-data.frame(profile1)
View(profile1)

profile<-tabular(1~length+(length*km_clust_10),
                 data=cust1_new)
profile2<-as.matrix(profile)
profile2<-data.frame(profile2)
View(profile2)

write.csv(profile1,"profile1.csv",row.names = F)
write.csv(profile2,"profile2.csv",row.names = F)

cust_new=cust1_new[,c(1,9)]
names(cust1_new)
lms1=merge(lms,cust_new,by= "CUSTOMERID",all.x = T)
#lms$km_clust_10.x=NULL
saveRDS(lms1,"lms.RDS")

describe(lms)
#deleting insignificant variables from lms data
lms1$INTEREST_START_DATE=NULL
lms1$AUTHORIZATIONDATE=NULL
lms1$CITY=NULL
lms1$MONTHOPENING=NULL
lms1$LAST_RECEIPT_DATE=NULL
lms1$LAST_RECEIPT_AMOUNT=NULL
lms1$NPA_IN_CURRENT_MONTH=NULL
lms1$NPA_IN_LAST_MONTH=NULL
 
#replacing missing value with mean
hist(lms1$BALANCE_TENURE)
lms1$BALANCE_TENURE=ifelse(is.na(lms1$BALANCE_TENURE),169,lms1$BALANCE_TENURE)


#since Joing only 10000 cust into 33k unique cust set so there is lots of NA in KM_clust 10 col
#KM _clust 10 is carrying Information from cust table, it's missing values i am imputing with Missranger.

#lms1$CUSTOMERID=NULL
lms2=missRanger(lms1, pmm.k =4, seed = NULL,num.trees=100,verbose = 1, returnOOB = FALSE)


library(ROSE)
library(caret)
library(dplyr)          
library(pROC)  
library(ROCR)
library(h2o)
library(ROSE)
library(caret)
library(dplyr)          
library(kernlab)       # support vector machine 
library(pROC)  
library(e1071)        # support vector machine 
library(ROCR)
library(dplyr)

set.seed(555)
names(lms2)

#Since train has 20k rows of unique Agreement ID,so ranked Obs based on outstanding pricipal 
#


lms2=lms2%>%group_by(AGREEMENTID)%>%mutate(ranks=order(desc(OUTSTANDING_PRINCIPAL),AGREEMENTID))
min_os=lms2[lms2$ranks==1,]

train1=merge(min_os,train,by="AGREEMENTID",all.y = T)
test1=merge(min_os,test,by="AGREEMENTID",all.y = T)



#names(trainset)
#Initializing H2o
library(h2o)
h2o.init(
  nthreads=-1,            ## -1: use all available threads
  max_mem_size = "2G")    ## specify the memory size for the H2O cloud
h2o.removeAll()           ## Clean slate - just in case the cluster was already running

train1$ranks=NULL
test1$ranks=NULL
train1$FORECLOSURE=factor(train1$FORECLOSURE,levels = c(0, 1))
test1$FORECLOSURE=NA
#Assignment within H2o
#removing Cust ID,Loan ID,
train1[,c(1,2,28,29)]=NULL
test1[,c(1,2,28,29)]=NULL

tr1=as.h2o(train1)
te=as.h2o(test1)
test=as.h2o(test1)


val=h2o.assign(tr1[1:6000,], "tr1.hex")
train <- h2o.assign(tr1[6001:20012,], "tr1.hex")   #Train data: H2O name train.hex
test <- h2o.assign(test, "test.hex")     #Test data: H2O name test.hex


###################### MODEL-1: RANDOM FOREST ######################
## run our first predictive model with RF

gbm1 <- h2o.gbm(
  training_frame = train,         
  validation_frame = val,       
  x=1:27,                         
  nfolds =3,                    # 5fold CV
  y=28,                           
  model_id = "gbm_covType1",     # name the model in H2O
  seed = 55005)      


plot(gbm1, timestep = "AUTO", metric = "AUTO")

#Performance Evaluation
summary(gbm1)    
gbm1@model$validation_metrics@metrics$AUC    #AUC

final_predictions_gbm1<-h2o.predict(
  object = gbm1,
  newdata = test[,-28],class="prob")

pred=as.data.frame(final_predictions_gbm1)

write.csv(pred,"GbM.csv")
mean(final_predictions_gbm1$predict==test$Status_Final)  ## test set accuracy

rf1 <- h2o.randomForest(         
  training_frame = train,        
  validation_frame = train,      
  x=1:27,                        ## the predictor columns, by column index
  y=28,                           ## the target index (what we are predicting)
  nfolds =5,                     ## 5 fold CV
  model_id = "rf_covType_v1",    ## name the model in H2O
  ##   not required, but helps use Flow
  binomial_double_trees = FALSE, seed = 1234,
  ntrees = 250,                  ## use 50-250 trees, stopping criteria will decide finally...
  score_each_iteration = T)      ## Predict against training and validation for each tree)  

#Performance Evaluation
 
sink('rf.txt',append = F)
summary(rf1)
rf1@model$validation_metrics@metrics$AUC    #AUC
sink()
final_predictions_rf1<-h2o.predict(
  object = rf1,
  newdata = test,'probs')
mean(final_predictions_rf1$predict==test$Status_Final)  ## test set accuracy

pred=as.data.table(final_predictions_rf1)
write.csv(pred,"rf.csv")

gbm3 <- h2o.gbm(
  training_frame = train,     
  validation_frame = train,   
  x=1:26,                    
  y=27,                       
  nfolds =5,
  ntrees = 50,               # add a few trees (from 20, though default is 50)
  learn_rate = 0.15,         # increase the learning rate even further
  max_depth = 3,               
  sample_rate = 0.5,          # use a random 50% of the rows to fit each tree
  col_sample_rate = 0.85,     # use 85% of the columns to fit each tree
  score_each_iteration = T,   
  model_id = "gbm_covType3",
  seed = 55005)           


#Performance Evaluation
summary(gbm3)    
gbm3@model$validation_metrics@metrics$AUC    #AUC

final_predictions_gbm3<-h2o.predict(
  object = gbm3,
  newdata = test,"probs")

pred=as.data.frame(final_predictions_gbm3)
write.csv(pred,'gbm3.csv')

##################
#autoML

aml <- h2o.automl(x = 1:26, y = 27,
                  training_frame = tr1,
                  max_models = 20,
                  seed = 1)


# GBM Hyperparamters
learn_rate_opt <- c(0.01, 0.03)
max_depth_opt <- c(3, 4, 5, 6, 9)
sample_rate_opt <- c(0.7, 0.8, 0.9, 1.0)
col_sample_rate_opt <- c(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
hyper_params <- list(learn_rate = learn_rate_opt,
                     max_depth = max_depth_opt,
                     sample_rate = sample_rate_opt,
                     col_sample_rate = col_sample_rate_opt)
nfolds <- 5
search_criteria <- list(strategy = "RandomDiscrete",
                        max_models = 3,
                        seed = 1)

gbm_grid <- h2o.grid(algorithm = "gbm",
                     x = 1:26,
                     y = 27,
                     training_frame = tr1,
                     
                     ntrees = 10,
                     seed = 1,
                     nfolds = nfolds,
                     fold_assignment = "Modulo",
                     keep_cross_validation_predictions = TRUE,
                     hyper_params = hyper_params,
                     search_criteria = search_criteria)

# Train a stacked ensemble using the GBM grid <- 24.34379
ensemble <- h2o.stackedEnsemble(x = 1:26,
                                y = 27,
                                training_frame = tr1,
                                model_id = "ensemble_gbm_grid_regression",
                                base_models = gbm_grid@model_ids)


sink("automl.txt",append = T)
summary(gbm_grid)    

final_predictions_gbm3<-h2o.predict(
  object = ensemble,
  
  newdata = test,"probs")
#predict.GBM2 <- as.data.frame(h2o.predict(my_gbm, te_fin_h20[,3:30]))
pred=as.data.frame(final_predictions_gbm3)
write.csv(pred,'ensemble.csv')


prostate.nb <- h2o.naiveBayes(x = 1:26, y = 27, training_frame = tr1, laplace = 1)
print(prostate.nb)

nb.pred <- predict(prostate.nb, test)

write.csv(as.data.frame(nb.pred),"nb.csv")

getwd()


dt1=h2o.randomForest(x = 1:26, y = 27, training_frame = tr1,
                 validation_frame =test ,
                 binomial_double_trees = T, seed = 1234)



dt_1=predict(dt1,test,"probs")

write.csv(as.data.frame(dt_1),"dt1.csv")


regression.model <- h2o.glm( y = 27, x =1:26, training_frame = tr1, family = "binomial")
summary(regression.model)
reg=predict(regression.model,test[,-27],"probs")
as.data.frame(h2o.varimp(regression.model))
write.csv(as.data.frame(reg),"logit.csv")

names(test1)
tr2=tr1[,c(26,6,3,4,7,20,22,23,11,12,2,1,5,13,21,24,27)]
test2=test[,c(26,6,3,4,7,20,22,23,11,12,2,1,5,13,21,24)]
colnames(tr2)

regression.model2 <- h2o.glm( y = 17, x =1:16, training_frame = tr2, family = "binomial")
summary(regression.model2)
reg=predict(regression.model2,test2,"probs")
write.csv(as.data.frame(reg),"logit2.csv")


h2o.shutdown()
y
head(lms2$AGREEMENTID,30)
a=lms2[lms2$AGREEMENTID==11252743,]
