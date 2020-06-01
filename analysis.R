#loading required libraries: tidyverse, caret, and lubridate.
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")
# the code to turn the csv file into rda file is
#reading csv file into rda file
# the csv file is downloaded manually from https://www.kaggle.com/rtatman/did-it-rain-in-seattle-19482017
# renamed to rain and is provided with the files.
#rain <- read_csv("./rain.csv")
#saving rda file
#save(rain, file = "./rain.rda")

#The rda file is provided, you can load it directly by
load("./rain.rda")
#having a quick look into the dataset.
head(rain)
# turning rain dataset into a dataframe.
rain <- as.data.frame(rain)
#adding seperate columns for month and year parsed from DATE column.
#adding column raining of class factor to indicate if it rained or not
rain <- rain %>% mutate(month = month(DATE), year = year(DATE) ,
                        raini = ifelse(RAIN== TRUE, "YES", "NO"), 
                        Rained = as.factor(raini)) %>%
  select(-raini) %>% select(-PRCP)
#examining prevelance of classes to predict
mean(rain$RAIN) 
# we got NA so we examine presence of nas in the data set
sum(is.na(rain$RAIN))
#examining three values that has NA 
rain$DATE[is.na(rain$RAIN)]
#removing NA since they are very few and unlikely to affect results
rain<- na.omit(rain)
#examining prevelance again
mean(rain$RAIN)

#partitioning the data set into train set and test set (80/20)
#setting the seed
suppressWarnings(set.seed(1998, sample.kind = "Rounding"))
#splitting index
test_index <- createDataPartition(rain$Rained, times = 1, p = 0.2, list = FALSE)
#generating test set
test_set <- rain[test_index, ]
#generating train set
train_set <- rain[-test_index, ]
#train set dimensions
dim(train_set)
#test set dimensions
dim(test_set)


#exploratory data visualization

#visualizing the correlaton between the month and average raining days
train_set %>% group_by(month) %>%
  summarize(sum = sum(RAIN)) %>%
  ggplot(aes(month, sum))+ geom_line() + xlab("Month of the year")+
  ylab("Number of rainy days") 


#examining the relationship between TMIN and number of rainy days with line graph
train_set %>% group_by(TMIN) %>%
  summarize(sum = sum(RAIN)) %>%
  ggplot(aes(TMIN, sum))+ geom_line()+ylab("Number of rainy days") 

#examining the relationship between TMAX and number of rainy days with line graph
train_set %>% group_by(TMAX) %>%
  summarize(sum = sum(RAIN)) %>%
  ggplot(aes(TMAX, sum))+ geom_line()+ylab("Number of rainy days") 

#examining the combined effect of TMAX and TMIN in classification of rainy days
train_set %>% 
  ggplot(aes(TMAX, TMIN, color = Rained))+ geom_point()

#knn nearest nighberhood

#training model on train set with TMAX, TMIN, and month as predictors
#tuning for different k values
fit_knn <-   train(Rained ~ TMAX+TMIN+month, method = "knn",
                    data = train_set,
                    tuneGrid = data.frame(k = seq(101,171,2)))
#plotting k againss accuracy values
ggplot(fit_knn, highlight = TRUE)
#obtaining k that produced the best model
fit_knn$bestTune$k
#obtaining accuracy of the best model
max(fit_knn$results$Accuracy)
#testing the trained model on the test set
#predicting classes
y_hat_knn <- predict(fit_knn, test_set, type="raw")
#comparing the predictions with the real values
cmknn <- confusionMatrix(y_hat_knn, factor(test_set$Rained))
#obtaining accuracy of applying our model on the test set
cmknn$overall["Accuracy"]


#regression tree model

#training model on train set with TMAX, TMIN, and month as predictors
#tuning for different cp values
fit_rpart <-   train(Rained ~ TMAX+TMIN+month, 
                     method = "rpart",
                     tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                     data = train_set)
#plotting accuray against cp values
plot(fit_rpart)
#best cp value
fit_rpart$bestTune$cp
#accuracy of best model
max(fit_rpart$results$Accuracy)
#The importance of predictors in the model
fit_rpart$finalModel$variable.importance
#plotting the final tree
plot(fit_rpart$finalModel, margin = 0.02)
text(fit_rpart$finalModel, cex = 0.5)
#testing the trained model on the test set
#predicting classes
y_hat_rpart <- predict(fit_rpart, test_set, type="raw")
#comparing the predictions with the real values
cmrp <- confusionMatrix(y_hat_rpart, factor(test_set$Rained))
#obtaining accuracy of applying our model on the test set
cmrp$overall["Accuracy"]

#random forest model
#setting the seed
suppressWarnings(set.seed(1999, sample.kind = "Rounding"))
#training a model 
fit_rf <- train(Rained ~ TMAX+TMIN+month,
                method = "rf",
                data = train_set)
#accuracy of the model
max(fit_rf$results$Accuracy)


#predicting classes of the test set
y_hat_rf <- predict(fit_rf, test_set, type="raw")
#comparing predictions to actual values
cmrf <- confusionMatrix(y_hat_rf, factor(test_set$Rained))
#obtaining accuracy of applying our model on the test set
cmrf$overall["Accuracy"]

