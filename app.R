# setwd("/home/nguye685/R/finalproject")
# 
# # set memory limits
# options(java.parameters = "-Xmx64048m") # 64048 is 64 GB
# 
# # Connect to a MariaDB version of a MySQL database
# library(RMariaDB)
# con <- dbConnect(RMariaDB::MariaDB(), host="datamine.rcac.purdue.edu", port=3306
#                  , dbname="kag_popular_news", user="kaggle_news", password="kaggle_pass")
# 
# # tables in db
# dbListTables(con)
# 
# # query
# d <- dbGetQuery(con, "select * from train")
# holdout <- dbGetQuery(con, "select * from test")
# ss <- dbGetQuery(con, "select * from sample_submission")
# 
# # disconnect from db
# dbDisconnect(con)
# 
# write.csv(x=d, file="traindata.csv")
# write.csv(x=holdout, file="holdoutdata.csv")
# write.csv(x=ss, file="submissionsample.csv")
############################ Load Libraries ####################################
library(dplyr)
library(shiny)
library(caret)
library(h2o)
library(shinydashboard)
library(shinythemes)
library(rsconnect)

d <- read.csv("traindata.csv")
holdout <- read.csv("holdoutdata.csv")
################################################################################
############################### Data Cleaning ##################################
################################################################################


########################### Set Correct Data Type ##############################
# d <- d %>%
#   mutate(is_weekend = as.factor(is_weekend)) %>%
#   mutate_each(funs= as.factor, starts_with("data")) %>%
#   mutate_each(funs =as.factor, starts_with("weekday"))
# 
# holdout <- holdout %>%
#   mutate(is_weekend = as.factor(is_weekend)) %>%
#   mutate_each(funs= as.factor, starts_with("data")) %>%
#   mutate_each(funs =as.factor, starts_with("weekday"))
# str(holdout)

# No missing data
sapply(d, function(x) sum(is.na(x)))
sapply(holdout, function(x) sum(is.na(x)))

##################### Prepare Data Set #########################################

#Make the target variable first column in the dataset
d <- d[,c(61,3:60)] 
str(d)
holdout <- holdout[,2:60]

# Make the target (shares) column name "y"
names(d)[1] <- "y"
str(d)

##################### Predictive Model Using Caret #############################
################################################################################
set.seed(123)

############################# One-hot Encoding #################################
# names(d)[1] <- "y"
# 
# 
# dummies <- dummyVars(y ~ ., data = d)
# ex <- data.frame(predict(dummies, newdata = tr))
# names(ex) <-gsub("\\.", "", names(ex))
# tr <- cbind(tr$y, ex)
# names(tr)[1] <- "y"
# 
# dummies <- dummyVars(~ ., data = te)
# ex <- data.frame(predict(dummies, newdata = te))
# names(ex) <-gsub("\\.", "", names(ex))
# te<- ex
# 
# rm(dummies, ex)

##################### Identify High Correlation ################################
Cor <- cor(d[, 2:ncol(d)])
highCor <- sum(abs(Cor[upper.tri(Cor)]) > 0.9)
summary(Cor[upper.tri(Cor)])

highlyCorDescr <- findCorrelation(Cor, cutoff = 0.9)
filteredDescr <- d[, 2:ncol(d)][, -highlyCorDescr]
Cor2 <- cor(filteredDescr)

summary(Cor2[upper.tri(Cor2)])

d<- cbind(d$y, filteredDescr)
names(d)[1] <- "y"

rm(filteredDescr, Cor, Cor2, highCor, highlyCorDescr)

########################## Remove Linear Combo #################################
y = d$y

d <- cbind(rep(1, nrow(d)), d[2:ncol(d)])

names(d)[1] <- "ones"

comboInfo <-findLinearCombos(d)
comboInfo

d <- d[, -comboInfo$remove]
d <- d[,c(2:ncol(d))]

d <- cbind(y, d)


###################### Remove Limited Varation #################################
nzv <- nearZeroVar(d, saveMetrics = TRUE)
head(nzv)
d <- d[, c(TRUE,!nzv$nzv[2:ncol(d)])]


####################### Standardize Using Min-max ##############################

preProcValues <- preProcess(d[,2:ncol(d)], method = c("YeoJohnson"))
preProcValues_te <- preProcess(holdout[,2:ncol(holdout)], method = c("YeoJohnson"))

d <- predict(preProcValues, d)
holdout <- predict(preProcValues_te, holdout)

d <- d %>%
 select(y, global_sentiment_polarity, num_keywords, avg_negative_polarity,
          LDA_01, num_imgs, kw_max_max, kw_min_avg,
         n_tokens_content, num_hrefs, kw_avg_max)

############################### H2o ################################
h2o.init()

h2o.clusterInfo()

data <- as.h2o(d)

y <- "y"                       # target variable to learn

x <- setdiff(names(data), y)

parts <- h2o.splitFrame(data, 0.8, seed=99) # randomly partition data into 80/20

train <- parts[[1]]                         # random set of training obs

test <- parts[[2]]

rf <- h2o.randomForest(x,y,train)

h2o.performance(rf, test)

varimp_nb <- h2o.varimp_plot(rf, num_of_features =NULL)

# h2o.shutdown()


############################### Build Shiny app ################################

ui <- fluidPage(
  theme = shinytheme("united"),
  titlePanel("News Popularity Prediction"),
  
  br(), 
  
  tabsetPanel(type = "tabs",
              tabPanel("Data Exploratory", 
                           sidebarLayout (
                             
                             sidebarPanel(
                               selectInput('var', 'Choose Variable', names(d),
                                           selected = names(d)[[3]])
                             ),
                             
                             mainPanel(h2("Relationship of Each variable And Shares"),
                                       plotOutput("explore_plot"),
                                       br(),
                                       helpText("Summary Statistics"),
                                       verbatimTextOutput("explore_stat"))
                           )),
                  
              tabPanel("Prediction", 
                       h2("Please Choose Different Values To Predict"),
                       br(),
                           fluidRow(
                             
                             column(3,
                                    sliderInput(inputId='GLOBAL_SENTIMENT_POLARITY', label='GLOBAL_SENTIMENT_POLARITY', 
                                                min = min(holdout$global_sentiment_polarity), 
                                                value=0, max=max(holdout$global_sentiment_polarity)),
              
                                    
                                    br(),
                                    
                                    sliderInput(inputId='NUM_KEYWORDS', label='NUM_KEYWORDS', 
                                                min = min(holdout$num_keywords), 
                                                value=5,max=max(holdout$num_keywords)),
                                    
                                    sliderInput(inputId='AVG_NEGATIVE_POLARITY', label='AVG_NEGATIVE_POLARITY', 
                                                            min = min(holdout$avg_negative_polarity), 
                                                            value=0, max=max(holdout$avg_negative_polarity))
                             ),
                             
                             column(4, offset = 1,
                                    
                                    sliderInput(inputId='LDA_01', label='LDA_01', 
                                                min = min(holdout$LDA_01), 
                                                value=0, max=max(holdout$LDA_01)),
          
                                    sliderInput(inputId='KW_MAX_MAX', label='KW_MAX_MAX', 
                                                            min = min(holdout$kw_max_max), 
                                                            value=0, max=max(holdout$kw_max_max)),
                                               
                                    sliderInput(inputId='KW_MIN_AVG', label='KW_MIN_AVG', 
                                                            min = min(holdout$kw_min_avg), 
                                                            value=0, max=max(holdout$kw_min_avg)),
                                                
                                    sliderInput(inputId='KW_AVG_MAX', label='KW_AVG_MAX', 
                                                            min = min(holdout$kw_avg_max), 
                                                            value=0, max=max(holdout$kw_avg_max))
                             ),
                             
                             column(4,
                                    
                                    sliderInput(inputId='NUM_IMGS', label='NUM_IMGS', 
                                                min = min(holdout$num_imgs), 
                                                value=0, max=max(holdout$num_imgs)),
                                                
                                    sliderInput(inputId='N_TOKENS_CONTENT', label='N_TOKENS_CONTENT', 
                                                            min = min(holdout$n_tokens_content), 
                                                            value=0, max=max(holdout$n_tokens_content)),  
                                          
                                    sliderInput(inputId='NUM_HREFS', label='NUM_REFS', 
                                                            min = min(holdout$num_hrefs), 
                                                            value=0, max=max(holdout$num_hrefs))
                             ),
                             
                             actionButton("plot", "Predict!"), 
                             
                             mainPanel("Predicted Shares", dataTableOutput("Pred"))
                           )
                  ),
              
              tabPanel("Model Statistics",
                         mainPanel(h2("Summary Statistics of The Predictive Model"),
                                   plotOutput('impVarplot'),
                                   
                                   br(),
                                   
                                   verbatimTextOutput("stat1")
                       ))
              
        )
)

server <- function(input,output) {
  output$explore_plot <- renderPlot ({
    plot(y=d$y, x=d[,input$var], 
         xlab = as.character(input$var),
         ylab = "Numbers of Shares")
  })
  
  output$explore_stat <- renderPrint ({
    summary(d[,input$var])
  })
  
  ############################## Prediction Tab ################################
  data2 <- eventReactive(input$plot,{
    
    as.h2o(data.frame(global_sentiment_polarity = input$GLOBAL_SENTIMENT_POLARITY,
                      num_keywords = input$NUM_KEYWORDS,
                      avg_negative_polarity = input$AVG_NEGATIVE_POLARITY,
                      LDA_01 = input$LDA_01,
                      kw_max_max = input$KW_MAX_MAX,
                      kw_min_avg = input$KW_MIN_AVG,
                      kw_avg_max = input$KW_AVG_MAX,
                      num_imgs = input$NUM_IMGS,
                      n_tokens_content = input$N_TOKENS_CONTENT,
                      num_hrefs = input$NUM_HREFS))
      
    })
  
  pred <- eventReactive(input$plot,{

    h2o.predict(rf,newdata = data2())
    
  })
  
  output$Pred <- renderDataTable(pred())
  
  output$impVarplot <- renderPlot({
    varimp_nb <- h2o.varimp_plot(rf, num_of_features =NULL)
  })
  
  output$stat1 <- renderPrint ({ 
    h2o.performance(rf, test)
  })
  
}

shinyApp(ui, server)
