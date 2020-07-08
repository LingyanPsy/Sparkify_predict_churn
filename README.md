# Sparkify-User-Churn-Prediction
The purpose of this project is to predict user churn rate for Sparkify based on music streaming related behavior data (user operations recorded by Sparkify).  The results, discussions and insights are available in a [blog post]().

This is the capstone project for Udacity Data Scientist Nanodegree.

## Table of Contents
1. [Installation](#Installation)
2. [Project Motivation](#Project-Motivation)
3. [File Descriptions](#Files-Descriptions)
4. [Analysis Steps](#Analysis-Steps)
5. [Results and Discussions](#Results-and-Discussions)


### Installation
* Spark (Read about installation [here](https://changhsinlee.com/install-pyspark-windows-jupyter/) )
* Anaconda 3
* Python 3.7 (pyspark, pandas, seaborn, numpy)

### Project Motivation
Predicting churn rates is a challenging and common problem that data scientists and analysts regularly encounter in any customer-facing business. If user churn rate could be accurately predicted, promotions or discounts could be sent to users to keep them longer in service. 

Additionally, the ability to efficiently manipulate large datasets with Spark is one of the highest-demand skills in the field of data.

Sparkify is a fictional music streaming platform created by Udacity. For this project we are given log data of this platform in order to drive insights and create a machine learning pipeline to predict churn. This project utilizes the 128Mb data in Spark local mode which can be scaled up when using the full dataset (12GB).

### File Descriptions
* sparkify.ipynb - A notebook with feature engineering and modelling
* FeatureVisualizations - a folder containing visualisations of results of feature explorations

### Analysis Steps

Raw datasets contains the following sub-fields:
- userId: unique identifier for each user
- firstName: demographic information of each user
- lastName: demographic information of each user
- location: demographic information of each user
- gender: demographic information of each user
- userAgent: device that the user used
- sessionId:unique identifier for each session
- itemInSession: unique identifier for each item in a same session
- page: the specific page of website that the user visited, used to identify churn
- song: if the page is 'NextSong', this field will show the name of the song, otherwise only show 'null'
- artist: if the page is 'NextSong', this field will show the name of the artist, otherwise only show 'null'
- level: categorical features that only has 2 values, free or paid
- registration: the timestamp of user registration
- ts: the timestamp of user action
- status : status code There are three HTTP status codes 307: Temporary Redirect, 404: Not Found, 200: OK
- auth : authentication (cancelled/logged in/logged out)
- method : PUT/GET
- length : length of item

**Data Cleaning**

Summarize null rate in each sub-fields. Remove items with blank 'userId'.

**Exploratary Analysis and feature engineering**

First, define churn rate using 'Cancellation Confirmation' in 'Page'. There were 225 unique users out of which 53 churned (23.11% churn rate). Data is grouped/summarized by userId.

Second, I engineered different features that could possible influence churn rate, including:
- artist count
- song count 
- average session length
- largest gap between visits  
- session counts 
- cancellation page count  
- gender 
- location 
- level 
- total length of visit 
- total registration days 
- session frequency: session count/registration days 

Then I plotted these features' relationship with churn rate and performed preliminary statistical analysis (see Sparkify.ipynb and Figures folder). Some turns out to be not effective on user churn rate, such as average session length, location etc. Therefore, after eliminating some features and added some new ones, the following features are used for the final model:
- artist count
- song count 
- largest gap between visits  
- session counts 
- gender 
- level 
- total length of visit (total_length)
- total registration days (reg_days) 
- average session item count (avg_session_items)
- different page visiting frequency(Downgrade, Error, Home, Logout, NextSong etc.)

**Modelling**

As it is a classification problem(churn/not churn)-LogisticRegression,RandomForest and GradientBoost algorithms have been used. F1 score is used as metric as only 23% of users churned.

### Results and Discussions
Of the three models, Random Forest Classifier required the least computational power, could handle data imbalance and has a high F1 score. Hence,the hyperparameters
were tuned.
| Model |F1 score |
| --- | --- |
| Logistic Regression | 75.46%|
| Random Forest | 63.62% |
| Gradient Boost | 63.76% |
| Decision Tree | 64.77% |

Using gridsearch, the best parameters are 

The top 5 important features are (feature importance plot in Figures folder) :
1.	soung_count
2.	total_length
3.  submit_downgrade
4.	save_setting
5.	reg_days

**Hyperparameter tuning**

Since the logistic regression model performed best, I further improved the model with parameter tuning using Grid Search. Due to the limit of computer power, two parameters 1) regParam ([0.1,0.01]) and 2) fitIntercept ([True, False]) were searched. The result model performance is slightly improved.
| Model |F1 score |
| --- | --- |
| Logistic Regression (tuned) | 76.74%|
| Random Forest | 63.62% |
| Gradient Boost | 63.76% |
| Decision Tree | 64.77% |

**Discussion**

The current model is not perfect, partly because it's only a mini-set of the full dataset (12GB). Here are several ways to improve model in the future:
1. The current model is limited by the computing power, thus used a very limited set in GridSearch. This could be improved using cloud computing techniques such as AWS or IBM. 
2. Most of the features didn't contribute much to the model. These features could be replaced with other features we haven't thought of. 
3. The good prediction power comes from current model is inflated because it didn't differentiate data from 'before cancel' and 'after cancel'. Features like registration days are not totally justified, it should be replaced by 'registration days before cancelation'. Similarly, in such models, it's questionable to consider features such as 'time since last session', instead, we should use 'time since cancellation'.


### Acknowledgements
Thanks to [Udacity](www.udacity.com) for the data and project motivation.
