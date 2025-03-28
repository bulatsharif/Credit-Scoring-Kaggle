## Introduction

This repository contains a solution to the Kaggle Competition - [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit).  This tiny pet project was done for educational purposes, to use practical knowledge obtained from Machine Learning Courses. The project consists of a jupyter notebook, and also a very small FastAPI application, to deploy the model to make predictions. Below I will briefly explain the main parts of the project. 

- Task of the dataset: Predict the probability that an individual will be more than 90 days delinquent on a loan within the next two years.

> Final model AUC score: **0.8739**

**Deployed model is available at: https://credit-scoring-kaggle.onrender.com/**

## Exploratory Data Analysis

We are given the data of 150.000 (for training) customers, where we have several columns:

- **RevolvingUtilizationOfUnsecuredLines**: A portion of available credit that you are using
- **age**: Your age
- **NumberOfTime30_59DaysPastDueNotWorse**: How many times you were 30-59 days past, but not worse?
- **DebtRatio**: Monthly debt payment, divided by monthly gross income
- **MonthlyIncome**: Monthly gross income
- **NumberOfOpenCreditLinesAndLoans**: Number of open credit lines and loans
- **NumberOfTimes90DaysLate**: Number of times you were 90 days late
- **NumberRealEstateLoansOrLines**: Number of mortgage or real estate loans
- **NumberOfTime60_89DaysPastDueNotWorse**: How many times you were 60-89 days past, but not worse?
- **NumberOfDependents**: How many dependents do you have?

EDA was performed in the following steps:
1. Firstly I looked at the the correlation of features in the dataset. There were highly correlated features, but because they were also highly correlated with the target variable, I decided to keep them.
2. I tried to pick specific groups and check whether they have an increase in correlation with the target variable. For example, people who had DebtRatio bigger than average value indeed had an increase in correlation between their features and target variable. 
3. I continued to work in this fashion and also checked several other specific groups: Those who have less monthly income, those who are in the 'dangerous' age, and those who have a lot of loans opened
4. Afterwards, I created kdeplots and relplots, to see the distribution and relations between several features. For example, younger people and people with a high portion of available credit tend to have more risk.

## Preprocessing

Based on the exploratory data analysis performed above, I created several new features, here I will describe some of them
- Number of days late: The sum of variables, which name like 'NumberOfTime%'
- Is the individual young? Does he have a low income? Does he have a big portion of available credit used?
- Performed logarithmic transformations to make data less skewed on several columns
- Squared other features, like number of dependents, etc.

## Model Selection

- I used three models for this classification task: Logistic Regression,  CatBoost, and XGBoost. Moreover, since the target variable was highly imbalanced, I tried several re-sampling techniques: RandomUnderSampler, ADASYN, SMOTE;
- Re-sampling techniques did not help to improve any results, and they were not efficient
- While logistic regression yielded promising results: 0.85 on AUC Score, gradient boosting models performed better
- The difference between XGBoost and CatBoost at first glance was negligibly small, so I decided to perform an offline A/B test
## A/B Testing Models
1. I split my dataset into training and test sets again
2. Trained both models with the same set of parameters
3. Performed predictions, and checked the correctness of predictions
4. Performed paired t-test, to conclude, whether there was a statistical difference

Conclusion: there was statistical significance with p-value = 0.00, CatBoost model was in the end better.

## Final Model and Deployment
Finally, I trained again CatBoost and made the final predictions. Resulting in an AUC Score of 0.8739.

I used FastAPI to create a simple site, on which I can enter data to predict the probability of having the risk of delinquency.

For deployment of the model, I used free available render.com:

- Build command:
```
pip install fastapi[standard] numpy pandas matplotlib catboost scikit-learn uvicorn[standard]
```

- Start command:
```
uvicorn website.main:app --reload --host 0.0.0.0
```

*Author: Bulat Sharipov, Innopolis University*



