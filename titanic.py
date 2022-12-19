import pandas as pd
import numpy as np

# Start Python Imports
import math, time, random, datetime, statistics

import matplotlib.pyplot as plt
import missingno
import seaborn as sb
plt.style.use('seaborn-whitegrid')

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
import catboost
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier, Pool, cv

# Let's be rebels and ignore warnings for now
import warnings
warnings.filterwarnings('ignore')

# Import data sets

train = pd.read_csv('~/Desktop/data/train.csv')
test = pd.read_csv('~/Desktop/data/test.csv')
gender_submit = pd.read_csv('~/Desktop/data/gender_submission.csv')

# Initial playing with data

train.head()

train.info()
train.describe()

# EDA

train['Age'].plot.hist(color = 'blue')
plt.show()

sb.heatmap(train.corr())
plt.show()

# Plotting missing values

missingno.matrix(train, figsize=(15,5))
plt.show()

print('We can see that there are missing values in Age and Cabin values as well as a small bit in Embarking')

train.isna().sum()

# Creating 2 new dataframes, one for discrete continuous variables and the other one for continuous variables

df_dis  = pd.DataFrame()
df_con = pd.DataFrame()

train.dtypes

# Check each variable to see which one to add or not

print(train.columns)

# How many people survived?
fig = plt.figure(figsize=(15,1))
sb.countplot(y='Survived', data=train)
plt.show()
print(train.Survived.value_counts())

df_dis['Survived'] = train['Survived']
df_con['Survived'] = train['Survived']

# Checking Pclass (the ticket class of the passenger)

sb.distplot(train['Pclass'])
plt.show()

df_dis['Pclass'] = train['Pclass']
df_con['Pclass'] = train['Pclass']

# Check names (this will typically have stuff like Mr, Ms, etc.)

titles = []
for i in train['Name']:
    titles.append(i.split(', ')[1].split('.')[0])

print(set(titles))
len(titles)
train['title'] = titles
sb.displot(titles)
plt.show()

df_dis['Title'] = titles
df_con['Title'] = titles

# Add sex into the dataframe

df_dis['Sex'] = train['Sex']
df_dis['Sex'] = np.where(df_dis['Sex'] == 'female',1,0)

df_con['Sex'] = train['Sex']

# How does the Sex variable look compared to Survival?
# We can see this because they're both binarys.

fig = plt.figure(figsize=(10, 10))
sb.distplot(df_dis.loc[df_dis['Survived'] == 1]['Sex'], kde_kws={'label': 'Survived'})
sb.distplot(df_dis.loc[df_dis['Survived'] == 0]['Sex'], kde_kws={'label': 'Did not survive'})
plt.show()

# Check Age

sb.distplot(train['Age'])
plt.show()

train['Age'].isna().sum()


(train.loc[train['Sex'] =='female']['Age']).describe()
statistics.mode(train.loc[train['Sex'] =='female']['Age'])
sb.distplot(train.loc[train['Sex'] =='female']['Age'])
plt.show()
#  from the graph, it looks closer to normal distributed, so we will fill nan values with the mean age for women


(train.loc[train['Sex'] =='male']['Age']).describe()
statistics.mode(train.loc[train['Sex'] =='male']['Age'])

sb.distplot(train.loc[train['Sex'] =='male']['Age'])
plt.show()
# from the graph, it looks skewed, so we will use the median value to fill nan for men

math.isnan(train[['Age','Sex']].iloc[888][0]) == True and (train[['Age','Sex']].iloc[888][1]) == 'female'

age = []

for i in range(len(train['Age'])):
    if math.isnan(train[['Age','Sex']].iloc[i][0]) == True and (train[['Age','Sex']].iloc[i][1]) == 'female':
        age.append(28)
    elif math.isnan(train[['Age','Sex']].iloc[i][0]) == True and (train[['Age','Sex']].iloc[i][1]) == 'male':
        age.append(29)
    else:
        age.append(train['Age'].iloc[i])

len(age)

df_dis['Age'] = age
df_dis['Age'] = pd.cut(df_dis['Age'],10) #bucket/bin age into different groups

df_con['Age'] = age

# Function to create count and distribution visualizations

def plot_count_dist(data, bin_df, label_column, target_column, figsize=(20, 5), use_bin_df=False):
    """
    Function to plot counts and distributions of a label variable and 
    target variable side by side.
    ::param_data:: = target dataframe
    ::param_bin_df:: = binned dataframe for countplot
    ::param_label_column:: = binary labelled column
    ::param_target_column:: = column you want to view counts and distributions
    ::param_figsize:: = size of figure (width, height)
    ::param_use_bin_df:: = whether or not to use the bin_df, default False
    """
    if use_bin_df: 
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sb.countplot(y=target_column, data=bin_df)
        plt.subplot(1, 2, 2)
        sb.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"})
        sb.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"})
        plt.show()
    else:
        fig = plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        sb.countplot(y=target_column, data=data)
        plt.subplot(1, 2, 2)
        sb.distplot(data.loc[data[label_column] == 1][target_column], 
                     kde_kws={"label": "Survived"})
        sb.distplot(data.loc[data[label_column] == 0][target_column], 
                     kde_kws={"label": "Did not survive"})
        plt.show()

# Check SibSp (Number of siblings/spouses onboard with the person)

train['SibSp'].isnull().sum()

train['SibSp'].value_counts()

df_dis['SibSp'] = train['SibSp']
df_con['SibSp'] = train['SibSp']

plot_count_dist(train, df_dis, 'Survived','SibSp')

# see that heavily skewed to zero, meaning that a majority of the passengers had zero siblings/spouses onboard so most were single

# Check Parch (The number of parents/children the passenger has onboard)

train['Parch'].isna().sum()

train['Parch'].value_counts()

df_dis['Parch'] = train['Parch']
df_con['Parch'] = train['Parch']

plot_count_dist(train, df_dis, 'Survived','Parch')

# similar result to SibSp

# Check Ticket

train[['Ticket','title']][1:10]

# Ticket is weird so let's skip it for this analysis

# Check Fare

train['Fare'].isna().sum()

df_dis['Fare'] = pd.cut(train['Fare'], 3)
df_con['Fare'] = train['Fare']

plot_count_dist(data=train,
                bin_df=df_dis,
                label_column='Survived', 
                target_column='Fare', 
                figsize=(10,5), 
                use_bin_df=True)

# Check Cabin (Where the passenger was staying)

train['Cabin'].isna().sum()

train['Cabin'].value_counts()

train[['Cabin','Fare']][20:30]

oc = train[['Cabin','Fare','Embarked']]

oc = oc.sort_values(by = 'Fare', ascending= False).reset_index()
oc[235:250]
oc[250:]
for i in range(len(oc)):
    if math.isnan(oc['']):
        oc['Cabin'].iloc[i] = 'C23 C25 C27'

len(train.loc[(train['Embarked'] == 'S') == True]['Cabin'].unique())

len(train.loc[train['Fare'] < 60]['Cabin'].unique())

# Drop for now

# Check Embarked

train['Embarked'].isna().sum()

# 2 missing values, we can drop those 

df_dis['Embarked'] = train['Embarked']
df_con['Embarked'] = train['Embarked']

df_dis = df_dis.dropna()
len(df_dis)
df_con = df_con.dropna()
len(df_con)

df_dis['Title'] = pd.factorize(df_dis['Title'])[0]
df_con['Title'] = pd.factorize(df_dis['Title'])[0]

#df_dis.drop(['Title'],axis=1)
#df_con.drop(['Title'],axis=1)
# Feature Encoding

one_hot_cols = df_dis.columns.tolist()
one_hot_cols.remove('Survived')

df_dis_enc = pd.get_dummies(df_dis, columns = one_hot_cols)

df_dis_enc.head()

df_con.head()

con_hot_cols = df_con.columns.tolist()
con_hot_cols.remove('Survived')
con_hot_cols.remove('Parch')
con_hot_cols.remove('Fare')
con_hot_cols.remove('Age')
con_hot_cols.remove('SibSp')
con_hot_cols.remove('Title')

df_con_enc = pd.get_dummies(df_con, columns = con_hot_cols)

df_con_enc.head()

# Building a model

x_train = df_con_enc.drop('Survived', axis = 1)
y_train = df_con_enc['Survived']

#create a function to spit out the results all the time instead of needing to constantly retype it

def ml_algo(algo, x_train, y_train, cv):
    
    #one pass
    model = algo.fit(x_train, y_train)
    accuracy = round(model.score(x_train, y_train) * 100, 2)

    #cross validation
    train_pred = model_selection.cross_val_predict(algo, x_train, y_train, cv = cv, n_jobs=-1)

    #cross validation accuracy metric
    accuracy_cv = round(metrics.accuracy_score(y_train, train_pred) * 100 , 2)

    return train_pred, accuracy, accuracy_cv

# Logistic Regression

start_time = time.time()

train_pred_logit, acc_logit, acc_cv_logit = ml_algo(LogisticRegression(), x_train, y_train, 10)

log_time = (time.time() - start_time)
print("Accuracy: %s" % acc_logit)
print("Accuracy CV 10-Fold: %s" % acc_cv_logit)
print("Running Time: %s" % datetime.timedelta(seconds=log_time))

# K-Nearest Neighbors

start_time = time.time()

train_pred_knn, acc_knn, acc_cv_knn = ml_algo(KNeighborsClassifier(), x_train, y_train, 10)

knn_time = (time.time() - start_time)
print("Accuracy: %s" % acc_knn)
print("Accuracy CV 10-Fold: %s" % acc_cv_knn)
print("Running Time: %s" % datetime.timedelta(seconds=knn_time))

# Gaussian Naive Bayes

start_time = time.time()
train_pred_gaussian, acc_gaussian, acc_cv_gaussian = ml_algo(GaussianNB(), x_train, y_train, 10)

gaussian_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gaussian)
print("Accuracy CV 10-Fold: %s" % acc_cv_gaussian)
print("Running Time: %s" % datetime.timedelta(seconds=gaussian_time))

# Linear Support Vector Machines (SVC)

start_time = time.time()
train_pred_svc, acc_linear_svc, acc_cv_linear_svc = ml_algo(LinearSVC(),x_train, y_train, 10)

linear_svc_time = (time.time() - start_time)
print("Accuracy: %s" % acc_linear_svc)
print("Accuracy CV 10-Fold: %s" % acc_cv_linear_svc)
print("Running Time: %s" % datetime.timedelta(seconds=linear_svc_time))

# Stochastic Gradient Descent

start_time = time.time()
train_pred_sgd, acc_sgd, acc_cv_sgd = ml_algo(SGDClassifier(),x_train, y_train, 10)

sgd_time = (time.time() - start_time)
print("Accuracy: %s" % acc_sgd)
print("Accuracy CV 10-Fold: %s" % acc_cv_sgd)
print("Running Time: %s" % datetime.timedelta(seconds=sgd_time))

# Decision Tree Classifier

start_time = time.time()
train_pred_dt, acc_dt, acc_cv_dt = ml_algo(DecisionTreeClassifier(),x_train, y_train, 10)

dt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_dt)
print("Accuracy CV 10-Fold: %s" % acc_cv_dt)
print("Running Time: %s" % datetime.timedelta(seconds=dt_time))

# Gradient Boost Trees

start_time = time.time()
train_pred_gbt, acc_gbt, acc_cv_gbt = ml_algo(GradientBoostingClassifier(), x_train,y_train, 10)

gbt_time = (time.time() - start_time)
print("Accuracy: %s" % acc_gbt)
print("Accuracy CV 10-Fold: %s" % acc_cv_gbt)
print("Running Time: %s" % datetime.timedelta(seconds=gbt_time))

# CatBoost Algorithm

cat_features = np.where(x_train.dtypes != np.float)[0]

cat_features

train_pool = Pool(x_train, y_train, cat_features)

catboost_mod = CatBoostClassifier(iterations= 1000, custom_loss=['Accuracy'],loss_function='Logloss')

import ipywidgets

catboost_mod.fit(train_pool, plot = True)

acc_catboost = round(catboost_mod.score(x_train, y_train) * 100, 2)

# Catboost cross validation

start_time = time.time()

# Set params for cross-validation as same as initial model
cv_params = catboost_mod.get_params()

# Run the cross-validation for 10-folds (same as the other models)
cv_data = cv(train_pool,
             cv_params,
             fold_count=10,
             plot=True)

# How long did it take?
catboost_time = (time.time() - start_time)

# CatBoost CV results save into a dataframe (cv_data), let's withdraw the maximum accuracy score
acc_cv_catboost = round(np.max(cv_data['test-Accuracy-mean']) * 100, 2)

# Print out the CatBoost model metrics
print("---CatBoost Metrics---")
print("Accuracy: {}".format(acc_catboost))
print("Accuracy cross-validation 10-Fold: {}".format(acc_cv_catboost))
print("Running Time: {}".format(datetime.timedelta(seconds=catboost_time)))

# Comparing all the model results

models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_knn, 
        acc_logit,  
        acc_gaussian, 
        acc_sgd, 
        acc_linear_svc, 
        acc_dt,
        acc_gbt,
        acc_catboost
    ]})
print("---Regular Accuracy Scores---")
models.sort_values(by='Score', ascending=False)

# As we can see, decision tree provided the best results, next being gradient descent for normal fits

# For cv

cv_models = pd.DataFrame({
    'Model': ['KNN', 'Logistic Regression', 'Naive Bayes', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'Gradient Boosting Trees',
              'CatBoost'],
    'Score': [
        acc_cv_knn, 
        acc_cv_logit,      
        acc_cv_gaussian, 
        acc_cv_sgd, 
        acc_cv_linear_svc, 
        acc_cv_dt,
        acc_cv_gbt,
        acc_cv_catboost
    ]})
print('---Cross-validation Accuracy Scores---')
cv_models.sort_values(by='Score', ascending=False)

# when comaring cv models, catboost was the best, followed by gradient trees again

# Feature Importance
def feature_importance(model, data):
    """
    Function to show which features are most important in the model.
    ::param_model:: Which model to use?
    ::param_data:: What data to use?
    """
    fea_imp = pd.DataFrame({'imp': model.feature_importances_, 'col': data.columns})
    fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False]).iloc[-30:]
    _ = fea_imp.plot(kind='barh', x='col', y='imp', figsize=(20, 10))
    return fea_imp
    #plt.savefig('catboost_feature_importance.png') 

feature_importance(catboost_mod, x_train)
plt.show()

# Precision and Recall

metrics = ['Precision', 'Recall', 'F1', 'AUC']

eval_metrics = catboost_mod.eval_metrics(train_pool,
                                           metrics=metrics,
                                           plot=True)

for metric in metrics:
    print(str(metric)+": {}".format(np.mean(eval_metrics[metric])))

# Testing the data

test.head()

df_con_enc.head()

# One hot encode the columns in the test data frame (like X_train)
test_embarked_one_hot = pd.get_dummies(test['Embarked'], 
                                       prefix='Embarked')

test_sex_one_hot = pd.get_dummies(test['Sex'], 
                                prefix='Sex')

test_plcass_one_hot = pd.get_dummies(test['Pclass'], 
                                   prefix='Pclass')

test = pd.concat([test, 
                  test_embarked_one_hot, 
                  test_sex_one_hot, 
                  test_plcass_one_hot], axis=1)

test.head()
test = test.rename(columns = {'Name': 'Title'})
test.head()
# Create a list of columns to be used for the predictions



wanted_test_columns = x_train.columns
wanted_test_columns
test[wanted_test_columns]

pred_test = catboost_mod.predict(test[wanted_test_columns])

pred_test[:20]

# Create a submission dataframe and append the relevant columns
submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = pred_test # our model predictions on the test dataset
submission.head()

gender_submit.head()

pd.crosstab(gender_submit['Survived'],pred_test)

# Are our test and submission dataframes the same length?
if len(submission) == len(test):
    print("Submission dataframe is the same length as test ({} rows).".format(len(submission)))
else:
    print("Dataframes mismatched, won't be able to submit to Kaggle.")

# Convert submisison dataframe to csv for submission to csv 
# for Kaggle submisison
submission.to_csv('~/Desktop/data/titanic_submission.csv', index=False)
print('Submission CSV is ready!')

