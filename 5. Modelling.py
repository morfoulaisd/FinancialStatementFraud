# -*- coding: utf-8 -*-
"""
@author: morfoula
"""
#Importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.util import ngrams
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import re
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.metrics import roc_auc_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
pio.renderers.default = 'browser'

df = pd.read_excel('C:/Users/morfo/Desktop/10K-MDA-Section-master/Model_Dataset.xlsx')


df.head()
df.drop_duplicates(inplace=True)

#-----------------Modelling with linguistic features-----------------------------------#


#Linguistic Features
#Defining X and Y Variables

df.columns
X=df[['sentence_count', 'flesch_ease',
        'pos_count', 'neg_count', 'unc_count', 'lit_count',
       'compound', 'neg', 'pos', 'polarity', 'subj']]
Y=df['Fraudulent']

#Train_Test_Split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25, random_state=0)


#Hyperparameter Tuning
classifier=RandomForestClassifier()

#Parameters
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 4, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]


# Create the random grid
random_grid= {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)

# Best Parameters based on RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = classifier,param_distributions = random_grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
rf_random.fit(X_train, y_train)
print ('Best Parameters: ', rf_random.best_params_, ' \n')

#Prediction and Results
model = RandomForestClassifier(n_estimators=600, min_samples_split= 4, min_samples_leaf= 4, max_features= 'sqrt', max_depth= None, bootstrap= True,oob_score=True, random_state=123)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Confusion Matrix
cm=confusion_matrix(y_test, predictions)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("Sensitivity: %.2f%%" %(TPR*100.0))
print("Specificity: %.2f%%" %(TNR*100.0))
print("Precision: %.2f%%" %(PPV*100.0))

# F1 Score
f1score=f1_score(y_test, predictions)
print("F1_score: %.2f%%" % (f1score * 100.0))

#Feature Importances
features=X.columns
importances=model.feature_importances_
sorted_idx=model.feature_importances_.argsort()
features=X.columns
importances=model.feature_importances_
sorted_idx=model.feature_importances_.argsort()
feature_importances=pd.DataFrame({'feature_names':features[sorted_idx], 'importance':importances[sorted_idx]})
fig = px.bar(feature_importances, title='Feature Importances: Linguistic Features', x='importance', y='feature_names'
             , template='plotly_white', color='importance',color_continuous_scale='plasma', text_auto=True, orientation='h')
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_yaxes(title='Linguistic Features', showticklabels=True)
fig.update_xaxes(title='Feature Importances', showticklabels=True)
fig.show()


#-----------------Modelling with ngrams-----------------------------------#
#-------------------------------1-N-gram-------------------------------#
#Defining X and Y Variables
X=df['Data_Edited']
Y=df['Fraudulent']

#Train_Test_Split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
#Vectorization
vect=CountVectorizer(min_df=2,ngram_range=(1,1)).fit(X_train)
feature_names=vect.get_feature_names_out()
X_train_vectorized=vect.transform(X_train)

# Best Parameters based on RandomizedSearchCV
rf=RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train_vectorized, y_train)
print ('Best Parameters: ', rf_random.best_params_, ' \n')

#Prediction and Results (73,24%)
model = RandomForestClassifier(n_estimators= 800, min_samples_split= 5, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= True,oob_score=True, random_state=123)
model.fit(X_train_vectorized,y_train)
y_pred = model.predict(vect.transform(X_test))
predictions = [round(value) for value in y_pred]

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Confusion Matrix
cm=confusion_matrix(y_test, predictions)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("Sensitivity: %.2f%%" %(TPR*100.0))
print("Specificity: %.2f%%" %(TNR*100.0))
print("Precision: %.2f%%" %(PPV*100.0))
# F1 Score
f1score=f1_score(y_test, predictions)
print("F1_score: %.2f%%" % (f1score * 100.0))

#Feature Importances
fi=model.feature_importances_
importance = [(feature_names[i], fi[i]) for i in range(0,len(fi))]
df1 = pd.DataFrame(importance)
df1.columns = ['feature', 'importance']
feature_importances=df1.sort_values(by='importance',ascending=False).iloc[:20,:]
feature_importances=feature_importances.sort_values(by='importance', ascending=True)
fig = px.bar(feature_importances, title='Feature Importances: 1-N-gram', x='importance', y='feature'
             , template='plotly_white', color='importance',color_continuous_scale='plasma', text_auto=True, orientation='h')
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_yaxes(title='1-N-gram', showticklabels=True)
fig.update_xaxes(title='Feature Importances', showticklabels=True)
fig.show()


#-------------------------------1-3-N-grams-------------------------------#
#Defining X and Y Variables
X=df['Data_Edited']
Y=df['Fraudulent']

#Train_Test_Split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
#Vectorization
vect=CountVectorizer(min_df=2,ngram_range=(1,3)).fit(X_train)
feature_names=vect.get_feature_names_out()
X_train_vectorized=vect.transform(X_train)

# Best Parameters based on RandomizedSearchCV
rf=RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train_vectorized, y_train)
print ('Best Parameters: ', rf_random.best_params_, ' \n')

#Prediction and Results(74,65%)
model = RandomForestClassifier(n_estimators= 1200, min_samples_split= 5, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 30, bootstrap= True,oob_score=True,random_state=123)
model.fit(X_train_vectorized,y_train)
y_pred = model.predict(vect.transform(X_test))
predictions = [round(value) for value in y_pred]

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Confusion Matrix
cm=confusion_matrix(y_test, predictions)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("Sensitivity: %.2f%%" %(TPR*100.0))
print("Specificity: %.2f%%" %(TNR*100.0))
print("Precision: %.2f%%" %(PPV*100.0))

# F1 Score
f1score=f1_score(y_test, predictions)
print("F1_score: %.2f%%" % (f1score * 100.0))

#Feature Importances
fi=model.feature_importances_
importance = [(feature_names[i], fi[i]) for i in range(0,len(fi))]
df1 = pd.DataFrame(importance)
df1.columns = ['feature', 'importance']
feature_importances=df1.sort_values(by='importance',ascending=False).iloc[:20,:]
feature_importances=feature_importances.sort_values(by='importance', ascending=True)
fig = px.bar(feature_importances, title='Feature Importances: 1-3-N-grams', x='importance', y='feature'
             , template='plotly_white', color='importance',color_continuous_scale='plasma', text_auto=True, orientation='h')
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_yaxes(title='1-3-N-grams', showticklabels=True)
fig.update_xaxes(title='Feature Importances', showticklabels=True)
fig.show()

#-------------------------------2-4-N-grams-------------------------------#
#Defining X and Y Variables
X=df['Data_Edited']
Y=df['Fraudulent']
#Train_Test_Split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2, random_state=0)
#Vectorization
vect=CountVectorizer(min_df=2,ngram_range=(2,4)).fit(X_train)
feature_names=vect.get_feature_names_out()
X_train_vectorized=vect.transform(X_train)

# Best Parameters based on RandomizedSearchCV
rf=RandomForestClassifier()
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random.fit(X_train_vectorized, y_train)
print ('Best Parameters: ', rf_random.best_params_, ' \n')

#Prediction and Results (76,06%)
model = RandomForestClassifier(n_estimators= 800, min_samples_split= 5, min_samples_leaf= 1, max_features= 'sqrt', max_depth= 100, bootstrap= True,oob_score=True,random_state=123)
model.fit(X_train_vectorized,y_train)
y_pred = model.predict(vect.transform(X_test))
predictions = [round(value) for value in y_pred]

#Accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

#Confusion Matrix
cm=confusion_matrix(y_test, predictions)
TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)

print("Sensitivity: %.2f%%" %(TPR*100.0))
print("Specificity: %.2f%%" %(TNR*100.0))
print("Precision: %.2f%%" %(PPV*100.0))

# F1 Score
f1score=f1_score(y_test, predictions)
print("F1_score: %.2f%%" % (f1score * 100.0))

#Feature Importances
fi=model.feature_importances_
importance = [(feature_names[i], fi[i]) for i in range(0,len(fi))]
df1 = pd.DataFrame(importance)
df1.columns = ['feature', 'importance']
feature_importances=df1.sort_values(by='importance',ascending=False).iloc[:20,:]
feature_importances=feature_importances.sort_values(by='importance', ascending=True)
fig = px.bar(feature_importances, title='Feature Importances: 2-4-N-grams', x='importance', y='feature'
             , template='plotly_white', color='importance',color_continuous_scale='plasma', text_auto=True, orientation='h')
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_yaxes(title='2-4-N-grams', showticklabels=True)
fig.update_xaxes(title='Feature Importances', showticklabels=True)
fig.show()





