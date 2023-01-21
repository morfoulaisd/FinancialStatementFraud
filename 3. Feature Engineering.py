# -*- coding: utf-8 -*-
"""
@author: morfoula
"""
#Importing packages
import pandas as pd
import numpy as np


df = pd.read_excel('C:/Users/morfo/Desktop/10K-MDA-Section-master/Cleaned_Dataset.xlsx')


#Feature Engineering based on Loughran-McDonald Dictionary
#Positive Words
pos = pd.read_csv('positive.csv')
pos=pos.iloc[:,0]
pos=pos.str.lower()

pos_count = 0

def loadPositive(sentence):
    pos_count = 0
    for word in pos:
        pos_count += sentence.lower().count(word)
    return (pos_count)

df['pos_count']=df['Data_Edited'].apply(loadPositive)

#Negative Words
neg = pd.read_csv('negative.csv')
neg=neg.iloc[:,0]
neg=neg.str.lower()

neg_count = 0

def loadNegative(sentence):
    neg_count = 0
    for word in neg:
        neg_count += sentence.lower().count(word)
    return (neg_count)

df['neg_count']=df['Data_Edited'].apply(loadNegative)

#Uncertain Words
unc = pd.read_csv('uncertainty.csv')
unc=unc.iloc[:,0]
unc=unc.str.lower()

unc_count = 0

def loadUncertain(sentence):
    unc_count = 0
    for word in unc:
        unc_count += sentence.lower().count(word)
    return (unc_count)

df['unc_count']=df['Data_Edited'].apply(loadUncertain)

#Litigious Words
lit = pd.read_csv('litigious.csv')
lit=lit.iloc[:,0]
lit=lit.str.lower()

lit_count = 0

def loadLitigious(sentence):
    lit_count = 0
    for word in lit:
        lit_count += sentence.lower().count(word)
    return (lit_count)

df['lit_count']=df['Data_Edited'].apply(loadLitigious)


#Feature Engineering based on Sentiment Analyzer

from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

analyzer = SentimentIntensityAnalyzer()
df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['Data_Edited']]
df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['Data_Edited']]
df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['Data_Edited']]
df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['Data_Edited']]
df['polarity']=df['Data_Edited'].apply(lambda x: TextBlob(x).sentiment[0])
df['subj']=df['Data_Edited'].apply(lambda x: TextBlob(x).sentiment[1])

df.drop_duplicates(inplace=True)

df.to_excel(r'C:\Users\morfo\Desktop\10K-MDA-Section-master\Model_Dataset.xlsx',index=False)
