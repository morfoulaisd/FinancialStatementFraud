# -*- coding: utf-8 -*-
"""@author: morfoula
"""

#Importing packages
import pandas as pd
import numpy as np
import glob
import os
from bs4 import BeautifulSoup

#Disable Copy Warning Message
pd.options.mode.chained_assignment = None 

#Read and Load Dataset
data = pd.read_excel('C:/Users/morfo/Desktop/10K-MDA-Section-master/Dataset.xlsx')

#Print Dataset
print(data.head())
data=data.dropna()

#Preprocessing of Text Data

#1. Convert html parts of text to readable text.
str1=data["Data"]

str_test={'Index': [], 'Data_Edited' : []}
for i in range(len(str1)):
    f=data.iloc[i,8]
    str_test['Index'].append(data.iloc[i,0])
    str_test['Data_Edited'].append(BeautifulSoup(f,"html.parser"))


str_test=pd.DataFrame(str_test)

final=pd.merge(data,str_test, on=['Index'])

#2. Selecting initial X & Y variables
df=final[["Data_Edited","Fraudulent"]]

#3. Inspecting variable types

print(df.dtypes)

#4. Text Cleaning

#4.1. Converting "Data_Edited" column to str
df["Data_Edited"]=df["Data_Edited"].apply(str)

#4.2. Stripping "Data_Edited" column
df["Data_Edited"]=df["Data_Edited"].str.strip()

#4.3. Lowercase "Data_Edited" column
df["Data_Edited"]=df["Data_Edited"].str.lower()

#4.4. Replace special characters
df['Data_Edited'] = df['Data_Edited'].replace('\n',' ', regex=True)
df['Data_Edited'] = df['Data_Edited'].replace('&amp;','&', regex=True)
df['Data_Edited'] = df['Data_Edited'].replace('\xa0','', regex=True)

#4.5. Remove whitespaces
def whitespace_remover(dataframe):
   
    # iterating over the columns
    for i in dataframe.columns:
         
        # checking datatype of each columns
        if dataframe[i].dtype == 'object':
             
            # applying strip function on column
            dataframe[i] = dataframe[i].map(str.strip)
        else:
             
            # if condn. is False then it will do nothing.
            pass

whitespace_remover(df)

#4.6. Remove extra spaces
import re

def extraspaces_remover(dataframe):
    return re.sub(' +',' ',dataframe)

df['Data_Edited']=df['Data_Edited'].apply(extraspaces_remover)


#4.7 Feature engineering (before specific characters are removed): 
    #1. Sentence count
    #2. Flesch Reading Ease Score
    #3. Fog Index

import textstat


df['sentence_count']=df['Data_Edited'].apply(textstat.sentence_count)
df['flesch_ease']=df['Data_Edited'].apply(textstat.flesch_reading_ease)
df['fog_index']=df['Data_Edited'].apply(textstat.gunning_fog)


#4.8. Remove headline(s) & phrase 'year ended' & numbers & punctuation

import re

def clean_1(x):
    return re.sub(r'^i.*?7.*?analysis', '',x,flags=re.IGNORECASE)
def clean_2(x):
    return re.sub(r'^\s*of financial condition and results of operations', '',x,flags=re.IGNORECASE)
def clean_3(x):    
    return re.sub(r'\s*of financial condition and results of operation', '', x,flags=re.IGNORECASE)
def clean_4(x): 
    return re.sub(r'\s*of financial condition and plan of operation', '', x,flags=re.IGNORECASE)
def clean_5(x): 
    return re.sub(r'\s*of results of operations and financial condition', '', x,flags=re.IGNORECASE)
def clean_6(x): 
    return re.sub(r'\s*of results of operations financial condition', '', x,flags=re.IGNORECASE)
def clean_7(x): 
    return re.sub(r'\s*of financial condition and overview', '', x,flags=re.IGNORECASE)
def clean_8(x): 
    return re.sub(r'\s*or plan of operation', '', x,flags=re.IGNORECASE)
def clean_9(x): 
    return re.sub(r'\s*or plan of operations', '', x,flags=re.IGNORECASE)
def clean_10(x): 
    return re.sub(r'\s*results of operations', '', x,flags=re.IGNORECASE)
def clean_11(x): 
    return re.sub(r'\s*or financial condition and results of operation', '', x,flags=re.IGNORECASE)
def clean_12(x): 
    return re.sub(r'\s*year ended.*?', '', x,flags=re.IGNORECASE)


      
df['Data_Edited']=df['Data_Edited'].apply(clean_1)
df['Data_Edited']=df['Data_Edited'].apply(clean_2)
df['Data_Edited']=df['Data_Edited'].apply(clean_3)
df['Data_Edited']=df['Data_Edited'].apply(clean_4)
df['Data_Edited']=df['Data_Edited'].apply(clean_5)
df['Data_Edited']=df['Data_Edited'].apply(clean_6)
df['Data_Edited']=df['Data_Edited'].apply(clean_7)
df['Data_Edited']=df['Data_Edited'].apply(clean_8)
df['Data_Edited']=df['Data_Edited'].apply(clean_9)
df['Data_Edited']=df['Data_Edited'].apply(clean_10)
df['Data_Edited']=df['Data_Edited'].apply(clean_11)



df['Data_Edited'] = df['Data_Edited'].str.replace(r'\d+', '', regex=True)

df['Data_Edited'] = df['Data_Edited'].str.replace(r'[^\w\s]+', '')

df['Data_Edited'].head()


#Converting "Data_Edited" column to str
df["Data_Edited"]=df["Data_Edited"].apply(str)

#4.9.Remove blank rows
nan_value=np.nan
df['Data_Edited'].replace(r'^s+$'," ",inplace=True,regex=True)
df.isnull().sum()
df=df[df.Data_Edited != ' ']
df=df[df.Data_Edited != '']

#4.10. Again Remove whitespaces

whitespace_remover(df)

#4.11. Again Remove extra spaces

df['Data_Edited']=df['Data_Edited'].apply(extraspaces_remover)


#4.12. Remove stopwords

from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df['Data_Edited'] = df['Data_Edited'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

#Removal of Phrase 'for the year ended'
df['Data_Edited']=df['Data_Edited'].apply(clean_12)

#Removal of duplicates
df=df.astype(str)
df.drop_duplicates(inplace=True)

#Lemmatization
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]


df['Data_Edited'] = df.Data_Edited.apply(lemmatize_text)
df['Data_Edited']=([' '.join(x) for x in df['Data_Edited']])

df.to_excel(r'C:\Users\morfo\Desktop\10K-MDA-Section-master\Cleaned_Dataset.xlsx',index=False)


