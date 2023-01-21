# -*- coding: utf-8 -*-
"""
@author: morfoula
"""
#Importing packages
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = 'browser'

df = pd.read_excel('C:/Users/morfo/Desktop/10K-MDA-Section-master/Model_Dataset.xlsx')
df.head()
X=df['Data_Edited']
Y=df['Fraudulent']


#1 WordCloud

#create a list of responses for each candidate using a list comprehension
cases = df.Fraudulent.unique()
corpus = [' '.join(df[(df.Fraudulent==c)].Data_Edited.tolist()) for c in cases]
cv=CountVectorizer(ngram_range=(3, 4))

#fit transform our text and create a dataframe with the result
X = cv.fit_transform(corpus)
X = X.toarray()
bow=pd.DataFrame(X, columns = cv.get_feature_names_out())
bow.index=cases

#---------------------------------------1---------------------------------------#

#create a pandas Series of the top 100 most frequent words
text=bow.loc[1].sort_values(ascending=False)[:100]

#create a dictionary Note: you could pass the pandas Series directoy into the wordcloud object
text2_dict=bow.loc[1].sort_values(ascending=False).to_dict()

#create the WordCloud object
wordcloud = WordCloud(min_word_length=3,
                      background_color='white', width=1600, height=800, colormap='gnuplot', collocation_threshold=50)

#generate the word cloud
wordcloud.generate_from_frequencies(text2_dict)

#plot
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('C:/Users/morfo/Desktop/Figure 1a.png',bbox_inches='tight')
plt.show()


#bar plot
text=bow.loc[1].sort_values(ascending=False)[:20]
text=pd.Series(text, name='value')
text=pd.DataFrame(text)
text.reset_index(inplace=True)
text.columns
x = text["index"]
y = text["value"]
fig = px.bar(text, title='Frequency of 20 Most Common Phrases | Class : Fraudulent', x='index', y='value'
             , template='plotly_white', color_discrete_sequence=px.colors.qualitative.Pastel, text_auto=True)
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_layout(legend=dict(x=0.1, y=1.1))
fig.update_yaxes(title='Frequency of phrases', showticklabels=True)
fig.update_xaxes(title='Phrases', showticklabels=True)
fig.show()

#---------------------------------------0---------------------------------------#

#create a pandas Series of the top 100 most frequent words
text=bow.loc[0].sort_values(ascending=False)[:100]

#create a dictionary Note: you could pass the pandas Series directoy into the wordcloud object
text2_dict=bow.loc[0].sort_values(ascending=False).to_dict()

#create the WordCloud object
wordcloud = WordCloud(min_word_length=3,
                      background_color='white', width=1600, height=800, colormap='gnuplot2', collocation_threshold=50)

#generate the word cloud
wordcloud.generate_from_frequencies(text2_dict)

#plot
plt.figure( figsize=(20,10) )
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('C:/Users/morfo/Desktop/Figure 2a.png',bbox_inches='tight')
plt.show()


#bar plot
text=bow.loc[0].sort_values(ascending=False)[:20]
text=pd.Series(text, name='value')
text=pd.DataFrame(text)
text.reset_index(inplace=True)
text.columns
x = text["index"]
y = text["value"]
fig = px.bar(text, title='Frequency of 20 Most Common Phrases | Class : Non-Fraudulent', x='index', y='value'
             , template='plotly_white', color_discrete_sequence=px.colors.qualitative.Pastel, text_auto=True)
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_layout(legend=dict(x=0.1, y=1.1))
fig.update_yaxes(title='Frequency of phrases', showticklabels=True)
fig.update_xaxes(title='Phrases', showticklabels=True)
fig.show()


#2 Comparative Bar Chart - WordCloud

#Fraudulent
text_1=bow.loc[1].sort_values(ascending=False)[:20]
text_1a=pd.Series(text_1, name='value')
df1=pd.DataFrame(text_1a)

df1.reset_index(inplace=True)
df1.rename(columns={'index':'ngrams'},inplace=True)
df1['Fraudulent']=1

#Non Fraudulent
text_0=bow.loc[0].sort_values(ascending=False)[:20]
text_0a=pd.Series(text_0, name='value')
df0=pd.DataFrame(text_0a)

df0.reset_index(inplace=True)
df0.rename(columns={'index':'ngrams'},inplace=True)
df0['Fraudulent']=0

df_vis=pd.concat([df1,df0], ignore_index=True)



fig = px.bar(df_vis, title='Phrases - Comparison: ' + 'Fraudulent' + ' | ' + 'Non-Fraudulent ', x='ngrams', y='value'
             , color='Fraudulent', template='plotly_white', color_discrete_sequence=px.colors.qualitative.Pastel
             , labels={'Fradulent': 'Class:', 'ngrams': 'Phrase'},text_auto=True)
fig.update_layout(legend_orientation="h")
fig.update_layout(xaxis = dict(
        tick0 = 0.8,
        dtick = 0.75),font=dict(size=18, color="black"))
fig.update_layout(legend=dict(x=0.1, y=1.1))
fig.update_yaxes(title='', showticklabels=False)
fig.show()


#3 Polarity/Subjectivity Distributions
#Polarity
fig=px.histogram(df, x="polarity", title='Comparison: ' + 'Fraudulent' + ' | ' + 'Non-Fraudulent Polarity',color='Fraudulent',template='plotly_white', color_discrete_sequence= ['navy','darkorange'],
                 labels={'Fradulent': 'Class:', 'polarity': 'Polarity'}
   ,text_auto=True )

fig.update_xaxes(title='Polarity Score', showticklabels=True)
fig.update_layout(font=dict(size=18, color="black"))
fig.update_yaxes(title='', showticklabels=True)
fig.show()

#Subjectivity
fig=px.histogram(df, x="subj", title='Comparison: ' + 'Fraudulent' + ' | ' + 'Non-Fraudulent Subjectivity',color='Fraudulent',template='plotly_white', color_discrete_sequence= ['mediumorchid','gold'],
                 labels={'Fradulent': 'Class:', 'subj': 'Subjectivity'}
   ,text_auto=True )

fig.update_xaxes(title='Subjectivity Score', showticklabels=True)
fig.update_layout(font=dict(size=18, color="black"))
fig.update_yaxes(title='', showticklabels=True)
fig.show()

#4 Correlation between features

df.columns
X=df[[ 'sentence_count', 'flesch_ease',
       'fog_index', 'pos_count', 'neg_count', 'unc_count', 'lit_count',
       'compound', 'neg', 'neu', 'pos', 'polarity', 'subj']]
corr=X.corr()
print(corr)
fig=px.imshow(corr,text_auto=True,title='Correlation Matrix of Features')
fig.update_layout(font=dict(size=18, color="black"))
fig.show()



