#!/usr/bin/env python
# coding: utf-8

# In[36]:


#importing libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from pandas import datetime
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_absolute_error,mean_squared_error
from math import sqrt
import joblib
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#Loading the Historical price data
df_price=pd.read_csv("C:/Users/user9/Downloads/DDOG.csv")
df_price.head()


# In[27]:


type(df_price['Date'][0])
df_price['Date'] = pd.to_datetime(df_price['Date'])
df_price.set_index('Date',inplace=True,drop=False)
df_price.dropna(inplace=True)
df_price.head()


# In[28]:


df_price.describe() #gives summary od historical price data


# Prediction of Stock Price using Numerical Analysis

# In[29]:


#Autocorrelation with lag 3
plt.figure()
lag_plot(df_price['Close'], lag=3)
plt.title('BSE Sensex - Autocorrelation plot with lag = 3')
plt.show()


# In[30]:


#visualization
plt.figure(figsize=(15,6))
plt.plot(df_price["Close"])
plt.xlabel("time")
plt.ylabel("price")
plt.show()      #gives BSE sensex of stock price w.r.t time


# In[32]:


#Modelling the data
train, test = df_price[:int(len(df_price)*0.9)], df_price[int(len(df_price)*0.9):]
plt.figure(figsize=(10,6))
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(df_price['Close'], 'black', label='Train data')
plt.plot(test['Close'], 'grey', label='Test data')
plt.legend()          #gives train,test data


# Prediction of News data using Textual Analysis

# In[37]:


from datetime import datetime
import string
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')


# In[39]:


#Loading the News Headlines data
df_news=pd.read_csv("C:/Users/user9/Downloads/india-news-headlines.csv")
df_news.head()


# In[41]:


df_news.shape  #gives dimensions of data


# In[42]:


df_news['publish_date'] = pd.to_datetime(df_news['publish_date'],format='%Y%m%d')


# In[44]:


#using the data from 2020-01-02 
df1 = df_news[df_news['publish_date']> '2020-01-02'].reset_index(drop=True)
df1.head()


# In[46]:


df1.shape


# In[47]:


df1 = df1.groupby('publish_date').agg({'headline_category':'first','headline_text': '. '.join}).reset_index()
df1.head()       #joining the rows of same dates


# In[48]:


df1.shape


# In[49]:


df1['headline_text'] =df1['headline_text'].str.lower()
count = 0
for df_news in df1['headline_text']:
    if 'bse' in df_news or 'sensex' in df_news :
        count+=1
count        #gives the total count for which BSE or Sensex occurs


# In[51]:


from nltk import sent_tokenize
def actual_news(data):
    for index in data.index:
        sentences = sent_tokenize(data['headline_text'][index])
        relevant_line = ' '.join(sent for sent in sentences if 'sensex' in sent or 'bse' in sent)
        if len(relevant_line)>5:
            data['headline_text'][index] =relevant_line
    return data
df2 = actual_news(df1)


# In[52]:


df2.head()   #gives the actual News headlines


# In[54]:


# Cleaning the text
from nltk import word_tokenize
def clean_txt(df_news):
    # Removing non alphanumeric text
    df_news = re.sub('[^a-zA-Z]',' ',df_news)
    return df_news
df1['headline_text'] = df1['headline_text'].apply(clean_txt)
df2['headline_text'] = df2['headline_text'].apply(clean_txt)
headlines = list(df1['headline_text'])
print(headlines)


# In[56]:


# Calculating the sentiment of each headline
score=[]
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
scores = []
for sentence in headlines:
    scores.append(sid.polarity_scores(sentence)['compound'])
len(scores)        


# Building the Hybrid Model

# In[58]:


# Now let's join the text data to original data
hybrid_data = pd.merge(left=df_price,right=df2,left_on=df_price['Date'],right_on=df2['publish_date'],how='outer')
hybrid_data.head()


# In[59]:


hybrid_data.dropna(inplace=True)


# In[60]:


# Taking only the necessary columns
hybrid_data = hybrid_data[['Date','Close','headline_text']].copy()
# Headlines
headlines = hybrid_data['headline_text'].apply('. '.join)
# Sentiment Score
scores = []
for sentence in headlines:
    sentiment = scores.append(sid.polarity_scores(sentence)['compound'])
hybrid_data['Sentiment'] = np.array(scores)
hybrid_data.isnull().sum()


# In[ ]:




