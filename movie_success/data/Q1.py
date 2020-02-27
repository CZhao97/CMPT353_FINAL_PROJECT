#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[2]:


wiki = pd.read_json('wiki-company.json.gz', orient='record', lines=True)
rt = pd.read_json('rotten-tomatoes.json.gz', orient='record', lines=True)
omdb = pd.read_json('omdb-data.json.gz', orient='record', lines=True)


# In[3]:


# drop NaN columns in 'nbox' and 'ncost'
wiki_dropNA = wiki.dropna(subset=['nbox', 'ncost', 'publication_date'])
# We want to get Profit by 'nbox' and 'ncost'
wiki_dropNA_box = wiki_dropNA[['nbox','ncost', 'publication_date','rotten_tomatoes_id']]

# we want to get only reviews from audiences and critics
rt_reviews = rt[['audience_average', 'audience_percent', 'audience_ratings', 'critic_average', 'critic_percent', 'rotten_tomatoes_id']]

# Merge (join) these two tables by common Rotten Tomatoes ID
wiki_join_rt = wiki_dropNA_box.merge(rt_reviews, left_on='rotten_tomatoes_id', right_on='rotten_tomatoes_id', suffixes=('_wiki', '_rt'))


# In[4]:


# Calculate the profit (Or, loss if nbox < ncost)
wiki_join_rt['profit'] = wiki_join_rt['nbox'] - wiki_join_rt['ncost']

# Filter data a little bit. We just keep movies with more than 40 reviews (Discard very uncommon movies)
wiki_join_rt = wiki_join_rt[wiki_join_rt['audience_ratings']>=40]
wiki_join_rt = wiki_join_rt.dropna(subset=['critic_average', 'critic_percent'])

# Note that there are 817 movies remained, which is not bad though
# wiki_join_rt   # just have a look


# In[5]:


# Inflation adjustment on the profit

def getyear(x):
    return x[0:4]

wiki_join_rt['year'] = wiki_join_rt['publication_date'].apply(getyear)
wiki_join_rt['year'] = pd.to_numeric(wiki_join_rt['year'])

# wiki_join_rt['year'].min()
# This shows the oldest movie comes from 1927
# Through the research on references, from 1927 to 2019, the value of US dollars has an inflation of 2.97% per year on average.
# We adjust this number according to year differences compared to 2019

def inflation_correct(x):
    year_difference = 2019 - wiki_join_rt['year']
    money_value = (1.0297) ** year_difference
    return x * money_value

wiki_join_rt['year_difference'] = 2019 - wiki_join_rt['year']
wiki_join_rt['money_value'] = (1.0297) ** wiki_join_rt['year_difference']
# wiki_join_rt['profit'] = wiki_join_rt['profit'].apply(inflation_correct)
# Calculate the "real" profit for each movie, based on its publication year and real value of money.
wiki_join_rt['profit'] = wiki_join_rt['profit']*wiki_join_rt['money_value']


# In[6]:


# Start to do the Linear Regression Models on these 3 variables (cirtic reviews / audience reviews / Profit): 

# audience reviews vs. profit
reg_audience = stats.linregress(wiki_join_rt['audience_percent'], wiki_join_rt['profit'])
plt.plot(wiki_join_rt['audience_percent'], wiki_join_rt['profit'],'b.')
plt.plot(wiki_join_rt['audience_percent'], reg_audience.slope*wiki_join_rt['audience_percent']+reg_audience.intercept, 'r-')
plt.title('Movies Profit vs. Audience reviews')
plt.xlabel('audience percent (out of 100)')
plt.ylabel('profit (boxoffice - cost)')

residuals_audience = wiki_join_rt['profit'] - (reg_audience.slope*wiki_join_rt['audience_percent'] + reg_audience.intercept)
print(reg_audience.slope, reg_audience.intercept)
print(reg_audience.pvalue)
print(reg_audience.rvalue)
print(reg_audience.rvalue**2)


# In[7]:


plt.hist(residuals_audience)


# In[8]:


# critic reviews vs. profit
reg_critic = stats.linregress(wiki_join_rt['critic_percent'], wiki_join_rt['profit'])
plt.plot(wiki_join_rt['critic_percent'], wiki_join_rt['profit'],'b.')
plt.plot(wiki_join_rt['critic_percent'], reg_critic.slope*wiki_join_rt['critic_percent']+reg_critic.intercept, 'r-')
plt.title('Movies Profit vs. Critic reviews')
plt.xlabel('Critic percent (out of 100)')
plt.ylabel('profit (boxoffice - cost)')

residuals_critic = wiki_join_rt['profit'] - (reg_critic.slope*wiki_join_rt['critic_percent'] + reg_critic.intercept)
print(reg_critic.slope, reg_critic.intercept)
print(reg_critic.pvalue)
print(reg_critic.rvalue)
print(reg_critic.rvalue**2)


# In[9]:


plt.hist(residuals_critic)


# In[10]:


# audience reviews vs. critic reviews
reg_reviews = stats.linregress(wiki_join_rt['critic_percent'], wiki_join_rt['audience_percent'])
plt.plot(wiki_join_rt['critic_percent'], wiki_join_rt['audience_percent'],'b.')
plt.plot(wiki_join_rt['critic_percent'], reg_reviews.slope*wiki_join_rt['critic_percent']+reg_reviews.intercept, 'r-')
plt.title('Audience reviews vs. Critic reviews')
plt.xlabel('Critic percent (out of 100)')
plt.ylabel('Audience percent (out of 100)')

residuals_reviews = wiki_join_rt['audience_percent'] - (reg_reviews.slope*wiki_join_rt['critic_percent'] + reg_reviews.intercept)
print(reg_reviews.slope, reg_reviews.intercept)
print(reg_reviews.pvalue)
print(reg_reviews.rvalue)
print(reg_reviews.rvalue**2)


# In[11]:


plt.hist(residuals_reviews)


# In[12]:


rt_reviewsOnly = rt.drop(['rotten_tomatoes_id','imdb_id', 'audience_ratings'],axis=1)
rt_reviewsOnly.corr()

