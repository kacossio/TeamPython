#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import all needed libraries
import os
import tweepy as tw
import pandas as pd


# In[19]:


consumer_key = "x7uD7H5UD5rtq40jkTFJxFl6Y" 
consumer_secret = "kJgoakf2z5hGw3ru8SFjhRrpOuNu4VyopgCqE3kr72iWnGZaN6" 
access_token = "770796574813351936-cvG7uCHzQNmRAXsx9UWWfzhQiv3E6O9" 
access_token_secret = "Kuy7m22lCAHGytvUYNWPDgAsppiROWMEC9jXS7UfIRN6Z" 

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)


# In[20]:


# Define the search term and the date_since date as variables
search_words = "military"
date_since = "2018-11-16"


# In[21]:


# Collect tweets
tweets = tw.Cursor(api.search,
                       q=search_words,
                       lang="en",
                       since=date_since).items(20)

# Collect a list of tweets
[tweet.text for tweet in tweets]


# In[22]:


new_search = search_words + " -filter:retweets"
new_search


# In[27]:


tweets = tw.Cursor(api.search, 
                           q=new_search,
                           lang="en",
                           since=date_since).items(20)

users_locs = [[tweet.user.screen_name,tweet.user.name,tweet.user.location,tweet.user.profile_image_url] for tweet in tweets]
users_locs


# In[28]:


tweet_text = pd.DataFrame(data=users_locs, 
                    columns=['Screen_name',"User_name", "location", "Profile_image"])
tweet_text


# In[ ]:




