#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[5]:


movies.head(1)


# In[6]:


credits.head(1)


# In[7]:


movies = movies.merge(credits,on='title')


# In[8]:


movies.head(1)


# In[9]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.info()


# In[12]:


movies.isnull().sum()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[17]:


movies['genres']=movies['genres'].apply(convert)


# In[ ]:


movies.head()


# In[18]:


movies['keywords']=movies['keywords'].apply(convert)


# In[19]:


movies.head()


# In[20]:


movies.iloc[0].cast


# In[21]:


def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[22]:


movies['cast']=movies['cast'].apply(convert3)


# In[23]:


movies.head()


# In[24]:


movies['crew'][0]


# In[25]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if(i['job']=='Director'):
           L.append(i['name'])
           break
    return L


# In[26]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[27]:


movies.head()


# In[28]:


movies['overview'][0]


# In[29]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[30]:


movies.head()


# In[31]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])


# In[32]:


movies.head()


# In[33]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[34]:


movies.head()


# In[35]:


new_df = movies[['movie_id','title','tags']]


# In[36]:


new_df['tags']


# In[37]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[38]:


new_df.head()


# In[39]:


new_df['tags'][0]


# In[40]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[41]:


new_df.head()


# In[42]:


new_df['tags'][0]


# In[43]:


import nltk


# In[44]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[45]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[46]:


new_df['tags']=new_df['tags'].apply(stem)


# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[48]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[49]:


vectors


# In[50]:


vectors[0]


# In[51]:


cv.get_feature_names()


# In[52]:


ps.stem('actions')


# In[53]:


stem('marine')


# In[54]:


from sklearn.metrics.pairwise import cosine_similarity


# In[55]:


similarity = cosine_similarity(vectors)


# In[56]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[57]:


similarity[1]


# In[58]:


def recommend(movie):
    movie_index = new_df[new_df['title']== movie].index[0]
    distances = similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[59]:


recommend('Avatar')


# In[60]:


new_df.iloc[3083].title


# In[61]:


recommend('Batman Begins')


# In[62]:


import pickle


# In[63]:


pickle.dump(new_df,open('movie_dict.pkl','wb'))


# In[64]:


new_df['title'].values


# In[67]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))

