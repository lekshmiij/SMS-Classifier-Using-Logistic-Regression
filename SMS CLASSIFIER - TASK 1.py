#!/usr/bin/env python
# coding: utf-8

# # SMS CLASSIFIER
# ### classifying the SMS into spam and not spam(ham)

# ##### NOTE : Logistic regression model is used here as it is considered as the best option for binary classification task

# #### importing the libraries

# In[25]:


import numpy as np  ## to create numpy arrays 
import pandas as pd ## to create dataframes
from sklearn.model_selection import train_test_split ## to split dataset into training and testing
from sklearn.feature_extraction.text import TfidfVectorizer ## convert text data to numerical form
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score ## for evaluation of model


# #### Data insertion and preprocessing

# In[27]:


raw_data = pd.read_csv('C:/Users/lekshmi/Downloads/maildata.csv')


# In[28]:


print(raw_data)


# In[29]:


#as it contains null values, we are replacing the null values with a null string
data = raw_data.where((pd.notnull(raw_data)),'')


# In[31]:


data.head()


# In[32]:


# number of rows and columns in the data
data.shape


# #### for better understanding by the model, we are using label encoding i.e label spam mail as 0;  ham mail as 1;

# In[37]:


data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1


# In[38]:


# separating the data into texts and label

X = data['Message']

Y = data['Category']


# In[39]:


print(X)


# In[40]:


print(Y)


# #### splitting of data into training and testing data for model training

# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


# ##### here training data is accounted for 80 percent. also random state is used to have the same set of data every time the code runs

# In[42]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# #### feature extraction

# In[15]:


# transform the text data to feature vectors that can be used as input to the Logistic regression

feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase=True)


# ##### here the vectoriser operates by giving scores to the words. min_df implies that words with frequency more than 1 is to be considered, stop_words=english indicates that words like this,that, the is to be ignored

# In[44]:


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

# convert Y_train and Y_test values as integers

Y_train = Y_train.astype('int') ## to convert into integre from objects or string
Y_test = Y_test.astype('int')


# In[45]:


print(X_train)


# In[17]:


print(X_train_features)


# #### training the model using logistic regression

# In[18]:


model = LogisticRegression()


# In[46]:


# training the Logistic Regression model with the training data
model.fit(X_train_features, Y_train)


# #### evaluation of model

# In[47]:


# prediction on training data

pred_train = model.predict(X_train_features)
accuracy_train = accuracy_score(Y_train, pred_train)


# In[48]:


print('Accuracy on training data : ', accuracy_train)


# ##### as the accuracy score is more than 95%, the model fitted is good

# In[49]:


# prediction on test data

pred_test = model.predict(X_test_features)
accuracy_test = accuracy_score(Y_test, pred_test)


# In[50]:


print('Accuracy on test data : ', accuracy_test)


# ##### the accuracy score is similar to training data. so that means the model is free from overfitting or underfitting

# ### PREDICTION

# In[53]:


input_sms = ["I can't express how grateful I am for your support. Your kindness has truly been a blessing, and I promise to honor my commitment to you."]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_sms)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham SMS')

else:
  print('Spam SMS')


# In[54]:


input_sms = ["Congratulations! You've been selected to receive a $1000 gift card. Click the link to claim your prize now:. Don't miss out on this exclusive offer!"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_sms)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham SMS')

else:
  print('Spam SMS')


# In[ ]:




