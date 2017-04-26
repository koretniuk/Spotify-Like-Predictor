
# coding: utf-8

# For this project I will be using three modules:
# * GaussianNB from sklearn.naive_bayes which will be the learner and predictor
# * pandas for manipulation with csv files
# * dateutil.parser to clean up the data

# In[2]:

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import dateutil.parser as parser


# In[3]:

# Creating classifyer

clf = GaussianNB()


# In[4]:

# train is used interchangebly with fit. So when we say fit it means - train. 
# clf.fit(Var1, Var2), where Var1 - features and Var2 - labels
#clf.predict() - prediction based on trained data. We put Var1 (feature) on input and receive Var (label) on output
# var pred - vector of predictions
# accuracy is no. of points classified correctly / all points (in test set)
# to determine accuracy, clf.score(X, y, sample_weight=None) is used. 
# another method is to use accurac_score from sklearn.metrics import accuracy_score
# accuracy_score(pred, label_test)


# In[5]:

# Loading features (here - pl) and labels (here - labels) using pandas

pl = pd.read_csv('songs_i_like.csv')
labels = pd.read_csv('labels.csv')


# In[6]:

#creating DataFrames based on imported csv-files

pl = pd.DataFrame(pl)
labels = pd.DataFrame(labels)

# At this point it came on me that I do not really need IDs, Title of song, and Artist for this excersise. 
# I am getting rid of thos columns.
# At the same time I decided that I want to keep them in csv-files for future when I will be able to match
# song title and artist with prediction results.

pl = pl.drop(pl.columns[[0, 1, 2]], axis=1)


# In[7]:

# Creating array out of DataFrames with features to keep them as sub-arrays [[1,2,3],[1,2,3]]

X = pl.as_matrix()

# labels need to be one dimensional array, so I am converting it to that [[1],[0]] -> [1,0]

labels = labels.values.flatten()


# In[8]:

# At this point of time I decided that the full date format is not necessary, so I am extracting year only.
# At the same time I am extracting minutes value (without seconds) to simplify it a bit and convert to int.

i=0
while i < len(X):
    X[i][0] = parser.parse(X[i][0]).year
    X[i][6] = parser.parse(X[i][6]).hour
    i += 1


# In[9]:

# Launching learning with .fit()

clf.fit(X, labels)

# Checking how succesful the learning was

clf.score(X, labels)


# In[10]:

# Entering song by t.a.t.u - not gonna get us (my childhood's crush) to verify accuracy. 
# Expecting 1 (like) or 0 (dislike).

clf.predict([2014,120,89,72,-6,80,5,0,14])


# In[ ]:



