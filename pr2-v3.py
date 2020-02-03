
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap

#sklearn
from sklearn import datasets, svm, metrics
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
#preprocessing
from sklearn.preprocessing import StandardScaler,normalize
# Dimenionality Reduction
from sklearn.decomposition import PCA,FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import random_projection
#Feature selection
from sklearn.feature_selection import VarianceThreshold
#Under sampling
from imblearn.under_sampling import RandomUnderSampler
#Over sampling
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, SMOTENC,RandomOverSampler
#Combined sampling
from imblearn.combine import SMOTETomek
#Algorithms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier,AdaBoostClassifier,RandomForestClassifier,ExtraTreesClassifier,GradientBoostingClassifier,GradientBoostingRegressor
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier,Perceptron,PassiveAggressiveClassifier,RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier

from sklearn.utils import resample
from sklearn.pipeline import *
from sklearn.metrics import f1_score,confusion_matrix,classification_report,make_scorer,average_precision_score,precision_recall_curve

import pandas_ml as pdml

import warnings
warnings.filterwarnings("ignore")

np.random.seed(30)
from scipy.stats import itemfreq


# In[2]:


#Read data
train_data = np.genfromtxt("data/train.dat")
test_data_final = np.genfromtxt("data/test.dat")
train_labels=pd.read_csv('data/train.labels',header=None) #read labels as panda dataframe
train_labels = train_labels.values #convert labels from panda dataframe to numpy
labels = train_labels.ravel()

train_data_final=train_data
train_labels_final=train_labels


# In[3]:


#Split into test-train data in ration of 80:20
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels,test_size=0.20,stratify=train_labels,random_state=35)
y1=y_train


# In[4]:


#Scale the threshold
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[5]:


#Feature selection using variance threshold

fs = VarianceThreshold()
X_train_fs=fs.fit_transform(X_train_scaled)
X_test_fs=fs.transform(X_test_scaled)


# In[6]:


#Oversampling using SMOTE to handle imbalanced data
sm = SMOTE('not minority',k_neighbors = 3)
X_sm, y_sm = sm.fit_sample(X_train_fs, y_train)


# In[7]:


# Knn-classifier
X_knn=X_sm
y_knn=y_sm
X_test_knn=X_test_fs
clf_knn=KNeighborsClassifier(n_neighbors = 8)
clf_knn.fit(X_knn,y_knn)
y_pred_knn=clf_knn.predict(X_test_knn)
print('Knn : f1_score = ',f1_score(y_test, y_pred_knn,average='weighted'))


# In[8]:


#Random Forest classifier
X_rf=X_sm
y_rf=y_sm
X_test_rf=X_test_fs

clf_rf=RandomForestClassifier()
clf_rf.fit(X_rf,y_rf)
y_pred_rf=clf_rf.predict(X_test_rf)
print('Randomforest : f1_score = ',f1_score(y_test, y_pred_rf,average='weighted'))


# In[9]:


#Extra trees classifier
X_test_et=X_test_fs
X_et=X_sm
y_et=y_sm
clf_et=ExtraTreesClassifier(n_estimators=1000,class_weight='balanced')
clf_et.fit(X_et,y_et)
y_pred_et=clf_et.predict(X_test_et)
print('Extra-tree : f1_score = ',f1_score(y_test, y_pred_et,average='weighted'))


# In[10]:


#Extra trees classifier on test data
X_et=X_sm
y_et=y_sm
clf_et=ExtraTreesClassifier(n_estimators=1000,class_weight='balanced')
clf_et.fit(X_et,y_et)
y_pred_test=clf_et.predict(X_test_fs)
print(y_pred_test)


# In[11]:


y_pred_test.shape


# In[12]:


np.savetxt('pr2-nov1-vc-v3_1.dat', y_pred_test, delimiter=" ", fmt="%s")

