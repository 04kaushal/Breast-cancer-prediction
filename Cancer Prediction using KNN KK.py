#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import warnings
#import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import scatter_matrix
warnings.filterwarnings("ignore")
os.chdir(r"C:\Users\ezkiska\Videos\Imarticus\Python\3rd week 21st nd 22nd dec\sun 22 Logistic Reg nd K mean\cancerData\cancerData")
cancer_data = pd.read_csv('cancerdata.csv')


# In[3]:


cancer_data.head()


# In[4]:


cancer_data.columns


# In[5]:


cancer_data = cancer_data.drop(['id'], axis=1)
#Let's focus on Target variable now  which is diagnosis
cancer_data['diagnosis'].value_counts()


# In[6]:


cancer_data.isnull().sum() # check for null values


# In[7]:


cancer_data.dtypes


# In[8]:


#Let's map Target as 1 and 0

cancer_data['diagnosis'] = cancer_data.diagnosis.map({'B':0, 'M':1})


# In[10]:


cancer_data.describe()


# In[11]:


cancer_data.dtypes


# In[13]:


cancer_data.info()


# In[14]:


cancer_data.shape


# In[15]:


cancer_data['diagnosis'].value_counts()


# In[16]:


#calculation proportioon
major = np.round(cancer_data.diagnosis.value_counts()[0]/cancer_data.shape[0],3)*100
minor = 100 - major # fairly balanced data


# In[17]:


major


# In[18]:


minor


# In[19]:


# Plot histograms for each variable
cancer_data.hist(figsize = (15, 15))
plt.show()


# In[20]:


'''
DATA VISUALISATION
'''

sns.set(style='darkgrid', palette='husl', 
        font='sans-serif', font_scale=1, color_codes=True, rc=None)
sns.countplot(x=cancer_data['diagnosis'], data = cancer_data)


# In[21]:


groups = cancer_data.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.perimeter_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.xlabel("perimeter_mean")
plt.ylabel("texture_mean")
plt.show() 


# In[22]:


groups = cancer_data.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.radius_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("radius_mean")
plt.show()


# In[23]:


groups = cancer_data.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.area_mean, group.texture_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("texture_mean")
plt.xlabel("area_mean")
plt.show() 


# In[24]:


groups = cancer_data.groupby('diagnosis')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.smoothness_mean, group.compactness_mean, marker='o', ms=3.5, linestyle='', 
            label = 'Malignant' if name == 1 else 'Benign')
ax.legend()
plt.ylabel("compactness_mean")
plt.xlabel("smoothness_mean")
plt.show() 


# In[25]:


groups = cancer_data.groupby('diagnosis')
fig, ax = plt.subplots()

for name, group in groups:
    ax.plot(group.concavity_mean, group.symmetry_mean, marker='o', linestyle='',
            ms = 3.5, label = 'Malignant' if name ==1 else 'Benign')
ax.legend()
plt.xlabel ("concavity_mean")
plt.ylabel ("symmetry_mean")
plt.show()


# In[26]:


'''
Checking the correlation between variables

'''

plt.figure(figsize=(18,18))
sns.heatmap(cancer_data.corr(), annot= True, cmap = 'cubehelix_r')
plt.show()


# In[28]:


#Method 1 :
################################### Wihtout scaling #####################################

print('*'*80)
print('WITHOUT SCALING:')

#loading libraries

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

X = np.array(cancer_data.iloc[:,1:])
y = np.array(cancer_data['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(X_train,y_train)

knn.score(X_test,y_test)

KNN_score = knn.score(X_test,y_test)

'''
#Performing cross validation
'''
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
#perform 10 fold cross validation
for k in range(1,51,2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = cross_val_score(knn,X_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores.append(scores.mean())

#Misclassification error versus k
MSE = [1-i for i in cv_scores]

#determining the best k
optimal_k = neighbors[MSE.index(min(MSE))]
print('The optimal number of neighbors is: %d ' %optimal_k)

#plot misclassification error versus k

plt.figure(figsize = (10,6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Misclassification Error')
plt.show()


# In[29]:


KNN_score


# In[30]:


#Without Hyper Parameters Tuning
#importing the metrics module

from sklearn import metrics
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#learning
model.fit(X_train,y_train)
#Prediction
prediction=model.predict(X_test)
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Matrix:\n",metrics.confusion_matrix(prediction,y_test))


# In[32]:


#With Hyper Parameters Tuning
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[9,11,13,15],
          'leaf_size':[25,30,35,40],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(X_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(X_test)
#evaluation(Accuracy)
print("Accuracy after tuning:",metrics.accuracy_score(prediction,y_test))
#evaluation(Confusion Metrix)
print("Confusion Matrix after tuning:\n",metrics.confusion_matrix(prediction,y_test))


# In[34]:


############################### AFter scaling ##################################################
print('*'*80)
print('AFTER SCALING:')
#loading libraries

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

#X = np.array(data.iloc[:,1:])
#y = np.array(data['diagnosis'])

scaler = StandardScaler()
Xs = scaler.fit_transform(X) # X has been scaled using z transformation (normal assumption from plotting)

Xs_train, Xs_test, y_train, y_test = train_test_split(Xs,y, test_size = 0.33, random_state = 42)

knn = KNeighborsClassifier(n_neighbors = 12)
knn.fit(Xs_train,y_train)

knn.score(Xs_test,y_test)


# In[36]:


#Performing cross validation
neighbors = []
cv_scores_s = []
#perform 10 fold cross validation
for k in range(1,51,2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors = k)
    scores_s = cross_val_score(knn,Xs_train,y_train,cv=10, scoring = 'accuracy')
    cv_scores_s.append(scores_s.mean())
    
    
#Misclassification error versus k
MSE_s = [1-x for x in cv_scores_s]

#determining the best k
optimal_k_s = neighbors[MSE_s.index(min(MSE_s))]
print('The optimal number of neighbors is: %d ' %optimal_k_s)

#plot misclassification error versus k

plt.figure(figsize = (10,6))
plt.plot(neighbors, MSE_s)
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Misclassification Error')
plt.show()


# In[37]:


#Without Hyper Parameters Tuning
#importing the metrics module
from sklearn import metrics
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#learning
model.fit(Xs_train,y_train)
#Prediction
prediction_s = model.predict(Xs_test)
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction_s,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction_s,y_test))


# In[38]:


#With Hyper Parameters Tuning
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
#making the instance
model = KNeighborsClassifier(n_jobs=-1)
#Hyper Parameters Set
params = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=1)
#Learning
model1.fit(Xs_train,y_train)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction_s = model1.predict(Xs_test)
#evaluation(Accuracy)
print("Accuracy after tuning:",metrics.accuracy_score(prediction_s,y_test))
#evaluation(Confusion Metrix)
print("Confusion Metrix after tuning:\n",metrics.confusion_matrix(prediction_s,y_test))


# In[39]:


# Model Comparison between scaled and not scaled -----------------------------------

print('*'*80)
print('Comparison')

from sklearn.metrics import roc_auc_score, roc_curve

# for un-scaled
knn.fit(X_train, y_train)
pred_prob = knn.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  pred_prob)
auc = roc_auc_score(y_test, pred_prob)
plt.plot(fpr,tpr,label=", auc="+str(np.round(auc,3)))
plt.legend(loc=4)
plt.tight_layout()

# for scaled
knn.fit(Xs_train, y_train)
pred_prob = knn.predict_proba(Xs_test)[::,1]
fpr, tpr, _ = roc_curve(y_test,  pred_prob)
auc_s = roc_auc_score(y_test, pred_prob)
plt.plot(fpr,tpr,label=", auc_s="+str(np.round(auc_s,3)))
plt.legend(loc=4)
plt.ylabel('Sensitivity')
plt.xlabel('1 - Specificity')
plt.title('ROC')
plt.tight_layout()


# In[41]:


#plot misclassification error versus k

dic = {'K':neighbors, 'MSE':[i*100 for i in MSE], 'MSE_s':[j*100 for j in MSE_s]}
df = pd.DataFrame.from_dict(dic)#.transpose()

plt.figure(figsize = (10,6))
plt.plot('K', 'MSE', data=df, marker='o', color='blue', linewidth=2)
plt.plot('K', 'MSE_s', data=df, marker='^', color='red', linewidth=2)
plt.legend()
plt.xlabel('Number of neighbors (K)')
plt.ylabel('Error %')
plt.show()

print('*'*80)
print('Optimal no. of neighbors without scaling: ',optimal_k)
print('Optimal no. of neighbors after scaling: ',optimal_k_s)

print('*'*80)
print('F1 score without scaling:\n', metrics.classification_report(y_test, prediction)) 
print('*'*80)
print('F1 score after scaling:\n', metrics.classification_report(y_test, prediction_s)) 


# In[ ]:




