# -*- coding: utf-8 -*-
"""
Created on Thu Sep 09 18:52:54 2020

@author: Gautam kumar
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import datetime
from sklearn.metrics import confusion_matrix

#importing the data 
traindata=pd.read_csv('train.csv')
#print(traindata.head())
print(traindata.isna().sum())
print(traindata.info())
traindata['Age']=traindata['Age'].fillna(traindata['Age'].mean())
traindata['Embarked']=traindata['Embarked'].fillna(traindata['Embarked'].value_counts().index[0])
traindata=traindata.drop(['Cabin','Name','Ticket'],axis=1)
traindata['Sex']=pd.get_dummies(traindata['Sex'])
traindata['Embarked']=pd.get_dummies(traindata['Embarked'])
x=traindata.drop('Survived',axis=1)
y=traindata.Survived
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.ensemble import RandomForestClassifier
RFC=RandomForestClassifier(n_estimators=100)
RFCModel=RFC.fit(x_train,y_train)

pickle.dump(RFCModel, open('trainingpklmodel.pkl','wb'))
model = pickle.load(open('trainingpklmodel.pkl','rb'))

print('training has been completed and  its dump into pkl files')

#predicted=RFCModel.predict(x_test)
#print(predicted)
#cm=confusion_matrix(predicted,y_test)
#accuracy=(cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
#print(cm)
#print(accuracy)
