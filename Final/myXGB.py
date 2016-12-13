#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xgboost as xgb
import numpy as np
import myparse as mp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_validation import StratifiedKFold

# read csv include first line
enroll_train = mp.readcsv("enrollment_train.csv")
truth_train = mp.readcsv("truth_train.csv")
sample_train_x = mp.readcsv("sample_train_x.csv")
sample_test_x = mp.readcsv("sample_test_x.csv")
aug_graph_train = mp.readcsv("augmentGraph_train.csv")
aug_graph_test =mp.readcsv("augmentGraph_test.csv")

aug_feat_train = mp.readcsv("feat.csv")

aug_train = aug_graph_train[1:,1:].astype(float)
data_train = sample_train_x[1:,1:].astype(float)
aug_feat = aug_feat_train[0:,1:].astype(float)

data_train = np.hstack((data_train,aug_train))
data_train = np.hstack((data_train,aug_feat))
print np.shape(data_train)
label_train = truth_train[0:,1].astype(float)

aug_test = aug_graph_test[1:,1:].astype(float)
data_test = sample_test_x[1:,1:].astype(float)
data_test = np.hstack((data_test,aug_test))

#Pre-Processing
preprocess = StandardScaler()
#preprocess = RobustScaler()

data_train = preprocess.fit_transform(data_train)
data_test = preprocess.fit_transform(data_test)

pca =PCA(n_components=40)
data_train = pca.fit_transform(data_train)

param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 10
plst = param.items()


skf = StratifiedKFold(label_train, n_folds=5)
#CV ing
corr=np.array([])
print ('running cross validation')
for train_index, test_index in skf:
    dtrain = xgb.DMatrix(data_train[train_index], label=label_train[train_index])
    dtest = xgb.DMatrix(data_train[test_index],label=label_train[test_index])
    watchlist = [(dtest,'eval'),(dtrain,'train')]
    bst = xgb.train(param,dtrain,num_round,watchlist)
    preds = bst.predict(dtest)
    labels = dtest.get_label()
    total = np.shape(labels)[0]
    preds[preds>0.5]=1
    preds[preds<=0.5]=0
    cor =  (total-sum(abs(preds-labels)))/total
    print 'Corr = %0.6f' % cor
    corr=np.append(corr,cor)
    #print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
print 'Mean Precision is %0.6f' %np.mean(corr) 

'''
dtrain = xgb.DMatrix(data_train,  label=label_train)
param = {'max_depth':3, 'eta':1, 'silent':1, 'objective':'binary:logistic'}
num_round = 10

bst = xgb.train(param,dtrain,num_round)
'''

'''
dummy = np.zeros(np.shape(data_test)[0])
dtest = xgb.DMatrix(data_test,label= dummy)
pred_test = bst.predict(dtest)
#pred_test[pred_test>0.5]=1
#pred_test[pred_test<=0.5]=0
print np.shape(pred_test)

f = open('XGBreglog.csv','wb')
for i in range(0,len(pred_test)):
    f.write(str(sample_test_x[i+1,0])+','+str(pred_test[i])+'\n')
'''
'''
print ('running cross validation, with cutomsized loss function')
def logregobj(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0-preds)
    return grad, hess
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'error', float(sum(labels != (preds > 0.0))) / len(labels)

param = {'max_depth':2, 'eta':1, 'silent':1}
# train with customized objective
xgb.cv(param, dtrain, num_round, nfold = 5, seed = 0,
       obj = logregobj, feval=evalerror)
'''