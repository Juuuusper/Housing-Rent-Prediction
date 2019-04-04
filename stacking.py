# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 15:25:05 2018

@author: Administrator
"""

from heamy.dataset import Dataset
from heamy.estimator import  Classifier
from heamy.pipeline import ModelsPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from xgboost import XGBClassifier


#创建数据集
dataset = Dataset(X_train,y_train,X_n)#对无标签训练集进行预测时将X_test替换为X_n


model_xgb = Classifier(dataset=dataset, estimator=XGBClassifier,parameters={
                                                                              'reg_alpha':0.01,
                                                                              'n_estimators':100,
                                                                              'objective':'binary:logistic',
                                                                               'seed': 32,
                                                                              'gamma':0.4,
                                                                              'colsample_bytree':0.75,
                                                                              'subsample':0.8,}, name='xgb')

model_xgb2 = Classifier(dataset=dataset, estimator=XGBClassifier,parameters={
                                                                              'seed': 128,
                                                                              'gamma':0.4,
                                                                              'reg_alpha':0.01,
                                                                              'n_estimators':100,
                                                                              'objective':'binary:logistic',
                                                                               'colsample_bytree':0.75,
                                                                              'subsample':0.8,}, name='xgb')


model_lgb = Classifier(dataset=dataset, estimator=lgb.LGBMClassifier,parameters={
                                                                                'reg_lambda':0.002,
                                                                                'max_depth':6,
                                                                                'min_child_weight':0.001,
                                                                                'num_leaves':30,
                                                                                'seed':32,
                                                                                'n_estimators':70,
                                                                                'boosting_type':'gbdt',
                                                                                'reg_alpha':0.001,
                                                                                'colsample_bytree':0.5,
                                                                                'min_child_samples':24,}, name='lgb')

model_lgb2 = Classifier(dataset=dataset, estimator=lgb.LGBMClassifier,parameters={'n_estimators':70,
                                                                                'boosting_type':'gbdt',
                                                                                'max_depth':6,
                                                                                'min_child_weight':0.001,
                                                                                'num_leaves':30,
                                                                                'seed':128,
                                                                                'reg_alpha':0.001,
                                                                                'reg_lambda':0.002,
                                                                                'colsample_bytree':0.5,
                                                                                'min_child_samples':24}, name='lgb')

model_lg = Classifier(dataset=dataset, estimator=LogisticRegression,name='lg')

pipeline = ModelsPipeline(model_lgb,model_lgb2,model_xgb,model_xgb2,model_lg)
stack_ds = pipeline.stack(k=10,seed=111)
stacker = Classifier(dataset=stack_ds, estimator=LogisticRegression)
stacker.validate(k=10,scorer=roc_auc_score)
results = stacker.predict()
