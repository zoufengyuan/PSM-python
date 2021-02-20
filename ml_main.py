# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:26:18 2021

@author: FengY Z
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import Linear_Model as lm
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score,auc,f1_score,recall_score

# In[]
#应用一些机器学习的方法进行结果对比
#首先应用单因素回归筛选一遍变量
#然后构建logistic、决策树、随机森林、gbdt、SVM、贝叶斯等模型
#应用boostrap得到auc置信区间，并用统计检验方法进行对比
class Pipeline_ML(object):
    def __init__(self,data,label,n):
        self.data = data
        self.label = label
        self.vars = list(data.columns)
        self.base_vars,_ = self.single_factor_filter()
        self.n = n
    def single_factor_filter(self):
        glm = lm.glm_regression(dataframe = source_data[self.vars],class_threshold = 1, label = self.label) 
        base_vars, log, GOF = glm.uni_selection(p = 0.05, link = 'logit')
        return base_vars, log
    def evaluation(self,model,test_x,test_y):
        predict = model.predict(test_x)
        y_score = model.predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, y_score[:,1], pos_label=1)
        model_auc = auc(fpr, tpr)
        model_f1_score = f1_score(test_y, predict)
        recall = recall_score(test_y,predict, average='micro')
        return model_auc,model_f1_score,recall
    def single_samples_model(self,train_x,train_y,test_x,test_y):
        logistic_model = LogisticRegression(penalty = 'l2',class_weight = 'balanced')
        dt_model = DecisionTreeClassifier(class_weight = 'balanced',max_depth = 4,min_samples_leaf = 2,min_samples_split = 7)
        rf_model = RandomForestClassifier(class_weight = 'balanced',n_estimators = 40,max_depth = 6,min_samples_leaf = 4,min_samples_split = 5)
        gbdt_model = GradientBoostingClassifier(n_estimators = 60,learning_rate = 0.05,max_depth = 2,min_samples_leaf = 2,min_samples_split = 5)
        nb_model = GaussianNB()
        model_dic = {'logistic':logistic_model,'dt':dt_model,'rf':rf_model,'gbdt':gbdt_model,'nb':nb_model}
        train_auc_dic = {}
        test_auc_dic = {}
        for model in model_dic:
            model_dic[model].fit(train_x,train_y)
            train_auc,_,_ = self.evaluation(model_dic[model],train_x,train_y)
            test_auc,_,_ = self.evaluation(model_dic[model],test_x,test_y)
            train_auc_dic[model] = train_auc
            test_auc_dic[model] = test_auc
        return train_auc_dic,test_auc_dic
    def booststrap(self):
        train_auc_dic_all = {}
        test_auc_dic_all = {}
        selected_data = self.data[self.base_vars+[self.label]]
        for i in tqdm(range(self.n)):
            train_x = selected_data.sample(frac = 0.8,replace = False)
            test_x = selected_data.drop(train_x.index)
            train_y = train_x.pop(self.label)
            test_y = test_x.pop(self.label)
            train_auc_dic,test_auc_dic = self.single_samples_model(train_x,train_y,test_x,test_y)
            for key in train_auc_dic:
                if key in train_auc_dic_all:
                    train_auc_dic_all[key].append(train_auc_dic[key])
                else:
                    train_auc_dic_all[key] = [train_auc_dic[key]]
            for key in test_auc_dic:
                if key in test_auc_dic_all:
                    test_auc_dic_all[key].append(test_auc_dic[key])
                else:
                    test_auc_dic_all[key] = [test_auc_dic[key]]
        return train_auc_dic_all,test_auc_dic_all
    def get_best_parm(self):
        train_x = self.data[self.base_vars+[self.label]]
        train_y = train_x.pop(self.label)
        parm_dic = {}
        score_dic = {}
        dt_parms = {'max_depth':[2,3,4,5,6],'min_samples_split':[5,6,7,8,9],'min_samples_leaf':[2,3,4,5,6]}
        dt_model = DecisionTreeClassifier(class_weight = 'balanced')
        dt_search = GridSearchCV(dt_model,dt_parms,cv = 5)
        rf_parms = {'n_estimators':[30,40,50,60],'max_depth':[2,3,4,5,6],'min_samples_split':[5,6,7,8,9],'min_samples_leaf':[2,3,4,5,6]}
        rf_model = RandomForestClassifier(class_weight = 'balanced')
        rf_search = GridSearchCV(rf_model,rf_parms,cv = 5)
        gbdt_parms = {'n_estimators':[30,40,50,60],'learning_rate':[0.01,0.05,0.1],'max_depth':[2,3,4,5,6],'min_samples_split':[5,6,7,8,9],'min_samples_leaf':[2,3,4,5,6]}
        gbdt_model = GradientBoostingClassifier()
        gbdt_search = GridSearchCV(gbdt_model,gbdt_parms,cv = 5)
        search_dic = {'dt':dt_search,'rf':rf_search,'gbdt':gbdt_search}
        for key in tqdm(search_dic):
            search_dic[key].fit(train_x,train_y)
            parm_dic[key] = search_dic[key].best_params_
            score_dic[key] = search_dic[key].best_score_
        return parm_dic,score_dic
    def auc_t_test(self,auc_dic_all):
        model_list = list(auc_dic_all.keys())
        test_result = pd.DataFrame(np.zeros((len(model_list),len(model_list))),columns = model_list,index = model_list)
        for base_key in model_list:
            for compared_key in model_list:
                if base_key == compared_key:
                    test_result.loc[base_key,compared_key]=1.0
                else:
                    p_value = ks_2samp(auc_dic_all[base_key],auc_dic_all[compared_key])[1]
                    test_result.loc[base_key,compared_key]=round(p_value,4)
        return test_result
    def main(self):
        train_auc_dic_all,test_auc_dic_all = self.booststrap()
        train_exam_result = self.auc_t_test(train_auc_dic_all)
        test_exam_result = self.auc_t_test(test_auc_dic_all)
        train_percentile_result = {}
        test_percentile_result = {}
        for key in train_auc_dic_all:
            median = np.percentile(train_auc_dic_all[key],50)
            low_percent = np.percentile(train_auc_dic_all[key],2.5)
            high_percent = np.percentile(train_auc_dic_all[key],97.5)
            train_percentile_result[key] = '{}({}-{})'.format(round(median,3),round(low_percent,3),round(high_percent,3))
        for key in test_auc_dic_all:
            median = np.percentile(test_auc_dic_all[key],50)
            low_percent = np.percentile(test_auc_dic_all[key],2.5)
            high_percent = np.percentile(test_auc_dic_all[key],97.5)
            test_percentile_result[key] = '{}({}-{})'.format(round(median,3),round(low_percent,3),round(high_percent,3))
        return train_percentile_result,test_percentile_result,train_exam_result,test_exam_result
    
if __name__ == '__main__':
    source_data = pd.read_excel('xxx.xlsx')
    label = 'ARDS'
    ml_ensemble = Pipeline_ML(source_data,label,1000) 
    #parm_dic,score_dic = ml_ensemble.get_best_parm()
    train_percentile_result,test_percentile_result,train_exam_result,test_exam_result = ml_ensemble.main() 
    print(train_percentile_result)        
    print(test_percentile_result)      