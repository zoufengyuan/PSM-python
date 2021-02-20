# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 14:17:20 2021

@author: FengY Z
"""

import pandas as pd
import numpy as np
from psm_model_python_R import Psm_R
from tqdm import tqdm
import aidcloudroutine.Analysis as ana
import Linear_Model_update2 as lm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_curve, roc_auc_score,auc,f1_score,recall_score
# In[]
#先进行单因素筛选变量
#再进行lasso回归
#最后进行PSM分析
class Pipeline_Psm(object):
    def __init__(self,data,label,multicate_var,cate_var):
        self.data = data
        self.label = label
        self.vars = list(data.columns)
        self.multicate_var = multicate_var
        self.binary_var = list(set(cate_var)-set(multicate_var))
    def single_factor_filter(self):
        glm = lm.glm_regression(dataframe = source_data[self.vars],class_threshold = 1, label = self.label) 
        base_vars, log, GOF = glm.uni_selection(p = 0.05, link = 'logit')
        log.to_excel('单因素回归结果.xlsx')
        return base_vars, log
    def evaluation(self,model,test_x,test_y):
        predict = model.predict(test_x)
        y_score = model.predict_proba(test_x)
        fpr = dict()
        tpr = dict()
        fpr, tpr, thresholds = roc_curve(test_y, y_score[:,1], pos_label=1)
        model_auc = auc(fpr, tpr)
        model_f1_score = f1_score(test_y, predict)
        recall = recall_score(test_y,predict, average='micro')
        print('predict_auc: %f,predict_f1_score: %f,recall: %f'%(model_auc,model_f1_score,recall))
        csfont = {'fontname':'Times New Roman'}
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.labelweight'] = 'bold'
        plt.figure()
        plt.xlabel('False Positive Rate',**csfont)
        plt.ylabel('True Positive Rate',**csfont)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Classifier ROC',**csfont)
        plt.plot(fpr, tpr, color='blue', lw=2, label='ROC area = %0.3f)' % model_auc)
        plt.legend(loc="lower right")
        plt.show()
        
    def lasso_filter(self):
        base_vars, log = self.single_factor_filter()
        train_x,test_x,train_y,test_y = train_test_split(self.data[base_vars],self.data[self.label],test_size=0.2)
        model_lasso = LogisticRegressionCV(penalty='l1',class_weight = 'balanced',cv = 5,Cs = 20,solver = 'liblinear').fit(train_x, train_y)
        self.evaluation(model_lasso,test_x,test_y)
        
        abs_coef = pd.Series(np.abs(model_lasso.coef_)[0], index = train_x.columns)
        coef = pd.Series(model_lasso.coef_[0], index = train_x.columns).reset_index()
        lasso_imp_coef = abs_coef.sort_values().reset_index()
        coef.rename(columns = {0:'coef'},inplace = True)
        lasso_imp_coef.rename(columns = {0:'abs_coef'},inplace = True)
        lasso_imp_coef = lasso_imp_coef.merge(coef,on = 'index',how = 'left')
        lasso_imp_coef.to_excel('lasso回归特征选择结果.xlsx',index = None)
        lasso_selected_vars = list(lasso_imp_coef['index'][lasso_imp_coef['abs_coef']!=0])
        experimental_vars = list(set(lasso_selected_vars)&set(self.binary_var))
        return lasso_selected_vars,experimental_vars
    
    def psm_model(self):
        lasso_selected_vars,experimental_vars = self.lasso_filter()
        psm_data = self.data[lasso_selected_vars+[self.label]]
        var_num_dic = {var:'X'+str(i+1) for i,var in enumerate(psm_data.columns)}
        num_var_dic = {item[1]:item[0] for item in var_num_dic.items()}
        psm_data.rename(columns = var_num_dic,inplace = True)
        label = var_num_dic[self.label]
        f = pd.ExcelWriter('psm分析结果.xlsx')
        summary_data = pd.DataFrame()
        for var in experimental_vars:
            experiment_var = var_num_dic[var]
            psm = Psm_R(psm_data,label,experiment_var)
            try:
                stat_df = psm.matched_df_distri_stats()
                stat_df['var'].replace(num_var_dic,inplace = True)
                summary_df = psm.summary()
                summary_df['vars'].replace(num_var_dic,inplace = True)
                summary_df.to_excel(f,sheet_name = '{}_回归结果.xlsx'.format(var),index = None)
                tmp_summary = summary_df[summary_df['vars']==var]
                summary_data = summary_data.append(tmp_summary)
            except:
                print(source_data[var].value_counts())
            stat_df.to_excel(f,sheet_name = '{}_数据分布.xlsx'.format(var),index = None)      
        summary_data = summary_data.reset_index(drop = True)
        summary_data.to_excel('psm分析结果汇总.xlsx',index = None)
        f.save()
if __name__ == '__main__':      
    source_data = pd.read_excel('xxx.xlsx')
    label = 'xxx'
    multicate_var = []#多分类变量
    cate_var = []#分类变量
    psm = Pipeline_Psm(source_data,label,multicate_var,cate_var)
    base_vars,log = psm.single_factor_filter()
    psm.psm_model()