# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 18:37:30 2021

@author: 86156
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy import stats
#from causalinference import CausalModel
import warnings
from tqdm import tqdm
import statsmodels.stats.weightstats as sw
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
warnings.filterwarnings('ignore')
pandas2ri.activate()

class Descriptive_Stat(object):
    def __init__(self,data_all,data,label):
        #label: string type
        #cont_var,cate_var,multicate_var: list
        self.label = label
        self.cont_var,self.multicate_var,self.cate_var = self.vars_classify(data_all)
        for var in self.multicate_var:
            self.cont_var.append(var)
            self.cate_var.remove(var)
        self.multicate_var = []
        self.binary_var = [i for i in self.cate_var if i not in self.multicate_var]
    def vars_classify(self,data):
        cont_var = []
        multicate_var = []
        vars_list = list(data.columns)
        vars_list.remove(self.label)
        for var in vars_list:
            unique = data[var].dropna().unique()
            nunique_n = data[var].nunique()
            if isinstance(unique[0],str):
                multicate_var.append(var)
                continue
            if nunique_n < 10 and nunique_n > 2:
                multicate_var.append(var)
            elif nunique_n>=10:
                cont_var.append(var)
        cate_var = list(set(vars_list)-set(cont_var))
        return cont_var,multicate_var,cate_var
         
    def label_dist(self,data):
        distribution = data[self.label].value_counts()
        return(distribution)
    
    def encap_mean_std(self, x):
        """
        Input: An array or a series
        Output: format: Mean(SD)
        """
        out = str(round(x['mean'],2)) + '(' + str(round(x['std'],2)) + ')'
        return(out)

    def encap_percentiles(self, x):
        """
        Input: An array or a series
        Output: format: median[25%,75%]
        """
        out = str(x['50%'])+'('+str(x['25%']) + ',' + str(x['75%']) + ')'
        return(out) 
    def Norm_var_test(self, group0, group1, alpha = 0.05):
        """
        Carry out Shapiro Test and Levene's Test on continuous variables.
        If a variable passes both tests, One_Way_F_test will be performed,
        otherwise, Kruskal test will be performed. 
        Input:
            group0: Dataframe for group 0
            group1: Dataframe for group 1
            alpha: Significance level
        Output:
            f_test: Dataframe(Feature: F test p value)
            kru_test: Dataframe(Feature: Kruskal test p value)
        """
        f_test = {}
        kru_test = {}
        for i in self.cont_var:
            try:
                g0 = group0[i].dropna()
                g1 = group1[i].dropna()
                #Perform Shapirio Test on each group
                sha_p0 = stats.shapiro(g0)[1]
                sha_p1 = stats.shapiro(g1)[1]
                #Perfor Levene Test
                L_p = stats.levene(g0,g1)[1]
                #H0 is valid if p_value greater than alpha
                if (sha_p0 > alpha and sha_p1 > alpha and L_p > alpha):
                    f_test[i] = round(stats.f_oneway(g0,g1)[1],3)
                else:
                    kru_test[i] = round(stats.kruskal(g0,g1)[1],3)
            except TypeError:
                print('TypeError: Feature ',i)
                break
        f_test = pd.DataFrame({'P_value':f_test})
        kru_test = pd.DataFrame({'P_value':kru_test})
        return(f_test, kru_test)
        
    def chi2_test(self,x,y):
        """
        Input:
            x: Categorical variables
            y: Label
        Output:
            p: Chi square p value
        """
        p = np.round(stats.chi2_contingency(pd.crosstab(x,y))[1],3)
        return(p)
    
    def encap_percent(self,x,sample_size,decimal = 2):
        """
        Input: An array or a series
        Output: format: count(percentage)
        """
        out = str(x) + '(' + str(np.round((x*100)/sample_size,decimal)) + '%' + ')'
        return(out)
        
    def cont_stats(self,data,group0, group1, stats_type = 'mean_std', decimal = 2):
        """
        Create statistical table including mean, std, and p_value for each case
        Input:
            df: The whole dataframe
            group0: Group 0 of df
            group1: Group 1 of df
            stats_type: String type, {'mean_std','percentiles'}
        Output:
            cont_stat: A dataframe including mean, std, and p_value for each case
        """
        desc_all = np.round(data[self.cont_var].describe(),decimal)
        desc_group0 = np.round(group0[self.cont_var].describe(),decimal)
        desc_group1 = np.round(group1[self.cont_var].describe(),decimal)
        if stats_type == 'mean_std':
            overall_stat = pd.DataFrame(desc_all.apply(lambda x: self.encap_mean_std(x)),columns = ['All'])
            g0_stat = pd.DataFrame(desc_group0.apply(lambda x: self.encap_mean_std(x)),columns = ['Group 0'])
            g1_stat = pd.DataFrame(desc_group1.apply(lambda x: self.encap_mean_std(x)),columns = ['Group 1'])
            cont_stat = pd.concat([overall_stat,g0_stat,g1_stat],axis = 1)
        elif stats_type == 'percentiles':
            overall_stat = pd.DataFrame(desc_all.apply(lambda x: self.encap_percentiles(x)),columns = ['All'])
            g0_stat = pd.DataFrame(desc_group0.apply(lambda x: self.encap_percentiles(x)),columns = ['Group 0'])
            g1_stat = pd.DataFrame(desc_group1.apply(lambda x: self.encap_percentiles(x)),columns = ['Group 1'])
            cont_stat = pd.concat([overall_stat,g0_stat,g1_stat],axis = 1)
        return(cont_stat)
    
    def cate_stat(self,data,group0, group1, decimal=2):
        """
        Input:
            df:Whole dataset
            group0: Group 0 of df
            group1: Group 1 of df
            decimal: Decimal precision
        Output:
            Statistical analysis of all categorical variables,
            corresponding p values are also presented.
            Binary variables and multi-category variables are included
        """
        distribution = self.label_dist(data)
        # Binary statistics
        binary_cate = self.binary_var
        binary_all_descr = pd.DataFrame(np.sum(data[binary_cate]),columns = ['All'])
        binary_g0_descr = pd.DataFrame(np.sum(group0[binary_cate]),columns = ['Group 0'])
        binary_g1_descr = pd.DataFrame(np.sum(group1[binary_cate]),columns = ['Group 1'])
        binary_cate_test = pd.DataFrame(data[binary_cate].apply(lambda x:self.chi2_test(x,data[self.label])),
                                        columns = ['P_value'])
        binary_stats = pd.concat([binary_all_descr, binary_g0_descr,binary_g1_descr,binary_cate_test],axis = 1)
        # Multi-category statistics
        if len(self.multicate_var) != 0:
            multi_cate_test = pd.DataFrame(data[self.multicate_var].apply(lambda x:self.chi2_test(x,data[self.label]))
                                           ,columns = ['P_value'])
            cate_descr_full = pd.DataFrame()
            for i in self.multicate_var:
                cate_descr_header = pd.DataFrame([['']*3],columns = ['All','Group 0','Group 1'], index = [i])   
                cate_descr_body  = pd.concat([data.groupby(i)[i].size(),group0.groupby(i).size(),group1.groupby(i).size()],axis = 1)
                cate_descr_body.columns = ['All','Group 0','Group 1']
                cate_descr_body = cate_descr_body.fillna(0)
                cate_descr_full = cate_descr_full.append(pd.concat([cate_descr_header,cate_descr_body],axis = 0))
            cate_descr_full['P_value'] = np.NaN
            cate_descr_full.loc[multi_cate_test.index,'P_value'] = multi_cate_test.loc[multi_cate_test.index,'P_value']
            cate_descr_full.fillna('',inplace = True)
            full_cate_stats = binary_stats.append(cate_descr_full)
        else:
            full_cate_stats = binary_stats
        
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'All'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'All'].apply(lambda x: self.encap_percent(x,data.shape[0]))
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 0'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 0'].apply(lambda x: self.encap_percent(x,distribution[0]))
        full_cate_stats.loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 1'] = full_cate_stats.\
            loc[full_cate_stats[full_cate_stats.All != ''].index,'Group 1'].apply(lambda x: self.encap_percent(x,distribution[1]))
        return(full_cate_stats)
class Psm_R(Descriptive_Stat):
    def __init__(self,data,target_var,experimental_var):
        self.data = data
        self.target_var = target_var
        self.experimental_var = experimental_var
    def gen_matched_data(self):
        covars = list(self.data.columns)
        covars.remove(self.target_var)
        covars.remove(self.experimental_var)
        with localconverter(robjects.default_converter + pandas2ri.converter):
            r_data = robjects.conversion.py2rpy(self.data)
            covar_vars = robjects.FactorVector(covars)
            experiment = robjects.FactorVector([self.experimental_var])
            robjects.r('''
                       library(MatchIt)
                       psm = function(r_data, experiment,covar_vars) {
                        xnam <- paste0(covar_vars)
                        ynam <- paste0(experiment)
                        f <- as.formula(paste(paste(ynam),"~",paste(xnam, collapse = "+")))
                           mod_match <- matchit(f,data = r_data,ratio = 1,method = "nearest",caliper = 0.01)
                           data_m <- match.data(mod_match)
                           return(data_m)
                           }
                       ''')
        match_data = robjects.r.psm(r_data,experiment,covar_vars)
        return match_data

    def matched_df_distri_stats(self):
        '''
        get the distribution of matched data and source data
        
        return 
        ------------
        stat_df:dataframe,a dataframe contains the distribution of matched data and source data
        '''
        query_treat = self.gen_matched_data()
        query_treat = query_treat.drop(['distance', 'weights','subclass'],axis = 1)
        super(Psm_R,self).__init__(self.data,query_treat,self.experimental_var)
        group0 = query_treat[query_treat[self.experimental_var]==0]
        group1 = query_treat[query_treat[self.experimental_var]==1]
        print(self.cont_var)
        f,k = self.Norm_var_test(group0,group1)
        if len(self.cont_var) != 0:
            full_stat = pd.concat([pd.concat([self.cont_stats(query_treat,group0, group1, stats_type = 'mean_std'),k],axis = 1),
                           self.cate_stat(query_treat,group0,group1)])
        else:
            full_stat = self.cate_stat(query_treat,group0,group1)
        all_group0 = self.data[self.data[self.experimental_var]==0]
        all_group1 = self.data[self.data[self.experimental_var]==1]
        all_f,all_k = self.Norm_var_test(all_group0,all_group1)
        if len(self.cont_var) != 0:
            all_full_stat = pd.concat([pd.concat([self.cont_stats(self.data,all_group0, all_group1, stats_type = 'mean_std'),all_k],axis = 1),
                           self.cate_stat(self.data,all_group0,all_group1)])
        else:
            all_full_stat = self.cate_stat(self.data,all_group0,all_group1)
        all_full_stat.rename(columns = {'All':'All_All','Group 0':'Group 0 All','Group 1':'Group 1 All','P_value':'P_value All'},inplace = True)
        stat_df = pd.concat([all_full_stat,full_stat],axis = 1)
        stat_df = stat_df.drop([self.target_var])
        stat_df = stat_df.reset_index()
        stat_df.rename(columns = {'index':'var'},inplace = True)
        return stat_df
   
    def summary(self):
        '''
        get summary of the psm model
        
        Parameters
        ----------
        
        return 
        ------------
        summary_df: dataframe, a dataframe of model summary contains coef and pvalues 
        '''
        query_treat = self.gen_matched_data()
        query_treat = query_treat.drop(['distance', 'weights','subclass'],axis = 1)
        y = query_treat.pop(self.target_var)
        model = sm.Logit(y,query_treat).fit()
        summary = model.summary()
        coef = model.params.reset_index()
        coef.rename(columns = {'index':'vars',0:'coef'},inplace = True)
        pvalues = model.pvalues.reset_index()
        pvalues.rename(columns = {'index':'vars',0:'pvalues'},inplace = True)
        coef['exp(coef)'] = np.exp(coef['coef'])
        summary_df = pd.merge(coef,pvalues,on = 'vars',how = 'left')
        return summary_df

if __name__ == '__main__':
    ecls = pd.read_csv(r"ecls.csv").dropna(axis=0,how='any')
    experiment_var = 'catholic'
    target_var = 'race_white'
    covars = ['race_white' , 'w3income', 'p5hmage', 'p5numpla', 'w3momed_hsb']
    covars.append(experiment_var)
    data = ecls[covars]
    psm = Psm_R(data,target_var,experiment_var)
    match_data = psm.gen_matched_data()
    stat_df = psm.matched_df_distri_stats()


























