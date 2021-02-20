# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 14:33:16 2020

@author: 86156
"""
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.spatial.distance import cdist
from scipy import stats
from causalinference import CausalModel
import warnings
from tqdm import tqdm
import statsmodels.stats.weightstats as sw
from joblib import Parallel, delayed
warnings.filterwarnings('ignore')



class Descriptive_Stat(object):
    def __init__(self,data_all,data,label):
        #label: string type
        #cont_var,cate_var,multicate_var: list
        self.label = label
        self.cont_var,self.multicate_var,self.cate_var = self.vars_classify(data_all)
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

class Psm(Descriptive_Stat):
    
    def __init__(self,target_var,
                 experimental_var,k = 1,
                 threshold = 0.02,
                 cal_type = 'kn',
                 replace = False,
                 match_type = 'nn',
                 common = False):
        '''
        Build a propensity score matching model(PSM) based on logistic regression
        
        Note:It can only realize the analysis of binary variable
        
        Parameters
        ----------
        data: dataframe without Nan,includes independent and dependent variable
        
        target_var: str,the column name of dependent variable
        
        experimental_var: str,the column name of experimental variable with 0 and 1
        
        k: int,the number of neighbors,default 1
        
        threshold: float,the threshold for caliper matching,default 0.02
        
        cal_type: str,the type of the method calculating the similarity(distance) between the controled samples and 
        the treated samples according to the psm score, 'kn' means neighbors without threshold,'cm' means calipers
        with threshold, default 'kn'
        
        replace:bool,whether sampling with replacement,default False
        
        match_type:str,the type of the matched method,'nn' means the method of Nearest neighbor matching,
        'Mahalanobis' means the method of Mahalanobis Distance matching,default 'nn'
        
        common:bool,Whether to choose overlapping individuals,default False
        '''
        
        self.target_var = target_var
        self.experimental_var = experimental_var
        self.k = k
        self.threshold = threshold
        self.cal_type = cal_type
        self.replace = replace
        self.match_type = match_type
        self.common = common
    def cal_ps(self,ps_data):
        '''
        calculate the propensity score based on logistic model
        
        return
        ------------
        ps_data: source data with ps_score for the experimental_var
        '''
        target_series = ps_data.pop(self.target_var)
        experimental_series = ps_data.pop(self.experimental_var)
        model = LogisticRegression(class_weight = 'balanced')
        model.fit(ps_data,experimental_series)
        predict_prob = model.predict_proba(ps_data)
        ps_data[self.target_var] = target_series
        ps_data[self.experimental_var] = experimental_series
        ps_data[self.experimental_var+'_ps_score'] = predict_prob[:,1]
    
    def cal_distance(self,value,ordered_values):
        '''
        Filter the orderd_values to get the k nearest values to the value
        
        Parameters
        -------------
        value: float,the target value
        
        ordered_values: list,the alternative list
        
        return 
        ----------------
        chosed_index: list,the index list selected closest to the value
        '''
        if self.cal_type not in ['kn','cm']:
            raise ValueError("cal_type must be 'kn' or 'cm'," "got %r"%(self.cal_type))
        if self.threshold <= 0:
            raise ValueError("threshold must be a positive number," "got %r"%(self.threshold))
        if self.k<=0:
            raise ValueError("k must be a positive number," "got %r"%(self.k))
        if self.common:
            if value > np.max(ordered_values) or value < np.min(ordered_values):
                return []
        if len(set(ordered_values))==1:
            return []
        diff_value = abs(np.array(ordered_values)-value)
        chosed_index = []
        max_value = np.max(diff_value)
        if self.cal_type == 'kn':
            i = 0
            while i< self.k:
                index = np.argmin(diff_value)
                chosed_index.append(index)
                diff_value[index] = max_value
                i+=1
        elif self.cal_type == 'cm':
            i = 0
            while i<self.k:
                min_value = diff_value.min()
                if min_value>self.threshold:
                    break
                else:
                    index = np.argmin(diff_value)
                    chosed_index.append(index)
                    diff_value[index] = max_value
                    i+=1
        return chosed_index
            
    def nn_match(self,ps_data):
        '''
        the method of Nearest neighbor matching:
        the subjects in the treatment group are sorted reversed, 
        and then starting from the first subject in the treatment group, 
        find an individual whose propensity score is the closest neighbor 
        in the control group as the matching object
        
        return
        ---------
        query_treat: dataframe,matched data
        '''
        self.cal_ps(ps_data)
        query_treat = ps_data[ps_data[self.experimental_var]==1]
        query_control = ps_data[ps_data[self.experimental_var]==0]
        control_list = query_control[self.experimental_var+'_ps_score'].to_list()
        treat_to_control = dict()
        control_to_treat = dict()
        removed = []
        for i in query_treat[self.experimental_var+'_ps_score'].index:
            treat_score = query_treat[self.experimental_var+'_ps_score'].loc[i]
            chosed_index = self.cal_distance(treat_score,control_list)
            if self.cal_type == 'cm' and len(chosed_index)==0:
                removed.append(i)
                continue
            query_treat = query_treat.append(query_control.iloc[chosed_index])
            
            if not self.replace:
                for x in chosed_index:
                    control_list[x] = float('inf') 
            df_index = query_control.iloc[chosed_index].index.to_list()
            for index in df_index:
                if index in control_to_treat:
                    control_to_treat[index].append(i)
                else:
                    control_to_treat[index] = [i]
            treat_to_control[i] = df_index
        query_treat = query_treat.drop(removed)
        del ps_data[self.experimental_var+'_ps_score']
        return query_treat,treat_to_control,control_to_treat
    def Mahalanobis_match(self,data):
        '''
        the method of Mahalanobis Distance matching:
        Apply Mahalanobis distance to find the most similar samples
        
        return
        ----------
        query_treat: dataframe,matched data
        '''
        query_treat = data[data[self.experimental_var]==1]
        query_control = data[data[self.experimental_var]==0]
        distance_var_list = list(query_treat.columns)
        distance_var_list.remove(self.experimental_var)
        X = query_treat[distance_var_list].values
        Y = query_control[distance_var_list].values
        ma_distance = cdist(X,Y,'mahalanobis')
        min_dis = np.argmin(ma_distance,axis = 1)
        df_index = query_control.iloc[min_dis].index.to_list()
        treat_to_control = dict()
        control_to_treat = dict()
        for i,index in enumerate(query_treat.index):
            treat_to_control[index] = [df_index[i]]
            if df_index[i] in control_to_treat:
                control_to_treat[df_index[i]].append(index)
            else:
                control_to_treat[df_index[i]] = [index]
        query_treat = query_treat.append(query_control.iloc[min_dis])
        return query_treat,treat_to_control,control_to_treat
    def matched_df_distri_stats(self,data):
        '''
        get the distribution of matched data and source data
        
        return 
        ------------
        stat_df:dataframe,a dataframe contains the distribution of matched data and source data
        '''
        if self.match_type == 'nn':
            query_treat,treat_to_control,control_to_treat = self.nn_match(data)
        if self.match_type == 'Mahalanobis':
            query_treat,treat_to_control,control_to_treat = self.Mahalanobis_match(data)
        group0 = query_treat[query_treat[self.experimental_var]==0]
        group1 = query_treat[query_treat[self.experimental_var]==1]
        super(Psm,self).__init__(data,query_treat,self.experimental_var)
        f,k = self.Norm_var_test(group0,group1)
        if len(self.cont_var) != 0:
            full_stat = pd.concat([pd.concat([self.cont_stats(query_treat,group0, group1, stats_type = 'mean_std'),k],axis = 1),
                           self.cate_stat(query_treat,group0,group1)])
        else:
            full_stat = self.cate_stat(query_treat,group0,group1)
        all_group0 = data[data[self.experimental_var]==0]
        all_group1 = data[data[self.experimental_var]==1]
        all_f,all_k = self.Norm_var_test(all_group0,all_group1)
        if len(self.cont_var) != 0:
            all_full_stat = pd.concat([pd.concat([self.cont_stats(data,all_group0, all_group1, stats_type = 'mean_std'),all_k],axis = 1),
                           self.cate_stat(data,all_group0,all_group1)])
        else:
            all_full_stat = self.cate_stat(data,all_group0,all_group1)
        all_full_stat.rename(columns = {'All':'All_All','Group 0':'Group 0 All','Group 1':'Group 1 All','P_value':'P_value All'},inplace = True)
        stat_df = pd.concat([all_full_stat,full_stat],axis = 1)
        stat_df = stat_df.drop([self.target_var])
        stat_df = stat_df.reset_index()
        stat_df.rename(columns = {'index':'var'},inplace = True)
        return stat_df
   
    def summary(self,data):
        '''
        get summary of the psm model
        
        Parameters
        ----------
        
        return 
        ------------
        summary_df: dataframe, a dataframe of model summary contains coef and pvalues 
        '''
        if self.match_type not in ['nn','Mahalanobis']:
            raise ValueError("cal_type must be 'nn' or 'Mahalanobis'," "got %r"%(self.match_type))
        
        if self.match_type == 'nn':
            query_treat,treat_to_control,control_to_treat = self.nn_match(data)
            y = query_treat.pop(self.target_var)
            query_treat.pop(self.experimental_var+'_ps_score')
            model = sm.Logit(y,pd.DataFrame(query_treat)).fit()
        elif self.match_type == 'Mahalanobis':
            query_treat,treat_to_control,control_to_treat = self.Mahalanobis_match(data)
            y = query_treat.pop(self.target_var)
            model = sm.Logit(y,pd.DataFrame(query_treat)).fit()
        summary = model.summary()
        print(summary)
        coef = model.params.reset_index()
        coef.rename(columns = {'index':'vars',0:'coef'},inplace = True)
        pvalues = model.pvalues.reset_index()
        pvalues.rename(columns = {'index':'vars',0:'pvalues'},inplace = True)
        coef['exp(coef)'] = np.exp(coef['coef'])
        summary_df = pd.merge(coef,pvalues,on = 'vars',how = 'left')
        return summary_df
    
    def cal_ATT(self,query_treat,treat_to_control):
        self.n_1 = 0
        self.sums_1 = 0
        for key in treat_to_control:
            y1_i = query_treat[self.target_var].loc[key]
            y0_i_hat_df = query_treat[self.target_var].loc[treat_to_control[key]].reset_index().drop_duplicates()
            y0_i_hat = y0_i_hat_df[self.target_var].values[0]
            dis = y1_i - y0_i_hat
            self.n_1 +=1
            self.sums_1+=dis
        return self.sums_1/self.n_1
    
    def cal_ATU(self,query_treat,control_to_treat):
        self.n_0 = 0
        self.sums_0 = 0
        for key in control_to_treat:
            if len(control_to_treat[key])==1:
                y0_i = query_treat[self.target_var].loc[key]
                y1_i_hat = query_treat[self.target_var].loc[control_to_treat[key][0]]
            else:
                y0_i = query_treat[self.target_var].loc[key].values[0] 
                y1_i_hat = np.mean(query_treat[self.target_var].loc[control_to_treat[key]])
            dis = y1_i_hat-y0_i
            self.n_0 +=1
            self.sums_0 += dis
        return self.sums_0/self.n_0
    
    def cal_ATE(self,query_treat,treat_to_control,control_to_treat):

        return (self.sums_1+self.sums_0)/(self.n_1+self.n_0)
        

class Evaluate(Psm):
    def __init__(self,data,target_var,
                 experimental_var,k = 1,
                 threshold = 0.02,
                 cal_type = 'kn',
                 replace = False,
                 match_type = 'nn',
                 common = False,
                 epoch = 100):
        #super(Evaluate,self).__init__(target_var,experimental_var,k,threshold,cal_type,replace,match_type,common)        
        self.epoch = epoch
        self.data = data
        self.target_var = target_var
        self.experimental_var = experimental_var
        self.k = k
        self.threshold = threshold
        self.cal_type = cal_type
        self.replace = replace
        self.match_type = match_type
        self.common = common
        
    
    def boostrap(self):
        n = len(self.data)
        boostrap_data = self.data.sample(n = n,replace = True).reset_index()
        del boostrap_data['index']
        return boostrap_data
    
    def cal_index(self):
        boostrap_data = self.boostrap()
        query_treat,treat_to_control,control_to_treat = self.nn_match(boostrap_data)
        ATT = self.cal_ATT(query_treat,treat_to_control)
        ATU = self.cal_ATU(query_treat,control_to_treat)
        ATE = self.cal_ATE(query_treat,treat_to_control,control_to_treat)
        return [ATT,ATU,ATE]
        
    def parallel(self):
        result = Parallel(n_jobs=-1, backend='multiprocessing',verbose = 1)(delayed(self.cal_index)() for i in tqdm(range(self.epoch)))
        return result
    
    def cal_df(self):
        result = self.parallel()
        tmp = list(zip(*result))
        df = pd.DataFrame({'ATT':tmp[0],'ATU':tmp[1],'ATE':tmp[2]})
        return df
    
#Example:
    
if __name__ == '__main__':
    psm_data = pd.read_excel('psm_test.xlsx')
    target_var = '第1次_呼吸系统_围术期肺部并发症_PPCs_术后并发症名称_new'     
    experimental_var = '第1次_是否术前激素使用'
    psm_model =  Psm(target_var,experimental_var,k = 1,replace = True,match_type = 'Mahalanobis')
    query_treat,treat_to_control,control_to_treat = psm_model.nn_match(psm_data)
    psm_summary = psm_model.summary(psm_data)
    psm_stats_df = psm_model.matched_df_distri_stats(psm_data)
    ATT = psm_model.cal_ATT(query_treat,treat_to_control)
    ATU = psm_model.cal_ATU(query_treat,control_to_treat)
    ATE = psm_model.cal_ATE(query_treat,treat_to_control,control_to_treat)
    print(ATT,ATU,ATE)
    

'''
psm_summary:
    
Optimization terminated successfully.
         Current function value: 0.203118
         Iterations 7
                                   Logit Regression Results                                   
==============================================================================================
Dep. Variable:     第1次_呼吸系统_围术期肺部并发症_PPCs_术后并发症名称_new   No. Observations:                 1810
Model:                                          Logit   Df Residuals:                     1802
Method:                                           MLE   Df Model:                            7
Date:                                Wed, 30 Dec 2020   Pseudo R-squ.:                 0.06984
Time:                                        09:42:51   Log-Likelihood:                -367.64
converged:                                       True   LL-Null:                       -395.25
Covariance Type:                            nonrobust   LLR p-value:                 1.358e-09
==============================================================================================
                                 coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------------------
const                         -4.4885      0.656     -6.845      0.000      -5.774      -3.203
第1次_术前气管插管                     1.6342      0.482      3.391      0.001       0.690       2.579
第1次_生化_谷草转氨酶AST_术前             0.3315      0.118      2.807      0.005       0.100       0.563
第1次_麻醉时长_min_NA表示未获得数据_new     0.2885      0.115      2.515      0.012       0.064       0.513
第1次_是否术前激素使用                   0.8497      0.358      2.376      0.017       0.149       1.551
第1次_血常规_红细胞总数RBC_术前           -0.3860      0.126     -3.067      0.002      -0.633      -0.139
第1次_血常规_超敏C反应蛋白hsCRP_术前        0.2263      0.127      1.781      0.075      -0.023       0.475
第1次_是否术前使用抗生素                  0.5132      0.214      2.401      0.016       0.094       0.932
==============================================================================================

psm_test:
Treatment Effect Estimates: Matching

                     Est.       S.e.          z      P>|z|      [95% Conf. int.]
--------------------------------------------------------------------------------
           ATE      0.026      0.012      2.094      0.036      0.002      0.051
           ATC      0.026      0.013      1.931      0.053     -0.000      0.052
           ATT      0.027      0.013      2.098      0.036      0.002      0.052

'''   


        

        
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        