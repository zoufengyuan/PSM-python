# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 14:30:45 2020

@author: MZH
"""
import statsmodels.api as sm
import pandas as pd
import scipy
import numpy as np

"""
packages version
statsmodels: 0.12.0
numpy: 1.19.1
pandas: 1.1.3
"""
class glm_regression():
    def __init__(self, dataframe, label, class_threshold = 10, multicate_var = None):
        if multicate_var == None:
            # num_class = dataframe.apply(lambda x:len(pd.unique(x)))
            num_class= dataframe.nunique(axis = 0)
            self.multicate_var = list(num_class.index[((num_class>2)&(num_class<class_threshold))])
        else:
            self.multicate_var = multicate_var
        self.label = label
        self.columns = dataframe.columns.tolist()
        self.df = dataframe
    
    def r_square(self, model, dataframe, X):
        corr_matrix = np.corrcoef(dataframe[self.label],model.predict(dataframe[X]))
        corr = corr_matrix[0,1]
        r_square = corr**2
        return r_square
         
    def HL_test(self,model,dataframe,X,g=10):
        """
        Ref:
           https://www.statisticshowto.com/hosmer-lemeshow-test/
        ----------
        Parameters
        ----------
        model : Logistic model
        dataframe
        X : Independent variables
        g : Number of group in HL test. The default is 10.
        Returns
        -------
        statistics: chi square statistics
        p: p value
        degree_freedom: degrees of freedom
        """
        new_df = dataframe[X + [self.label]].copy()
        new_df['proba'] = model.predict(new_df)
        new_df['proba_group'] = pd.qcut(new_df['proba'],g,duplicates = 'drop')
        new_g = len(new_df.proba_group.unique())
        one_mean = new_df.groupby('proba_group')['proba'].mean()
        zero_mean = 1 - one_mean
        expect_ones = (new_df.groupby('proba_group')[self.label].count()*one_mean).values
        expect_zeros = (new_df.groupby('proba_group')[self.label].count()*zero_mean).values
        expect = np.concatenate((expect_zeros.reshape(new_g,1), expect_ones.reshape(new_g,1)),axis = 1)
        obs = new_df.groupby(['proba_group',self.label]).size()
        obs = obs.values.reshape(-1,2)
        degree_freedom = new_g-2
        statistics = np.sum((obs - expect)**2 / expect)
        p = 1 - scipy.stats.chi2.cdf(statistics, degree_freedom)
        return((statistics, p, degree_freedom))


    def LRT(self, restricted_model, full_model):
        """
        Description:
            Likelihood ratio test p value
            see:https://en.wikipedia.org/wiki/Likelihood-ratio_test
        """
        llf_full = full_model.llf #value log likelihood of full model
        llf_restr = restricted_model.llf
        df_full = full_model.df_resid 
        df_restr = restricted_model.df_resid 
        lrdf = (df_restr - df_full) #degrees of freedom
        lrstat = -2*(llf_restr - llf_full)
        pvalue = scipy.stats.chi2.sf(lrstat, df=lrdf)
        return(pvalue)
    
    def reg_type(self,link):
        if link == 'logit':
            fam = sm.families.Binomial()
        elif link == 'linear':
            fam = sm.families.Gaussian()
        return fam
    
    def LRT_multi(self, X, exclude, dataframe, link = 'logit'):
        """
        Parameters
        ----------
        X : list,array,set,etc
            All independent variables
        exclude : string
            Independent variable of interest
        dataframe : pandas dataframe
        link : string, bool, the default is 'logit'.
               'logit' for logistic regression
               'linear' for linear regression
        Returns
        -------
        p : P value of LRT
        """
        fam = self.reg_type(link)
        res_formula = self.label + ' ~ ' + '+'.join([i for i in X if i != exclude])
        restricted = sm.GLM.from_formula(res_formula, family = fam, data=dataframe)
        restr_model = restricted.fit()
        full_formula = self.label + ' ~ ' + '+'.join(X)
        fulled = sm.GLM.from_formula(full_formula, family = fam, data=dataframe)
        fulled_model = fulled.fit()
        p = np.round(self.LRT(restr_model, fulled_model),3)
        return p

    def LRT_uni(self,x, dataframe,link = 'logit'):
        fam = self.reg_type(link)
        restricted = sm.GLM.from_formula(self.label + '~1',family = fam, data = dataframe)
        restr_model = restricted.fit()
        fulled = sm.GLM.from_formula(self.label + '~' + x, family = fam, data = dataframe)
        fulled_model = fulled.fit()
        p = np.round(self.LRT(restr_model,fulled_model),3)
        return p


    def preprocess(self):
        """
        Description:
            rename columns' name
        Returns
        -------
        cols_dict : Dictionary
            {original variable name1 : var1, original variable name2: var2}
        cols_dict_rev : Dictionary
            Reverse of cols_dict
        new_df : pandas dataframe
            Renamed dataframe
        new_multi_cate : list
            Renamed multi-categorical variables
        """
        cols = [i for i in self.columns if i != self.label]
        cols_dict = dict(zip(cols, ['var' + str(i) for i in range(0,len(cols))]))
        cols_dict_rev = {v:k for k, v in cols_dict.items()}
        new_df = self.df.rename(columns = cols_dict)
        new_multi_cate = [cols_dict[i] for i in self.multicate_var]
        new_df[new_multi_cate] = new_df[new_multi_cate].astype('str')
        return cols_dict, cols_dict_rev,new_df,new_multi_cate
    
    def uni_reg(self, link = 'logit'):
        """
        Parameters
        ----------
        link : string, bool, the default is 'logit'.
               'logit' for logistic regression
               'linear' for linear regression
        Returns
        -------
        cols_dict : Dictionary
            {original variable name1 : var1, original variable name2: var2}
        cols_dict_rev : Dictionary
            Reverse of cols_dict
        new_df : pandas dataframe
            Renamed dataframe
        new_multi_cate : list
            Renamed multi-categorical variables
        uni_log_results : pandas dataframe
            Univariate regression result
        """
        cols_dict, cols_dict_rev, df, multi_cate = self.preprocess()
    
        uni_log_results = pd.DataFrame()
        for i in cols_dict.values():
            try:
                formula = self.label + '~' + i
                if link == 'logit':
                    uni_log  = sm.GLM.from_formula(formula,family=sm.families.Binomial(), data = df)
                    results = uni_log.fit()
                    result = pd.DataFrame({'Uni OR':np.round(np.exp(results.params[1:]),2),
                                           'Uni P Value':np.round(results.pvalues[1:],3)})
                    if i in multi_cate:
                        cate_p = pd.DataFrame({'Uni OR':'','Uni P Value':self.LRT_uni(i,dataframe = df)},
                                              index = [i])
                        result = pd.concat([cate_p,result])
                    else:
                        pass
                    uni_log_results = uni_log_results.append(result)
                elif link == 'linear':
                    uni_log  = sm.GLM.from_formula(formula,family=sm.families.Gaussian(), data = df)
                    results = uni_log.fit()
                    result = pd.DataFrame({'Uni Coef':np.round(results.params[1:],3),
                                           'Uni P Value':np.round(results.pvalues[1:],3)})
                    if i in multi_cate:
                        cate_p = pd.DataFrame({'Uni Coef':'','Uni P Value':self.LRT_uni(i,dataframe = df,
                                                                                link = 'linear')},index = [i])
                        result = pd.concat([cate_p,result])
                    else:
                        pass
                    uni_log_results = uni_log_results.append(result)
            except:
                print('Error raised: ', cols_dict_rev[i])
                break
        
        return cols_dict, cols_dict_rev, df, multi_cate,uni_log_results
    
    def multi_reg(self, X, dataframe, link = 'logit'):
        """
        Parameters
        ----------
        X : Independent variables for multivariate regression
        Returns
        -------
        mul_log_results : 
            Multivariate regression result
        """
        formula = self.label + ' ~ ' + '+'.join(X)
        if link == 'logit':
            mult_log = sm.GLM.from_formula(formula, family=sm.families.Binomial(), data=dataframe)
            result = mult_log.fit()
            mul_log_results = pd.DataFrame({'Multi OR': np.round(np.exp(result.params),2),
                          'Lower Bound':np.round(np.exp(result.conf_int()[0]),2),
                          'Upper Bound':np.round(np.exp(result.conf_int()[1]),2),
                                      'P Value':np.round(result.pvalues,3)})
            mul_log_results = mul_log_results.iloc[1:,:]
        elif link == 'linear':
            mult_reg = sm.GLM.from_formula(formula, family=sm.families.Gaussian(), data=dataframe)
            result = mult_reg.fit()
            mul_reg_results = pd.DataFrame({'Multi Coef': np.round(result.params,3),
                                            'Lower Bound':np.round(result.conf_int()[0],3),
                                            'Upper Bound':np.round(result.conf_int()[1],3),
                                            'P Value':np.round(result.pvalues,3)})
            mul_log_results = mul_reg_results.iloc[1:,:]
        return mul_log_results, result
    
    def process_final_df(self, dataframe, index_name):
        """
        Change dataframe's column name back to the original
        """
        dataframe = dataframe.rename(index = index_name)
        dataframe = dataframe.fillna('')
        dataframe[['Uni P Value','P Value']] = dataframe[['Uni P Value','P Value']].replace(0,'<0.001')
        return dataframe
    
    def insert_multi_p(self,X,multicate,result_df, dataframe, link='logit'):
        """
        Calculate the LRT p value for each multi-categorical variable
        """
        interset = set(X).intersection(set(multicate))
        if len(interset) == 0:
            result_df = result_df
        else:
            for k in interset:
                result_df.loc[k,'P Value'] = self.LRT_multi(X = X, exclude = k,dataframe = dataframe, link = link)
        return result_df
    

    def log_reg(self, p = 0.05,g = 10):
        """
        e.g: glm = lm.glm_regression(dataframe = temp_df,multicate_var = multi_cate, label = label)
            selection, log = glm.uni_selection(p = 0.05, link = 'logit')
        """
        cols_dict, cols_dict_rev, df, multi_cate,uni_log_results = self.uni_reg()
        # Multivariate Logistic
        subset = [i for i in uni_log_results[uni_log_results['Uni P Value']<p].index if i in cols_dict.values()]
        mul_log_results,model = self.multi_reg(X = subset, dataframe = df, link = 'logit')
        GOF = self.HL_test(model,df,subset, g = g)
        log_results = pd.concat([uni_log_results, mul_log_results],axis = 1)
        log_results = self.insert_multi_p(subset, multi_cate, log_results, df)
        log_results = self.process_final_df(log_results, index_name = cols_dict_rev)
        subset = [cols_dict_rev[i] for i in subset]
        print('{} variables were selected: \n'.format(len(subset)),
              '\n',subset)
        return(subset,log_results,GOF)


    def linear_reg(self,p = 0.05):
        cols_dict, cols_dict_rev, df, multi_cate,uni_reg_results = self.uni_reg(link = 'linear')
        # Multivariate lr
        subset = [i for i in uni_reg_results[uni_reg_results['Uni P Value']<p].index if i in cols_dict.values()]
        mul_reg_results,model = self.multi_reg(X = subset, dataframe = df, link = 'linear')
        GOF = self.r_square(model, df, subset)
        reg_results = pd.concat([uni_reg_results, mul_reg_results],axis = 1)
        reg_results = self.insert_multi_p(subset, multi_cate, reg_results, df, link = 'linear')
        reg_results = self.process_final_df(reg_results, index_name = cols_dict_rev)
        subset = [cols_dict_rev[i] for i in subset]
        print('{} variables were selected: \n'.format(len(subset)),
              '\n',subset)
        return(subset,reg_results,GOF)
    
    
    def uni_selection(self,p=0.05, link = 'logit'):
        """
        Build multivariate regression with selected variables 
        whose p values are less than p in univariate regression
        """
        if link == 'logit':
            selection, resutls, GOF = self.log_reg(p = p)
        elif link == 'linear':
            selection, resutls, GOF = self.linear_reg(p = p)
        return(selection, resutls, GOF)
    
    def stepwise_bw(self,threshold = 0.05, link = 'logit', g = 10):
        """
        Backward elimination method for stepwise regression.
        Not recommended when having large amount of variables and small sample size
        e.g: glm = lm.glm_regression(dataframe = df_copy,multicate_var = multi_cate, label = label)
             selectio, log3 = glm.stepwise_bw()
        """
        fam = self.reg_type(link)
        cols_dict, cols_dict_rev, df, multi_cate,uni_log_results = self.uni_reg(link = link)
        include = list(cols_dict.values())
        remove = {}
        stop = False
        while stop == False:
            formula = self.label + ' ~ ' + '+'.join(include)
            fit = sm.GLM.from_formula(formula, family=fam, data=df).fit()
            pvals = fit.pvalues[[i for i in include if i not in multi_cate]]
            interset = set(include).intersection(set(multi_cate))
            if len(interset) == 0:
                pass
            else:
                for ele in interset:
                    pvals[ele] = self.LRT_multi(X = include, exclude = ele, dataframe = df, link = link)
            largest_p = pvals.max()
            if largest_p >= threshold:
                try:
                    worst_feature = pvals.index[pvals.argmax()]
                except:
                    worst_feature = pvals.argmin()
                include.remove(worst_feature)
                remove[worst_feature] = largest_p
            elif largest_p < threshold:
                stop = True
        final_results,model = self.multi_reg(X = include, dataframe = df, link = link)
        if link == 'logit':
            GOF = self.HL_test(model,df,include, g = g)
        elif link == 'linear':
            GOF = self.r_square(model, df, include)
        final_results = pd.concat([uni_log_results, final_results],axis = 1)
        final_results = self.insert_multi_p(include, multi_cate, final_results, df, link = link)
        final_results = self.process_final_df(final_results, index_name = cols_dict_rev)
        include = [cols_dict_rev[i] for i in include]
        return(remove, include, final_results, GOF)
    
    # def stepwise_fw(self,link):
    #     cols_dict, cols_dict_rev, df, multi_cate,uni_log_results = self.uni_reg(link = link)
    #     include = list(cols_dict.values())
    #     initial_var = include[np.argmin(uni_reg.loc[include,'Uni P Value'])]
    #     stop = False
    #     while stop == False:
    #         formula = self.label + ' ~ ' + '+'.join(include)
    #         fit = sm.GLM.from_formula(formula, family=fam, data=df).fit()
            
    def stepwise_2way(self,threshold_in=0.05, threshold_out = 0.05, link = 'logit', g = 10):
        """
        Perform bidirectional stepwise regressio
        Referece:
            https://online.stat.psu.edu/stat501/lesson/10/10.2
        e.g:
        glm = lm.glm_regression(dataframe = df,multicate_var = multi_cate, label = label)
        selection4, log3 = glm.stepwise_2way()
        """
        fam = self.reg_type(link)
        include = []
        cols_dict, cols_dict_rev, df, multi_cate,uni_log_results = self.uni_reg(link = link)
        iteration = 1
        print('Selection threshold: ',threshold_in,'\n','Removal threshold: ', threshold_out)
        while True:
            changed = False
            # forward step
            candidates = [i for i in list(cols_dict.values()) if i not in include]
            pvals = pd.Series(index=include)
            for candidate in candidates:
                formula = self.label + '~' + '+'.join([candidate] + include)
                if candidate not in multi_cate:
                    uni_log  = sm.GLM.from_formula(formula,family=fam, data = df).fit()
                    pvals[candidate] = uni_log.pvalues[candidate]
                elif candidate in multi_cate:
                    pvals[candidate] = self.LRT_multi(X = [candidate] + include, exclude = [candidate],dataframe = df, link = link)
            min_p = pvals.min()
            if min_p < threshold_in:
                try:
                    best_candidate = pvals.index[pvals.argmin()]
                except:
                    best_candidate = pvals.argmin()
                include.append(best_candidate)
                changed=True
            mul_log_results,_ = self.multi_reg(X = include, dataframe = df, link = link)
            bw_pvals = mul_log_results['P Value'][[i for i in include if i not in multi_cate]]
            print(bw_pvals)
            interset = set(include).intersection(set(multi_cate))
            if len(interset) == 0:
                pass
            else:
                for k in interset:
                    bw_pvals[k] = self.LRT_multi(X = include, exclude = k, dataframe = df)
            largest_p = bw_pvals.max() # null if pvalues is empty
            if largest_p > threshold_out:
                changed=True
                try:
                    worst_feature = bw_pvals.index[bw_pvals.argmax()]
                except:
                    worst_feature = bw_pvals.argmin()
                include.remove(worst_feature)
                if worst_feature == best_candidate:
                    print('Endless Loop occurs: ', cols_dict_rev[worst_feature],'\n',
#                          'first_pvalue:',pvals,'\n',
                          'Iteration ends without including variable: ',cols_dict_rev[worst_feature],'\n',
#                          'second_pvalue:',bw_pvals,'\n',
                          'Selected Variables: ', '\n', [cols_dict_rev[i] for i in include])
                    break
            print('Iter: ', iteration, '\n', 
                  'Selected Variables: ', '\n',[cols_dict_rev[i] for i in include])
            iteration += 1
            if not changed:
                break
        final_results,model = self.multi_reg(X = include, dataframe = df, link = link)
        if link == 'logit':
            GOF = self.HL_test(model,df,include, g = g)
        elif link == 'linear':
            GOF = self.r_square(model, df, include)
        final_results = pd.concat([uni_log_results, final_results],axis = 1)
        final_results = self.insert_multi_p(include, multi_cate, final_results, df, link = link)
        final_results = self.process_final_df(final_results, index_name = cols_dict_rev)
        include = [cols_dict_rev[i] for i in include]
        return include,final_results,GOF
