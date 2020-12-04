import pandas as pd
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.stats.api as sms
from statsmodels.compat import lzip
import statsmodels.tools.eval_measures as emeas
import numpy as np
import matplotlib.gridspec as gridspec
import math
from statsmodels.graphics.gofplots import ProbPlot
import json
from scipy.stats import bartlett



#TODO:
# -Test stepwise regressions
# -Make AIC stepwise regressions
# -Test simple regression for categorical/multivariate
# -Knock out the rest of OLSRR



class simple_regression:
    
    def __init__(self, formula, data):
        self.model = ols(formula, data).fit()
        self.summary = self.model.summary()
        self.residuals = self.model.resid
        self.fittedvalues = self.model.fittedvalues
        self.influence = self.model.get_influence()
        self.obs = self.model.nobs
        self.nump = len(self.model.params)
        
        
    def ols_residual_plots(self, predictor):
        fig = plt.figure(figsize=(12,8))
        fig = sm.graphics.plot_regress_exog(self.model, predictor, fig=fig)
        plt.show()
        
    def ols_test_breusch_pagan(self):
        names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
        bp = sms.het_breuschpagan(self.residuals, self.model.model.exog)
        return lzip(names, bp)
    
    def ols_aic(self):
        return self.model.aic
    
    def ols_apc(self):
        return ((self.obs + self.nump)/(self.obs-self.nump))*(1-self.model.rsquared)
    
    def ols_fpe(self):
        return self.model.mse_model * ((self.obs + self.nump) / self.obs)
    
    def ols_hsp(self):
        return self.model.mse_model / (self.obs - self.nump - 1)
    
    def ols_leverage(self):
        return self.influence.hat_matrix_diag
    
    def ols_msep(self):
        return (self.model.mse_model * (self.obs+1) * (self.obs-2)) / (self.obs * (self.obs - self.nump - 1))
    
    def ols_plot_added_variable(self):
        fig = sm.graphics.plot_partregress_grid(self.model)
        fig.tight_layout(pad=1.0)
        
    def ols_plot_comp_plus_resid(self):
        fig = sm.graphics.plot_ccpr_grid(self.model)
        fig.tight_layout(pad=1.0)
        
    def ols_plot_obs_fit(self):
        sns.scatterplot([row[1] for row in self.model.model.exog], self.model.fittedvalues)
        plt.title("Fitted Values vs. Observed Values")
        plt.xlabel('Observed values')
        plt.ylabel('Fitted Values')
    
    def ols_plot_cooksd_bar(self):
        (c, p) = self.influence.cooks_distance
        plt.stem(np.arange(len(c)), c, markerfmt=",")
        
    def ols_plot_dfbetas(self):
        fig, axs = plt.subplots((math.ceil(self.nump/2)), 2)
        for i, ax in enumerate(axs.flat):
            if i < self.nump:
                self.influence.plot_index(y_var="dfbeta", ax=ax, idx=i)
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        plt.tight_layout()
        fig.show()
    
    def ols_plot_dffits(self):
        (c, p) = self.influence.dffits
        plt.stem(np.arange(len(c)), c, markerfmt=",")
        
    @staticmethod
    def check_linearity_assumption(fitted_y, residuals):
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(fitted_y, residuals,
                                       lowess=True,
                                       scatter_kws={'alpha': 0.5},
                                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

        plot_1.axes[0].set_title('Residuals vs Fitted')
        plot_1.axes[0].set_xlabel('Fitted values')
        plot_1.axes[0].set_ylabel('Residuals')

    @staticmethod
    def check_residual_normality(residuals_normalized):
        qq = ProbPlot(residuals_normalized)
        plot_2 = qq.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)
        plot_2.axes[0].set_title('Normal Q-Q')
        plot_2.axes[0].set_xlabel('Theoretical Quantiles')
        plot_2.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(residuals_normalized)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        for r, i in enumerate(abs_norm_resid_top_3):
            plot_2.axes[0].annotate(i,
                                    xy=(np.flip(qq.theoretical_quantiles, 0)[r],
                                        residuals_normalized[i]))


    @staticmethod
    def check_homoscedacticity(fitted_y, residuals_norm_abs_sqrt):
        plot_3 = plt.figure()
        plt.scatter(fitted_y, residuals_norm_abs_sqrt, alpha=0.5)
        sns.regplot(fitted_y, residuals_norm_abs_sqrt,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plot_3.axes[0].set_title('Scale-Location')
        plot_3.axes[0].set_xlabel('Fitted values')
        plot_3.axes[0].set_ylabel("$\\sqrt{|Standardized Residuals|}$")

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residuals_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        for i in abs_sq_norm_resid_top_3:
            plot_3.axes[0].annotate(i,
                                    xy=(fitted_y[i],
                                        residuals_norm_abs_sqrt[i]))

    @staticmethod
    def check_influcence(leverage, cooks, residuals_normalized):
        plot_4 = plt.figure()
        plt.scatter(leverage, residuals_normalized, alpha=0.5)
        sns.regplot(leverage, residuals_normalized,
                    scatter=False,
                    ci=False,
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plot_4.axes[0].set_xlim(0, max(leverage) + 0.01)
        plot_4.axes[0].set_ylim(-3, 5)
        plot_4.axes[0].set_title('Residuals vs Leverage')
        plot_4.axes[0].set_xlabel('Leverage')
        plot_4.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        leverage_top_3 = np.flip(np.argsort(cooks), 0)[:3]
        for i in leverage_top_3:
            plot_4.axes[0].annotate(i,
                                    xy=(leverage[i],
                                        residuals_normalized[i]))

    def ols_plot_diagnostics(self):
        linear_model = self.model
        diagnostic_result = {}

        fitted_y = linear_model.fittedvalues
        residuals = linear_model.resid
        residuals_normalized = linear_model.get_influence().resid_studentized_internal
        model_norm_residuals_abs_sqrt = np.sqrt(np.abs(residuals_normalized))
        leverage = linear_model.get_influence().hat_matrix_diag
        cooks = linear_model.get_influence().cooks_distance[0]

        self.check_linearity_assumption(fitted_y, residuals)
        self.check_residual_normality(residuals_normalized)
        self.check_homoscedacticity(fitted_y, model_norm_residuals_abs_sqrt)
        self.check_influcence(leverage, cooks, residuals_normalized)
        
    def ols_plot_resid_box(self):
        ax = sns.boxplot(self.model.resid)
        plt.title("Residual Box Plot")
    
    def ols_plot_resid_fit(self):
        fitted_y = self.model.fittedvalues
        residuals = self.model.resid
        self.check_linearity_assumption(fitted_y, residuals)
        
    def ols_plot_resid_hist(self):
        sns.histplot(data=self.model.resid, kde=True)
        plt.title("Residual Histogram")

    def ols_plot_resid_lev(self):
        plot_lm_4 = plt.figure(4)

        model_leverage = self.influence.hat_matrix_diag
        model_norm_residuals = self.influence.resid_studentized_internal
        model_cooks = self.influence.cooks_distance[0]
        plt.scatter(model_leverage, model_norm_residuals, alpha=0.5)
        sns.regplot(model_leverage, model_norm_residuals, 
                    scatter=False, 
                    ci=False, 
                    lowess=True,
                    line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})


        plot_lm_4.axes[0].set_title('Residuals vs Leverage')
        plot_lm_4.axes[0].set_xlabel('Leverage')
        plot_lm_4.axes[0].set_ylabel('Standardized Residuals')

        # annotations
        leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]

        for i in leverage_top_3:
            plot_lm_4.axes[0].annotate(i, 
                                    xy=(model_leverage[i], 
                                        model_norm_residuals[i]))
            
    def ols_plot_resid_qq(self):
        self.check_residual_normality(self.influence.resid_studentized_internal)
        
    def ols_plot_resid_stand(self):
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(np.arange(len(self.model.model.endog)), self.influence.resid_studentized_internal)

        plot_1.axes[0].set_title('Standardized Residuals Chart')
        plot_1.axes[0].set_xlabel('Observation')
        plot_1.axes[0].set_ylabel('Standardized Residuals')
    
    def ols_plot_resid_stud(self):
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(np.arange(len(self.model.model.endog)), self.influence.resid_studentized_external)

        plot_1.axes[0].set_title('Studentized Residuals Chart')
        plot_1.axes[0].set_xlabel('Observation')
        plot_1.axes[0].set_ylabel('Studentized Residuals')
        
    def ols_plot_resid_stud_fit(self):
        plot_1 = plt.figure()
        plot_1.axes[0] = sns.residplot(np.arange(len(self.model.fittedvalues)), self.influence.resid_studentized_external)

        plot_1.axes[0].set_title('Studentized Residuals vs. Fitted Values')
        plot_1.axes[0].set_xlabel('Fitted Values')
        plot_1.axes[0].set_ylabel('Studentized Residuals')
        
    def ols_plot_response(self):
        fig, axs = plt.subplots(nrows=3)
        sns.distplot(self.model.model.endog, ax=axs[0])
        sns.lineplot(self.model.model.endog, np.arange(len(self.model.model.endog)), ax=axs[1])
        sns.boxplot(self.model.model.endog, ax=axs[2])
        fig.set_size_inches(15, 10)
    
    def ols_pred_rsq(self):
        return self.model.rsquared_adj
    
    def ols_press(self):
        return self.influence.ess_press
    
    def ols_sbc(self):
        return self.model.bic
    
def ols_step_forward_p(X, y, threshold_in=0.05, verbose=False):
    initial_list = []
    included = list(initial_list)
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

        if not changed:
            break

    return included


def ols_step_forward_aic(X, y, verbose=False):
    initial_list = []
    included = list(initial_list)
    best_aic = math.inf
    while True:
        changed=False
        excluded = list(set(X.columns)-set(included))
        new_aic = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_aic[new_column] = model.aic
        temp_aic = new_aic.min()
        if temp_aic < best_aic:
            best_aic = temp_aic
            best_feature = new_aic.idxmin()
            included.append(best_feature)
            changed=True
            if verbose:
                print('Add  {:30} with aic {:.6}'.format(best_feature, best_aic))

        if not changed:
            break

    return included


        
def ols_step_backward_p(X, y, threshold_out=0.0000000005, verbose=False):
    included=list(X.columns)
    while True:
        changed=False
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.idxmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
        if not changed:
            break
    return included



def ols_plot_reg_line(response, predictor):
    ax = sns.regplot(x=predictor, y=response)
    plt.title("Regression Line")
    
def ols_test_bartlett(*args):
    return bartlett(*args)