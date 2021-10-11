"""
Helper functions to facilitate large samples hypothesis testing.
Based on:
    https://www.researchgate.net/publication/270504262_Too_Big_to_Fail_Large_Samples_and_the_p-Value_Problem

Made by Davide Badalotti under GPT-3.0 License
"""
from typing import Callable
import numpy as np
from typing import Union
from sklearn.utils import check_array
from tqdm import tqdm
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, ranksums
from functools import partial
import math
import matplotlib.pyplot as plt
plt.style.use(["ieee", "vibrant"])

class Tester:
    """
    Wrapper class for the testing framework.
    """

    def __init__(self) -> None:
        pass

    def ci_chart(
            self,
            X           : np.ndarray                   ,
            y           : np.ndarray            = None ,
            sample_size : float                 = 0.05 ,
            sample_max  : float                 = 1    ,
            balance     : bool                  = False,
            test_func   : Union[str,Callable]   = None ,
            plot_func   : Union[str,Callable]   = None ,
            **plotkwargs,
        ) -> np.ndarray:
        """
        Given a dataset, implements the logic for CI-plots
        based on dataset-samples of increasing dimensions
        and repeated tests. 
        Test and plot functions can be defined by the user and passed 
        directly, but some common tests are also implemented.

        Args
        ----
            * X : np.ndarray
                Dataset containing the instances, pandas dataframe is
                also accepted
            * y : np.1darray (optional)
                For test comparing two samples, binary vector
                differentiating each sample
            * sample_size : float in interval (0,1)
                Percentage of instances to be added to the sample at each
                iteration
            * sample_max : float
                Percentage of instances at last iteration
            * balance : for dual 
            * test_func : str or Callable
                Function to be specified for testing. Can be 
                [chi2, ranksums, ttest, welch] or a user defined function
                (see examples below)
            * plot_func : str or Callable
                Function to be specified for plotting. Can be
                [p-only] or user-defined or None. 
                When None, test_func must be a string.
        """
        #checks
        X = check_array(X, copy=False, force_all_finite='allow-nan', ensure_2d=False)
        assert 0 < sample_size < sample_max, "Sample size must be in (0,1) interval"
        if y is not None:
            assert y.shape[0] == X.shape[0], "X or y has no valid shape"
            assert ((y==0) | (y==1)).all() , "y contains invalid values"
        if isinstance(test_func, str):
            assert test_func in [
                'chi2', 'ranksums', 'ttest', 'welch'
            ], "Specify a valid test"
            # setting plot_func and test_func
            if test_func == "chi2":
                test_func = self.chi2_test
                plot_func = self.chi2_plot
            if test_func == "welch":
                test_func = partial(
                    self.ttest_test, stud=False
                )
                plot_func = self.mean_plot
            if test_func == "ttest":
                test_func = partial(
                    self.ttest_test, stud=True
                )
                plot_func = self.mean_plot
            if test_func == "ranksums":
                test_func = self.ranksums_test
                plot_func = self.median_plot
            plot_func = partial(
                plot_func,
                sample_max = sample_max,
                sample_size = sample_size,
                **plotkwargs
            )
        elif isinstance(test_func, Callable):
            assert isinstance(plot_func, Callable), "plot_func is not Callable"
            plot_func = partial(plot_func, **plotkwargs)
        else:
            raise AttributeError("Specify a valid test_func")
        # balancing data
        if balance:
            ind0 = np.argwhere(y==0).flatten()
            ind1 = np.argwhere(y==1).flatten()
            n  = min(ind1.shape[0], ind0.shape[0])
            if n == ind1.shape[0]:
                ind0 = np.random.choice(ind0, size=n)
            else:
                ind1 = np.random.choice(ind1, size=n)
            ind = np.concatenate((ind0, ind1))
            X = X[ind]
            y = y[ind]

        #loop for sampling
        ITERATIONS = int(sample_max/sample_size)  #number of iterations
        N          = int(sample_size*X.shape[0])  #points added per loop
        RESULTS    = [None] * ITERATIONS          #results list
        CUTS       = np.arange(0, X.shape[0]+1, N)[1:ITERATIONS+1]
        IDX        = np.arange(X.shape[0])        #sampling elements
        np.random.shuffle(IDX)
        
        for it, idx_max in enumerate(tqdm(CUTS)):
            ind = IDX[:idx_max]
            try:
                y_samp = y[ind]
            except TypeError:
                y_samp = None
            RESULTS[it] = test_func(X[ind], y_samp)
        
        # plotting
        return plot_func(RESULTS)
            
    def chi2_test(self, X, y):
        """
        Given a binary variable X and tested/control array y
        Performs chi2_contingency and returns conditional means
        """
        ct = pd.crosstab(columns=y, index=X)
        r0, r1 = np.mean(X[np.argwhere(y==0)]), np.mean(X[np.argwhere(y==1)])
        p  = chi2_contingency(ct)[1]
        return (r0, r1, p)
    
    def chi2_plot(
            self, 
            RESULTS     : list         , 
            sample_size : float        , 
            sample_max  : float        , 
            p_limit     : float = 0.05 ,
            labels      : tuple = (0,1), 
            title       : str   = None
            ):
        """
        Given an array of (ratio0, ratio1, pvalue) creates ci_plot
        Returns fig, ax 
        """
        r0 = [x[0] for x in RESULTS]
        r1 = [x[1] for x in RESULTS]
        p  = [x[2] for x in RESULTS]
        x  = np.linspace(sample_size, sample_max, len(p))
        fig, ax = plt.subplots()
        ax.set_title(title)
        # ratios
        ax.set_ylabel("Marginal ratios")
        ax.set_xlabel("Percentage of sampled dataset")
        ax.plot(
            x, r0, 
            marker="^", markersize=1.5, lw=0.7, label=labels[0]
        )
        ax.plot(
            x, r1,
            marker="^", markersize=1.5, lw=0.7, label=labels[1]
        )
        #p-value
        ax2 = ax.twinx()
        ax2.set_ylabel("p-values")
        ax2.plot(x, p, 'ko--', markersize=2, lw=0.7, label="p-values")
        ax2.hlines(
            p_limit, x[0], x[-1], 
            color="k", linewidth=0.3, label="p-value limit"
        )
        # legend 
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(handles = h1+h2, labels=l1+l2, fontsize='xx-small')  
        return fig, ax

    def ttest_test(self, X, y, stud=True):
        """
        Given X with shape (n_instances, 1) representing two 
        continuos distributions, y binary indicating the
        differentiating variable
        Performs student test and returns means, standard errors, p-values

        Returns
        -------
            5-ple : (m0, se0, m1, se1, p)
        """
        X0  = X[np.argwhere(y==0)]
        X1  = X[np.argwhere(y==1)]
        m0  = np.mean(X0)
        m1  = np.mean(X1)
        se0 = np.std(X0)/math.sqrt(X0.shape[0])
        se1 = np.std(X1)/math.sqrt(X1.shape[0])
        p   = ttest_ind(X0, X1, equal_var=stud)[1] 
        return (m0, se0, m1, se1, p) 
    
    def mean_plot(
            self, 
            RESULTS     : list         , 
            sample_size : float        , 
            sample_max  : float        ,
            p_limit     : float = 0.05 ,
            labels      : tuple = (0,1), 
            title       : str   = None
            ):
        """
        Given an array of (ratio0, ratio1, pvalue) creates ci_plot
        Returns fig, ax 
        """
        m0 = np.array([x[0] for x in RESULTS])
        se0= np.array([x[1] for x in RESULTS])
        m1 = np.array([x[2] for x in RESULTS])
        se1= np.array([x[3] for x in RESULTS])
        p  = np.array([x[4] for x in RESULTS])
        x  = np.linspace(sample_size, sample_max, len(p))
        fig, ax = plt.subplots()
        plt.suptitle(title)
        ax.set_ylabel("Mean values (shaded area = 2stderr)")
        ax.set_xlabel("Percentage of sampled dataset")
        #ratios
        ax.plot(
            x, m0, 
            marker="^", markersize=1.5, lw=0.7, label=labels[0]
        )
        ax.fill_between(
            y1 = m0+2*se0, y2 = m0-2*se0, x = x, 
            alpha = 0.20
        )
        ax.plot(
            x, m1,
            marker="^", markersize=1.5, lw=0.7, label=labels[1]
        )
        ax.fill_between(
            y1 = m1+2*se1, y2 = m1-2*se1, x = x,
            alpha = 0.20
        )
        #p-value
        ax2 = ax.twinx()
        ax2.set_ylabel("p-values")
        ax2.plot(x, p, 'ko--', markersize=1, lw=0.4, label="p-values")
        ax2.hlines(
            p_limit, x[0], x[-1], 
            color="k", linewidth=0.3, label="p-value threshold"
        )
        # legend 
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(handles = h1+h2, labels=l1+l2, fontsize='xx-small')  
        return fig, ax

    def ranksums_test(self, X, y):
        """
        Given X with shape (n_instances, 1) representing two 
        continuos distributions, y binary indicating the
        differentiating variable
        Performs ranksums and returns medians, CI, p-values
        #TODO IMPLEMENT CI FOR MEDIANS VIA BOOTSTRAP
        Returns
        -------
            7-ple : (m0, q0_low, q0_high, m1, q1_low, q1_high, p) 

        """
        X0  = X[np.argwhere(y==0)]
        X1  = X[np.argwhere(y==1)]
        m0  = np.median(X0)
        m1  = np.median(X1)
        q0_low, q0_high = self.bootstrap(X0)
        q1_low, q1_high = self.bootstrap(X1)
        p   = ranksums(X0, X1)[1]
        return (
            m0, q0_low, q0_high, 
            m1, q1_low, q1_high, 
            p
        ) 
    
    def median_plot(
            self, 
            RESULTS     : list         , 
            sample_size : float        , 
            sample_max  : float        ,
            p_limit     : float = 0.05 ,
            labels      : tuple = (0,1), 
            title       : str   = None
            ):
        """
        Given an array of (ratio0, ratio1, pvalue) creates ci_plot
        Returns fig, ax 
        """
        m0      = np.array([x[0] for x in RESULTS])
        q0_low  = np.array([x[1] for x in RESULTS])
        q0_high = np.array([x[2] for x in RESULTS])
        m1      = np.array([x[3] for x in RESULTS])
        q1_low  = np.array([x[4] for x in RESULTS])
        q1_high = np.array([x[5] for x in RESULTS])
        p       = np.array([x[6] for x in RESULTS])
        x       = np.linspace(sample_size, sample_max, len(p))
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_ylabel("Median values (shaded area = [5, 95] perc)")
        ax.set_xlabel("Fraction of sampled dataset")
        #ratios
        ax.plot(
            x, m0, 
            marker="^", markersize=1, lw=0.7, label=labels[0]
        )
        ax.fill_between(
            y1 = q0_high, y2 = q0_low, x = x, 
            alpha = 0.20
        )
        ax.plot(
            x, m1,
            marker="^", markersize=1, lw=0.7, label=labels[1]
        )
        ax.fill_between(
            y1 = q1_high, y2 = q1_low, x = x,
            alpha = 0.20
        )
        #p-value
        ax2 = ax.twinx()
        ax2.set_ylabel("p-values")
        ax2.plot(x, p, 'ko--', markersize=1, lw=0.4, label="p-values", alpha=0.5)
        ax2.hlines(
            p_limit, x[0], x[-1], 
            color="k", linewidth=0.3, label="p-value limit"
        )
        # legend 
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(handles = h1+h2, labels=l1+l2, fontsize='xx-small')  
        return fig, ax

    def bootstrap(self, X : np.ndarray) -> float:
        """
        Estimates median CI of X via bootstrap.
        #TODO improve flexibility and employ better
        estimates
        """
        it = 1000
        sample_size = int(X.shape[0]*0.1)
        if sample_size < 50: 
            sample_size=50
        # samples indexes
        index = np.random.randint(
            low=0, high=X.shape[0], size=it*sample_size
            
        )
        # obtaining 5 and 95 percentile
        return np.percentile(
            np.median(
                X[index].reshape(sample_size, it),
                axis=0
            ),
            [5, 95]
        )
