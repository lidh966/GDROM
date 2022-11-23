# Hidden Markov Models
#
# Author: Ron Weiss <ronweiss@gmail.com>
#         Shiqiao Du <lucidfrontier.45@gmail.com>
# API changes: Jaques Grobler <jaquesgrobler@gmail.com>
# Modifications to create of the HMMLearn module: Gael Varoquaux
# More API changes: Sergei Lebedev <superbobry@gmail.com>

"""
The :mod:`hmmlearn.hmm` module implements hidden Markov models.
"""

import numpy as np
#from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn import mixture
from . import _utils
from .base_DT import _BaseHMM
#from .utils import iter_from_X_lengths, normalize, fill_covars
import copy
#import pandas as pd
#from sklearn.tree import DecisionTreeRegressor
#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#from sklearn.tree import export_graphviz
#import pydotplus
#import sys
import random
from sklearn.preprocessing import StandardScaler

__all__ = [ "GaussianHMM"]

COVARIANCE_TYPES = frozenset(("spherical", "diag", "full", "tied"))


class GaussianHMM(_BaseHMM):
    r"""Hidden Markov Model with Gaussian emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    covariance_type : string, optional
        String describing the type of covariance parameters to
        use.  Must be one of

        * "spherical" --- each state uses a single variance value that
          applies to all features.
        * "diag" --- each state uses a diagonal covariance matrix.
        * "full" --- each state uses a full (i.e. unrestricted)
          covariance matrix.
        * "tied" --- all states use **the same** full covariance matrix.

        Defaults to "diag".

    min_covar : float, optional
        Floor on the diagonal of the covariance matrix to prevent
        overfitting. Defaults to 1e-3.

    startprob_prior : array, shape (n_components, ), optional
        Parameters of the Dirichlet prior distribution for
        :attr:`startprob_`.

    transmat_prior : array, shape (n_components, n_components), optional
        Parameters of the Dirichlet prior distribution for each row
        of the transition probabilities :attr:`transmat_`.

    means_prior, means_weight : array, shape (n_components, ), optional
        Mean and precision of the Normal prior distribtion for
        :attr:`means_`.

    covars_prior, covars_weight : array, shape (n_components, ), optional
        Parameters of the prior distribution for the covariance matrix
        :attr:`covars_`.

        If :attr:`covariance_type` is "spherical" or "diag" the prior is
        the inverse gamma distribution, otherwise --- the inverse Wishart
        distribution.

    algorithm : string, optional
        Decoder algorithm. Must be one of "viterbi" or`"map".
        Defaults to "viterbi".

    random_state: RandomState or an int seed, optional
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the Gaussian emissions.

    monitor\_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat\_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob\_ : array, shape (n_components, )
        Initial state occupation distribution.

    means\_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    covars\_ : array
        Covariance parameters for each state.

        The shape depends on :attr:`covariance_type`::

            (n_components, )                        if "spherical",
            (n_features, n_features)                if "tied",
            (n_components, n_features)              if "diag",
            (n_components, n_features, n_features)  if "full"

    Examples
    --------
    >>> from hmmlearn.hmm import GaussianHMM
    >>> GaussianHMM(n_components=2)
    ...                             #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    GaussianHMM(algorithm='viterbi',...
    """
    def __init__(self, inimodel, n_components=1, covariance_type='spherical',
                 min_covar=1e-3,
                 startprob_prior=1.0, transmat_prior=1.0,
                 means_prior=0, means_weight=0,
                 covars_prior=1e-2, covars_weight=1,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, tol=1, verbose=False,
                 params="stmc", lam = 0.8, maxlam = 0.99, shrink = 0.1, init_params="stmc", relax = "all", trials = 100, rand_state = 1000, iniassign = None, trials_iter = 1):
        _BaseHMM.__init__(self, inimodel = inimodel, n_components = n_components,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior, algorithm=algorithm,
                          random_state=random_state, n_iter=n_iter,
                          tol=tol, params=params, verbose=verbose,
                          init_params=init_params, maxlam=maxlam, lam = lam,shrink = shrink, relax = relax)

        self.covariance_type = covariance_type
        self.min_covar = min_covar
        self.means_prior = means_prior
        self.means_weight = means_weight
        self.covars_prior = covars_prior
        self.covars_weight = covars_weight 
        self.model = []
        self.trials = trials
        self.iniassign = iniassign
        self.rand_state = rand_state
        self.trials_iter = trials_iter
        #self.lastmodel = []
        for i in range(self.n_components):
            self.model.append(copy.deepcopy(self.inimodel))
        #    self.lastmodel.append(copy.deepcopy(self.inimodel))

    @property
    def covars_(self):
        """Return covars as a full matrix."""
        return fill_covars(self._covars_, self.covariance_type,
                           self.n_components, self.n_features)

    @covars_.setter
    def covars_(self, covars):
        self._covars_ = np.asarray(covars).copy()

    def _check(self):
        super(GaussianHMM, self)._check()

        self.means_ = np.asarray(self.means_)
        self.n_features = self.means_.shape[1]

        if self.covariance_type not in COVARIANCE_TYPES:
            raise ValueError('covariance_type must be one of {0}'
                             .format(COVARIANCE_TYPES))

        _utils._validate_covars(self._covars_, self.covariance_type,
                                self.n_components)

    def _init(self, O, F, lengths=None):
        super(GaussianHMM, self)._init(O, F, lengths=lengths)
       
        ''' 
        X = self.X

        _, n_features = O.shape
        
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
        
        
        for i in range(self.n_components):
            self.model[i] = copy.deepcopy(self.tree)
            
        
        self.n_features = n_features
        if 'm' in self.init_params or not hasattr(self, "means_"):
            kmeans = cluster.KMeans(n_clusters=self.n_components,
                                    random_state=self.random_state)
            kmeans.fit(X)
            self.means_ = kmeans.cluster_centers_
        if 'c' in self.init_params or not hasattr(self, "covars_"):
            cv = np.cov(X.T) + self.min_covar * np.eye(X.shape[1])
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars_ = \
                _utils.distribute_covar_matrix_to_match_covariance_type(
                    cv, self.covariance_type, self.n_components).copy()

        '''
              

        _, n_features = O.shape
        
        if hasattr(self, 'n_features') and self.n_features != n_features:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (n_features, self.n_features))
            
        data_all = np.concatenate((F, O), axis=1)
        #print(data_all.shape)
        
        if self.iniassign is None:
            kmeans = mixture.GaussianMixture(n_components=self.n_components, init_params = 'kmeans', n_init = 1, random_state = 1000)
            #kmeans = cluster.AgglomerativeClustering(n_clusters=self.n_components,
            #                            random_state=self.random_state)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_all)
    
            kmeans.fit(data_scaled)
            
            labels=kmeans.predict(data_scaled)
            
            try_once = copy.deepcopy(self)
            try_second = copy.deepcopy(self)
            try_second.n_iter = self.trials_iter
            try_once.iniassign = labels
            try_once.n_iter = self.trials_iter
            try_once.fit(O,F,lengths)
            current_best  = try_once.best_model
            current_high_fit = current_best.monitor_.best_fit
            current_labels = labels
            
            for ii in range(self.trials):
                print('Current iter:'+str(ii))
                
                kmeans = mixture.GaussianMixture(n_components=self.n_components, init_params = 'random', n_init = 1, random_state = ii)
                #kmeans = cluster.AgglomerativeClustering(n_clusters=self.n_components,
                #                            random_state=self.random_state)
                scaler = StandardScaler()
                data_scaled = scaler.fit_transform(data_all)
        
                kmeans.fit(data_scaled)
                
                labels=kmeans.predict(data_scaled)
                try_second.iniassign = labels
                try_second.fit(O,F,lengths)
                second_best  = try_second.best_model
                second_high_fit = second_best.monitor_.best_fit
                second_labels = labels
                
                if second_high_fit > current_high_fit:
                    current_best = copy.deepcopy(second_best)
                    current_high_fit = second_high_fit
                    current_labels = second_labels
                    
                    print('Labels updated!'+'Current iter = '+str(ii))
            labels = current_labels
            print('Final run!')
        else:
            labels = self.iniassign
            
        self.n_features = n_features
        self.means_ = []
        self.covars = []
        
        for i in range(self.n_components):
            O_i = O[labels == i,:]
            F_i = F[labels == i,:]
            self.model[i] = copy.deepcopy(self.tree)
            #self.lastmodel[i] = copy.deepcopy(self.tree)
            if len(F_i) == 0:
                random.seed(i)
                rand_select = random.sample(range(len(O)), 1000)
                O_i = O[rand_select,:]
                F_i = F[rand_select,:]
                
            self.model[i].fit(F_i, O_i)
            
            X = O_i - self.model[i].predict(F_i).reshape(-1,1)
            
            
            mean_i = np.mean(X)
            var_i = np.var(X)
            
         
       
            if 'm' in self.init_params or not hasattr(self, "means_"):
                self.means_.append([mean_i])
                
            if 'c' in self.init_params or not hasattr(self, "covars_"):
                self.covars.append(var_i)
        
        self.means_ = np.array(self.means_)
        self.covars = np.array(self.covars)
        self.covars = np.maximum(self.covars, 1e-15)
        
        self._covars_ = np.tile(self.covars[:, np.newaxis],
                        (1, self.n_features))
                  
                    
    def _compute_log_likelihood(self, O, F):
        n_samples,_ = O.shape
        lpr = np.zeros((n_samples, self.n_components))
        for i in range(self.n_components):
            X = O - self.model[i].predict(F).reshape(-1, 1)   
            lpr[:,i] = -0.5 * ( np.log(2 * np.pi) + np.log(self._covars_[i,:])
                      + np.dot(((X-self.means_[i,:]) ** 2), (1.0 / self._covars_[i,:]).T))
        return np.nan_to_num(lpr, posinf=1e10, neginf=-1e10)
    
    def _generate_sample_from_state(self, state, random_state=None):
        random_state = check_random_state(random_state)
        return random_state.multivariate_normal(self.means_[state], self.covars_[state])

    def _initialize_sufficient_statistics(self):
        stats = super(GaussianHMM, self)._initialize_sufficient_statistics()
        stats['post'] = np.zeros(self.n_components)
        stats['obs'] = np.zeros((self.n_components, self.n_features))
        stats['obs**2'] = np.zeros((self.n_components, self.n_features))
        return stats

    def _accumulate_sufficient_statistics(self, stats, O, F, framelogprob,
                                          posteriors, fwdlattice, bwdlattice):
        super(GaussianHMM, self)._accumulate_sufficient_statistics(
            stats, O, F, framelogprob, posteriors, fwdlattice, bwdlattice)
    
        stats['post'] += posteriors.sum(axis=0)

        for i in range(self.n_components):
            if 'm' in self.params or 'c' in self.params:
                
                obs = O - self.model[i].predict(F).reshape(-1,1)
                stats['obs'][i,:] += np.dot(posteriors[:,i].T, obs)
                
                if 'c' in self.params:
                    stats['obs**2'][i,:] += np.dot(posteriors[:,i].T, obs ** 2)


    def _do_mstep(self, stats, O, F, posteriors):
        super(GaussianHMM, self)._do_mstep(stats,O, F, posteriors)

        # TODO: find a proper reference for estimates for different
        #       covariance models.
        # Based on Huang, Acero, Hon, "Spoken Language Processing",
        # p. 443 - 445

    def _updateGM(self, stats, O, F, posteriors):
        '''
        '''
#        print('GM')
#        print(self.startprob_)
#        print(stats['obs'])
#        print(stats['post'])
#        print(self.model[0].coef_)
#        print(self.model[0].intercept_)
#
#        print(self.model[1].coef_)
#        print(self.model[1].intercept_)
#
#        print(self.transmat_)
#        
        denom = stats['post'].reshape(-1,1)
        np.nan_to_num(denom, posinf=1e10, neginf=-1e10)

        if 'm' in self.params:
            self.means_ = (stats['obs'])/ np.maximum(denom, 1e-15)                
            # self.means_  = np.zeros((self.n_components, 1))


            
        if 'c' in self.params:
            cv_num = (stats['obs**2'] - 2 * self.means_ * stats['obs']
                          + self.means_**2 * denom)
            cv_den =  denom
            self._covars_ = (cv_num) / np.maximum(cv_den, 1e-15)
            np.nan_to_num(self._covars_, posinf=1e10, neginf=-1e10)
            self._covars_ = np.maximum(self._covars_, 1e-15)

#        print(self.means_)
#        print(self._covars_)
#        print(self.model)
#                    
                