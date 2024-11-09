import numpy as np
import emcee
import dynesty
import matplotlib.pyplot as plt
from multiprocessing import Pool
import multiprocessing
import matplotlib as mpl
import time
import os
import pandas as pd
import corner
from scipy.stats import uniform, norm, truncnorm, loguniform
import yaml

class TTVFit:
    def __init__(self, data, period, parameters, parameter_names=[], kind='Linear', rej_sigma = None ,multiprocessing = False, limit=None, sampler='emcee',sampler_parameters=None,lin_parameters=None):        
        """
        Initialize TTVFit object.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Contains 'T_mid' and ''Uncertainty'' columns at least.
        period : float
            Orbital period of the exoplanet.
        parameters : list
            A list of initial parameters for the model, the first two parameters are expected to be t0 and period, the last parameter need to be the changing rate which is either pdot or domega/dN
        parameter_names : list, optional
            A list of parameter names for the model.
        kind : str, optional
            Type of TTV model. Supported values: 'Linear', 'Pdot', 'Precession'.
        rej_sigma : float, optional
            The threshold of rejection in terms of standard deviation. Default is None.
        multiprocessing : bool, optional
            Whether to use multiprocessing. Default is False.
        limit : int, optional
            Default is None.
        sampler : str, optional
            Sampler to use. Supported values: 'emcee', 'dynesty' and 'dynamic_dynesty'
        sampler_parameters : dict, optional
            The parameters for the sampler. Default is None.
        """
        self.data = data 
        self.period = period
        self.rej_sigma = rej_sigma
        self.lin_parameters = lin_parameters

        self.kind = kind
        self.data.index = range(len(self.data))
        self.sampler = sampler
        if self.sampler == 'emcee':
            self.Prior = self.emceePrior
        elif 'dynesty' in self.sampler:
            self.Prior = self.dynestyPrior

        self.data['Epoch'] = np.array(np.round((self.data['T_mid']-self.data['T_mid'][0])/period),dtype=int)
        if limit is not None:
            self.limit = limit
        else:
            self.limit = self.data['Epoch'].max()//1000*1000+1000
        self.data['Valid'] = np.array([True]*len(self.data))
        self.multiprocessing = multiprocessing
        self.parameters = parameters
        self.parameter_names = parameter_names
        self.sampler_parameters = sampler_parameters
        if 'Label' not in self.data.columns:
            self.data['Label'] = ['data']*len(self.data)

        if kind == 'Precession':
            self.Model = self.apsidalPrecessionModel
            print('You are fitting TTV with Precession model')
        elif kind == 'Pdot':
            self.Model = self.pdotModel
            print('You are fitting TTV with Pdot model')
        elif kind == 'Linear':
            self.Model = self.linearModel
            print('You are fitting TTV with Linear model')
        # elif kind == 'Pdot1':
        #     self.Model = self.pdotModel1
        #     print('You are fitting TTV with Pdot1 model')
        # elif kind == 'Multi':
        #     self.Model = self.apsidalPrecessionPdotModel
        #     print('You are fitting TTV with PrecessionPdot model')
        # elif kind == 'dPdot':
        #     self.Model = self.dpdotModel
        #     print('You are fitting TTV with dPdot model')
        # elif kind == 'FixedEccentricity':
        #     self.Model = self.apsidalPrecessionFixedEccentricityModel
        #     print('You are fitting TTV with FixedEccentricity model')
        # elif kind == 'OriginalPrecession':
        #     self.Model = self.apsidalPrecessionModelOriginal
        #     print('You are fitting TTV with OriginalPrecession model')
        # elif kind == 'OriginalMulti':
        #     self.Model = self.apsidalPrecessionPdotModelOriginal
        #     print('You are fitting TTV with OriginalPrecessionPdot model')
        else:
            raise ValueError("Not a valid kind of model")

    def getBIC(self):
        """
        Compute the Bayesian Information Criterion (BIC) for the model.
        
        Notes
        -----
        The BIC is a criterion for model selection among a finite set of models.
        It is based on the likelihood function and is defined as:
        BIC = chi2 + k \\* log(n),
        where chi2 is the chi-squared statistic, k is the number of free
        parameters, and n is the number of data points.
        
        Returns
        -------
        bic : float
            The Bayesian Information Criterion (BIC) for the model.
        """
        obs = np.array(self.data[self.data['Valid']]['T_mid'])
        pre = np.array(self.data[self.data['Valid']]['T_mid_pre'])
        error = np.array(self.data[self.data['Valid']]['Uncertainty'])
        k = len(self.parameters)
        c=[]
        for i in range(len(obs)):
            c.append((obs[i]-pre[i])**2/error[i]**2)
        c=np.array(c)
        chi2 = c.sum()
        return chi2+k*np.log(len(obs))

    def apsidalPrecessionModel(self, params):
        """
        Compute the times of transit for an apsidal precession model.

        Parameters
        ----------
        params : list
            The parameters of the apsidal precession model, in the order
            [t0, period, e, omega_0, d_omega_dE].

        Returns
        -------
        t_tras : array
            The times of transit for the apsidal precession model.

        Notes
        -----
        The apsidal precession model is given by:
        t_tras[i] = t_tras[i-1] + period + e*P_ano/(np.pi)*d_omega_dE*np.sin(omegas[i]),
        where P_ano is the anomalistic period, e is the eccentricity,
        omega_0 is the initial argument of periapsis, d_omega_dE is the rate
        of change of the argument of periapsis with respect to the mean
        anomaly, and omegas[i] is the argument of periapsis at the ith
        transit.
        """
        t0, period, e, omega_0, d_omega_dE = params
        t_tras = [np.nan]*self.limit
        omegas = [np.nan]*self.limit
        t_tras[0] = t0
        omegas[0] = omega_0
        P_ano = period/(1-d_omega_dE/(2*np.pi))
        for i in range(1,self.limit):
            omegas[i] = (omegas[i-1] + d_omega_dE) % (2*np.pi)
            t_tras[i] = t_tras[i-1] + period + e*P_ano/(np.pi)*d_omega_dE*np.sin(omegas[i])
        return t_tras
    
    # def apsidalPrecessionModelOriginal(self, params):
    #     t0, period, e, omega_0, d_omega_dE = params
    #     t_tras = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     for i in range(1,self.limit):
    #         t_tras[i] = t0 + period*i - (e*(period / (1-d_omega_dE/(2*np.pi))) / np.pi) * np.cos(omega_0 + i * d_omega_dE)
    #     return t_tras

    # def apsidalPrecessionFixedEccentricityModel(self, params):
    #     t0, period, omega_0, d_omega_dE = params
    #     e = 0.06
    #     t_tras = [np.nan]*self.limit
    #     omegas = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     omegas[0] = omega_0
    #     P_ano = period/(1-d_omega_dE/(2*np.pi))
    #     for i in range(1,self.limit):
    #         omegas[i] = (omegas[i-1] + d_omega_dE) % (2*np.pi)
    #         t_tras[i] = t_tras[i-1] + period + e*P_ano/(np.pi)*d_omega_dE*np.sin(omegas[i])
    #     return t_tras

    # def apsidalPrecessionModel(self, params):
    #     t_tras = []
    #     t0, period, e, omega_0, d_omega_dE = params
    #     t_tras = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     P_ano = period/(1-d_omega_dE/(2*np.pi))
    #     for i in range(1,self.limit):
    #         omega_N = (omega_0 + d_omega_dE*i) % (2*np.pi)
    #         t_tras[i] = t0 + i*period - e*P_ano/(np.pi)*np.cos(omega_N)
    #     return t_tras

    def pdotModel(self, params):
        """
        Compute the times of transit for a pdot model.

        Parameters
        ----------
        params : array
            The parameters of the pdot model, given by [t0, period0, pdot].

        Returns
        -------
        t_tras : array
            The times of transit for the pdot model.

        Notes
        -----
        The pdot model is given by:
        t_tras[i] = t_tras[i-1] + period[i-1],
        period[i] = (1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1],
        where pdot is the rate of change of the orbital period with respect to time,
        and factor is the conversion from years to days.
        """
        factor = 24*60*60*1000*365
        t0, period0, pdot = params
        t_tras = [np.nan]*self.limit
        period = [np.nan]*self.limit
        t_tras[0] = t0
        period[0] = period0
        for i in range(1,self.limit):
            period[i]=(1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1]
            t_tras[i] = t_tras[i-1] + period[i-1]
        return t_tras
    
    # def pdotModel1(self, params):
    #     factor = 24*60*60*1000*365
    #     t0, period0, pdot = params
    #     t_tras = [np.nan]*self.limit
    #     period = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     period[0] = period0
    #     for i in range(1,self.limit):
    #         period[i]=(1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1]
    #         t_tras[i] = t_tras[i-1] + (period[i-1]+period[i])/2
    #     return t_tras
    
    # def dpdotModel(self,params):
    #     t0, period0, pdot0, dpdot = params
    #     factor = 24*60*60*1000*365
    #     factor1 = 24*60*60
    #     pdot0 /= factor
    #     dpdot *= factor1
    #     t_tras = [np.nan]*self.limit
    #     period = [np.nan]*self.limit
    #     pdot = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     period[0] = period0
    #     pdot[0] = pdot0
    #     for i in range(1,self.limit):
    #         pdot[i] = pdot[i-1] + period[i-1]*dpdot
    #         # period[i]= period[i-1] + period[i-1]*pdot[i-1]
    #         period[i]=(1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1]

    #         t_tras[i] = t_tras[i-1] + period[i-1]
    #     return t_tras
    
    # def apsidalPrecessionPdotModel(self, params):
    #     factor = 24*60*60*1000*365
    #     t0, period0, pdot, e, omega_0, d_omega_dE = params
    #     t_tras = [np.nan]*self.limit
    #     period = [np.nan]*self.limit
    #     omegas = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     period[0] = period0
    #     omegas[0] = omega_0
    #     for i in range(1,self.limit):
    #         period[i]=(1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1]
    #         omegas[i] = (omegas[i-1] + d_omega_dE) % (2*np.pi)
    #         P_ano = period[i]/(1-d_omega_dE/(2*np.pi))
    #         t_tras[i] = t_tras[i-1] + period[i] + e*P_ano/(np.pi)*d_omega_dE*np.sin(omegas[i])
    #     return t_tras
    
    # def apsidalPrecessionPdotModelOriginal(self, params):
    #     factor = 24*60*60*1000*365
    #     t0, period0, pdot, e, omega_0, d_omega_dE = params
    #     t_tras = [np.nan]*self.limit
    #     period = [np.nan]*self.limit
    #     t_tras[0] = t0
    #     period[0] = period0
    #     for i in range(1,self.limit):
    #         period[i]=(1+pdot/(2*factor))/(1-pdot/(2*factor))*period[i-1]
    #         t_tras[i] = t0 + period[i]*i - (e*(period[i] / (1-d_omega_dE/(2*np.pi))) / np.pi) * np.cos(omega_0 + i * d_omega_dE)
    #     return t_tras
    
    def linearModel(self, params):
        """
        Compute the times of transit for a linear model.

        Parameters
        ----------
        params : array
            The parameters of the linear model, given by [t0, period].

        Returns
        -------
        t_tras : array
            The times of transit for the linear model.

        Notes
        -----
        The linear model is given by:
        t_tras[i] = t_tras[i-1] + period,
        where t0 is the initial time of transit and period is the orbital period.
        """
        t0, period = params
        t_tras = []
        t_tras = [np.nan]*self.limit
        t_tras[0] = t0
        for i in range(1,self.limit):
            t_tras[i] = t_tras[i-1] + period
        return t_tras
    
    def Likelihood(self, params):
        """
        Compute the likelihood of the observed data given the parameters.

        Parameters
        ----------
        params : array
            The parameters of the model.

        Returns
        -------
        likelihood : float
            The likelihood of the observed data given the parameters.

        Notes
        -----
        The likelihood is computed as the sum of the squared differences between the observed and predicted times of transit, divided by the squared uncertainties.
        """
        t_obs = np.array(self.data[self.data['Valid']]['T_mid'],dtype=object)
        t_err = np.array(self.data[self.data['Valid']]['Uncertainty'],dtype=object)
        epochs = np.array(self.data[self.data['Valid']]['Epoch'],dtype=object)
        t_pre = self.Model(params)
        t_pre = np.array(t_pre,dtype=object)
        x = []
        for i in range(len(t_obs)):
            xi = (t_obs[i]-t_pre[epochs[i]])**2/t_err[i]**2
            x.append(xi)
        x = np.array(x,dtype=object)
        return -0.5*np.sum(x)

    def emceePrior(self, params):
        """
        Compute the prior probability of the given parameters in the format required by emcee.
        
        Parameters
        ----------
        params : array
            The parameters to evaluate the prior probability for.
        
        Returns
        -------
        prior : float
            The log-prior probability of the given parameters. The prior distribution of each parameter is uniform if the last element of the parameter list is 'U',
            and normal if the last element is 'N'.
        
        Notes
        -----
        The prior probability is computed by summing the log-probabilities of each parameter.
        """
        log_probs = []
        for i, par in enumerate(params):
            if self.parameters[i][-1]=='U':
                log_probs.append(uniform.logpdf(par, self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0]))
            elif self.parameters[i][-1]=='N':
                log_probs.append(norm.logpdf(par, self.parameters[i][0], self.parameters[i][1]))
            elif self.parameters[i][-1]=='TN':
                mean, std ,low, high =   self.parameters[i][:-1]
                low_n, high_n = (low - mean) / std, (high - mean) / std  # standardize
                log_probs.append(truncnorm.logpdf(par, low_n, high_n, loc=mean, scale=std))
            elif self.parameters[i][-1]=='LU':
                log_probs.append(loguniform.logpdf(par, self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0]))
            else:
                ValueError('Prior distribution not recognized, should be "U", "N", "TN", "LU"')
        return np.sum(log_probs)

    def dynestyPrior(self, params):
        """
        Compute the prior probability of the given parameters in the format required by dynesty.
        
        Parameters
        ----------
        params : array
            The parameters to evaluate the prior probability for.
        
        Returns
        -------
        prior : array
            The prior probability of the given parameters, in the format required by dynesty. The prior distribution of each parameter is uniform if the last element of the parameter list is 'U',
            and normal if the last element is 'N'.
        
        Notes
        -----
        The prior probability is computed by summing the log-probabilities of each parameter.
        """
        params_list = []
        for i, par in enumerate(params):
            if self.parameters[i][-1]=='U':
                params_list.append(uniform.ppf(par, self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0]))
            elif self.parameters[i][-1]=='N':
                params_list.append(norm.ppf(par, self.parameters[i][0], self.parameters[i][1]))
            elif self.parameters[i][-1]=='TN':
                mean, std ,low, high =   self.parameters[i][:-1]
                low_n, high_n = (low - mean) / std, (high - mean) / std  # standardize
                params_list.append(truncnorm.ppf(par, low_n, high_n, loc=mean, scale=std))
            elif self.parameters[i][-1]=='LU':
                params_list.append(loguniform.ppf(par, self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0]))
            else:
                ValueError('Prior distribution not recognized, should be "U", "N", "TN", "LU"')
        return np.array(params_list)

    def Probability(self, params):
        """
        Compute the total probability of the given parameters, given by the log-prior probability plus the log-likelihood of the model.
        
        Parameters
        ----------
        params : array
            The parameters to evaluate the probability for.
        
        Returns
        -------
        probability : float
            The total probability of the given parameters.
        
        Notes
        -----
        The prior probability is computed by calling the Prior method, and the likelihood is computed by calling the Likelihood method.
        """
        lp = self.Prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.Likelihood(params)
    
    def Method(self):        
        """
        Perform the MCMC or nested sampling run to sample the posterior distribution, depending on the value of sampler.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The sampler to be used is determined by the sampler attribute of the object. The number of walkers, steps, and burnin steps are determined by the sampler_parameters attribute of the object.
        The multiprocessing flag determines whether to use multiprocessing or not. If multiprocessing is True, then the number of threads is set to be the number of cpus available minus 1.
        """
        nwalkers = 32
        nsteps = 3000
        nburnin = 300
        nlive = 3000
        dlogz = 0.0001
        nlive_init = 3000
        dlogz_init = 0.0001

        N_threads = multiprocessing.cpu_count() - 1
        if self.multiprocessing == True:pool = Pool(processes=N_threads)
        else:pool = None

        if self.multiprocessing == True:pool = Pool()
        else:pool = None
        ndim = len(self.parameters)

        start = time.time()
        if self.sampler == 'emcee':
            if self.sampler_parameters is not None:
                nwalkers = self.sampler_parameters['nwalkers']
                nsteps = self.sampler_parameters['nsteps']
                nburnin = self.sampler_parameters['nburnin']
            print('Start MCMC')
            pos = []
            for i in range(len(self.parameters)):
                if self.parameters[i][-1]=='U':
                    pos.append(uniform.rvs(self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0],size=nwalkers))
                elif self.parameters[i][-1]=='N':
                    pos.append(norm.rvs(self.parameters[i][0], self.parameters[i][1], size=nwalkers))
                elif self.parameters[i][-1]=='LU':
                    pos.append(loguniform.rvs(self.parameters[i][0], self.parameters[i][1]-self.parameters[i][0], size=nwalkers))
                elif self.parameters[i][-1]=='TN':
                    mean, std ,low, high =   self.parameters[i][:-1]
                    low_n, high_n = (low - mean) / std, (high - mean) / std  # standardize
                    pos.append(truncnorm.rvs(low_n, high_n, loc=mean, scale=std,size=nwalkers))
                else:
                    ValueError('Prior distribution not recognized, should be "U", "N", "TN", "LU"')

            pos = np.vstack(pos).T
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, self.Probability, pool=pool
            )
            sampler.run_mcmc(pos, nsteps, progress=True)
            self.samples = sampler.get_chain(discard=nburnin, flat=True)
            # self.samples = sampler.chain[:, nburnin:, :].reshape((-1, ndim))
            print('Finish MCMC')
        elif 'dynesty' in self.sampler:
            if self.sampler == 'dynesty':
                if self.sampler_parameters is not None:
                    nlive = self.sampler_parameters['nlive']
                    dlogz = self.sampler_parameters['dlogz']
                print('Start dynesty')
                sampler = dynesty.NestedSampler(
                    self.Likelihood,
                    self.Prior,
                    ndim,
                    # bound = 'multi', sample='rwalk', pool = pool, queue_size=N_threads,\
                    bound = 'balls', sample='rwalk', pool = pool, queue_size=N_threads,\
                    nlive=nlive,  # number of live points
                )
                # print('The citations of %s:\n'%self.sampler,sampler.citations)
                sampler.run_nested(dlogz=dlogz,print_progress=True)
                print('Finish dynesty')
            elif self.sampler == 'dynamic_dynesty':
                if self.sampler_parameters is not None:
                    nlive_init = self.sampler_parameters['nlive_init']
                    dlogz_init = self.sampler_parameters['dlogz_init']
                print('Start dynamic dynesty')
                sampler = dynesty.DynamicNestedSampler(
                    self.Likelihood,
                    self.Prior,
                    ndim,
                    # bound = 'multi', sample='rwalk', pool = pool, queue_size=N_threads,\
                    bound = 'balls', sample='rwalk', pool = pool, queue_size=N_threads,\
                )
                # print('The citations of %s:\n'%self.sampler,sampler.citations)
                sampler.run_nested(nlive_init=nlive_init,dlogz_init=dlogz_init,print_progress=True)
                print('Finish dynamic dynesty')
            else:
                raise ValueError("Sampler not recognized")
            results = sampler.results
            samples = results['samples']
            weights = np.exp(results['logwt'] - results['logz'][-1])
            threshold = np.percentile(weights,15)
            valid = weights > threshold
            samples = dynesty.utils.resample_equal(samples, weights)
            self.samples = samples[valid]
            self.results = results
            
        else:
            raise ValueError("Sampler not recognized")
        end = time.time()
        total_time = end - start
        if self.multiprocessing == True:
            pool.close()
            print("Multiprocessing took {0:.1f} seconds".format(total_time))
        else:
            print("Serial took {0:.1f} seconds".format(total_time))

        self.parameters_pre = [(v[1], v[2]-v[1], v[1]-v[0]) for v in zip(*np.percentile(self.samples, [16, 50, 84],axis=0))]
        self.params = [x[0] for x in self.parameters_pre]
        self.getTmids()


    def getTmids(self):
        """
        Calculate the transit mid-times from the best-fit parameters.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        The transit mid-times are calculated using the best-fit parameters
        and are stored in the `data` attribute of the `TTVFIT` instance
        under the key `'T_mid_pre'`.

        """
        t_tras = self.Model(self.params)
        t_mids = []
        for i in range(len(self.data['Epoch'])):
            t_mids.append(t_tras[self.data['Epoch'][i]])
        self.data['T_mid_pre'] = np.array(t_mids,dtype=object)
    
    def checkValid(self):
        """
        Check for outliers in the data and reject them.

        Parameters
        ----------
        None

        Returns
        -------
        count : int
            The number of outliers rejected

        Notes
        -----
        The outliers are rejected based on the difference between the observed transit times and the predicted transit times using the current parameters.
        The outliers are marked as invalid in the data attribute of the class.
        """
        numbers = self.data[self.data['Valid']].index
        times = self.data[self.data['Valid']]['T_mid']
        times_pre = self.data[self.data['Valid']]['T_mid_pre']
        if self.rej_sigma is None:
            return 0
        rej = self.rej_sigma*np.abs(np.std(times_pre-times ,ddof=1))
        self.rejection = rej
        print("rej:", rej)
        count = 0
        for i in numbers:
            if abs(self.data.loc[i,'T_mid']-self.data.loc[i,'T_mid_pre']) >= rej:
                print("Rejecting point", i, "with times =", self.data.loc[i,'T_mid'], "and times_pre =", self.data.loc[i,'T_mid_pre'])
                self.data.loc[i,'Valid'] = False
                count += 1
        return count
            
    def runMethod(self,path=None):
        """
        Run the method until no more outliers are rejected.

        Parameters
        ----------
        path : str
            The path to save the results to. If None, the results are not saved.

        Returns
        -------
        None

        Notes
        -----
        The method is run until no more outliers are rejected. The results are then printed to the console and saved to a file if path is not None.
        The file contains the predicted parameters, the initial parameters, and the BIC.
        """
        self.Method()
        key = self.checkValid()
        all_keys = [key]
        print('\nRejection = ',self.rejection,'\n')
        while key>0:
            self.Method()
            key = self.checkValid()
            all_keys.append(key)
            # print('\nRejection = ',self.rejection,'\n')
            if (all_keys[-1]==all_keys[-2] or all_keys[-1]>15):
                break
        print('All keys = ',all_keys)

        if (self.kind!='Linear'):
            if self.lin_parameters is None:
                self.lin = TTVFit(self.data.copy(),self.period,parameters=self.parameters[:2],rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler)
                self.lin.Method()
            print('\nThe BIC_pre is',self.getBIC())


        else:
            print('\nThe BIC_pre Linear is',self.getBIC())
        print('Parameters_pre : \n','\n'.join(['%s = %s ' % item for item in zip(self.parameter_names,self.parameters_pre)]),'\n')
        if path != None:
            with open(path+'_parameters.txt','w') as f:
                f.write('# The predicted parameters are : (value, upper, lower)\n')
                f.write('\n'.join(['%s : %s ' % item for item in zip(self.parameter_names,self.parameters_pre)]))
                f.write('\nBIC : %s' %self.getBIC())
                f.write('\n# The initial parameters are : \n')
                f.write('\n'.join(['%s : %s ' % item for item in zip(self.parameter_names,self.parameters)]))

    def plotFit(self,path=None,ax=None,unitt='s'):
        """
        Plot the O-C diagram of the data with the given axis or a new one if it is None.
        
        Parameters
        ----------
        path : str
            The path to save the plot data to a csv file if it is not None.
        ax : matplotlib.axes.Axes
            The axes to plot to if it is not None.
        unitt : str
            The unit of the x-axis, either 's', 'min', or 'h'.
        
        Returns
        -------
        None
        
        Notes
        -----
        The O-C diagram is the difference between the observed and calculated transit times.
        The calculated transit times are calculated using the parameters from the Model method.
        The plot includes the O-C diagram for both the valid and invalid data.
        The valid data is marked with a '.' and the invalid data is marked with a 'x'.
        The plot also includes the BIC of the fit.
        The plot is saved to a file at the given path if it is not None.
        """
        self.emax = self.data['Epoch'].max()//100*100+100
        self.unitt = unitt
        if self.unitt == 's':
            multi = 24*60*60
        elif self.unitt == 'min':
            multi = 24*60
        elif self.unitt == 'h':
            multi = 24
        notValid = ~self.data['Valid']

        cmap = mpl.colormaps['rainbow']
        colors = cmap(np.linspace(0, 1, len(set(self.data['Label']))))
        color_dict = dict([[x,y] for x,y in zip(set(self.data['Label']),colors)])
        
        if ax ==None:
            fig,ax = plt.subplots(figsize = (20,4))
        if self.kind == 'Linear':
            for i,v in enumerate(self.data['Valid']):
                if v:
                    ax.errorbar(self.data.loc[i,'Epoch'], (self.data.loc[i,'T_mid']-self.data.loc[i,'T_mid_pre'])*multi, yerr=self.data.loc[i,'Uncertainty']*multi, fmt='.',label=self.data.loc[i,'Label'],color=color_dict[self.data.loc[i,'Label']],ms=5,ecolor='dimgray',elinewidth=1,capsize=3)
            ax.errorbar(self.data[notValid]['Epoch'], (self.data[notValid]['T_mid']-self.data[notValid]['T_mid_pre'])*multi, yerr=self.data[notValid]['Uncertainty']*multi, fmt='.',ms=5,ecolor='dimgrey',color='silver',alpha=0.5,elinewidth=1,capsize=3)
            ax.axhline(0, c='k', lw=1, ls='--')
            ax.fill_between(self.data[self.data['Valid']]['Epoch'], (self.rejection)*multi,(-self.rejection)*multi, alpha=0.3, color='lightskyblue', label=r'%d$\sigma$ area'%self.rej_sigma)
            self.data.to_csv(path+'_plot_data.csv',index=False)
            
        else:
            t_pre = np.array(self.Model(self.params))
            if self.lin_parameters is None:
                t_pre_l = np.array(self.lin.Model(self.lin.params))
                data_lin_pre = np.array(self.lin.data['T_mid_pre'])
            else:
                t_pre_l = np.array(self.linearModel(self.lin_parameters))
                t_tras_lin = self.linearModel(self.lin_parameters)
                t_mids_lin = []
                for i in range(len(self.data['Epoch'])):
                    t_mids_lin.append(t_tras_lin[self.data['Epoch'][i]])
                data_lin_pre = np.array(t_mids_lin)
            ax.plot(list(range(0,self.emax)),(t_pre[0:self.emax]-t_pre_l[0:self.emax])*multi,color='royalblue')
            for i,v in enumerate(self.data['Valid']):
                if v:
                    ax.errorbar(self.data.loc[i,'Epoch'], (self.data.loc[i,'T_mid']-data_lin_pre[i])*multi, yerr=self.data.loc[i,'Uncertainty']*multi, fmt='.',label=self.data.loc[i,'Label'],color=color_dict[self.data.loc[i,'Label']],ms=5,ecolor='dimgray',elinewidth=1,capsize=3)
            ax.errorbar(self.data[notValid]['Epoch'], (self.data[notValid]['T_mid']-data_lin_pre[notValid])*multi, yerr=self.data[notValid]['Uncertainty']*multi, fmt='.',ms=5,ecolor='dimgrey',color='silver',alpha=0.5,elinewidth=1,capsize=3)
            ax.axhline(0, c='k', lw=1, ls='--')
            top=t_pre[0:self.emax]-t_pre_l[0:self.emax]+self.rejection
            down=t_pre[0:self.emax]-t_pre_l[0:self.emax]-self.rejection
            ax.fill_between(list(range(0,self.emax)), top*multi,down*multi, alpha=0.3,color='lightskyblue', label=r'%d$\sigma$ area'%self.rej_sigma)
            self.data.to_csv(path+'_plot_data.csv',index=False)
            with open(path+'_plot_model_data.csv','w') as f:
                f.write('# rej_sigma = %s\n'%self.rej_sigma)
                f.write('# Rejection = %s\n'%self.rejection)
                f.write('Epoch,T_pre,T_pre_l,Residual,Top,Down\n')
                for i in range(0,self.emax):
                    f.write('%s,%s,%s,%s,%s,%s\n'%(i,t_pre[i],t_pre_l[i],t_pre[i]-t_pre_l[i],top[i],down[i]))

        ax.tick_params(direction='in', which='both', labelsize=12)
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.set_ylabel("O - C [%s]"%unitt,fontsize=18)
        ax.set_xlabel("Epoch",fontsize=18)
        patches, labels = ax.get_legend_handles_labels()
        lh_dict = dict([[x,y] for x,y in zip(labels,patches)])
        patches, labels = list(lh_dict.values()), list(lh_dict.keys())
        ax.legend(patches, labels)
        if ax ==None:
            plt.tight_layout()
            if path != None:
                plt.savefig(path+'_plot.png',dpi=300)
                plt.savefig(path+'_plot.pdf',dpi=600)
            plt.show()

    def test(self, outdir, suffix):
        """
        Perform Leave-One-Out Cross Validation (LOOCV) on the given data.

        Parameters
        ----------
        outdir : str
            The directory to save the plots.
        suffix : str
            The suffix to append to the filename.

        Notes
        -----
        The method will run the LOOCV test on the given data, and save the plots in the specified directory.
        The BIC values for the LOOCV test are calculated using the given data.
        The method will also save the LOOCV data in a csv file with the filename `loocv_data%s.csv` in the specified directory.

        Returns
        -------
        None
        """
        BIC0s = []
        BICqs = []
        Params_list = []
        Away_list = []
        LastParam_list = []
        LastUncertainty_list = []
        deltaBICs = []
        for i in range(len(self.data['Epoch'])):
            select = np.ones_like(self.data['Epoch'],dtype=bool)
            select[i] = False
            fig,ax = plt.subplots(2,1,figsize = (20,6),sharex=True)
            data = self.data[select].copy().reset_index(drop=True)
            times0 = data['T_mid'][0]
            times0_est = [times0 - 0.05, times0 + 0.05,'U']
            parameters = [times0_est]
            parameters += self.parameters[1:]
            # lin = TTVFit(data.copy(),self.period,parameters[:2],self.parameter_names[:2],'Linear',rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler,sampler_parameters=self.sampler_parameters)
            lin = TTVFit(data.copy(),self.period,parameters[:2],self.parameter_names[:2],'Linear',rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler)
            lin.runMethod(path=outdir+'test_lin%s_select%d'%(suffix,i))
            lin.plotFit(path=outdir+'test_lin%s_select%d'%(suffix,i),ax=ax[1])
            # main = TTVFit(data.copy(),self.period,parameters,self.parameter_names,self.kind,rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler,sampler_parameters=self.sampler_parameters,lin_parameters=lin.params)
            BIC0 = lin.getBIC()
            main = TTVFit(data.copy(),self.period,parameters,self.parameter_names,self.kind,rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler,lin_parameters=lin.params)
            main.runMethod(path=outdir+'test_main%s_select%d'%(suffix,i))
            main.plotFit(path=outdir+'test_main%s_select%d'%(suffix,i),ax=ax[0])
            BICq = main.getBIC()
            BIC0s.append(BIC0)
            BICqs.append(BICq)
            Params_list.append(main.parameters_pre)
            LastParam_list.append(main.parameters_pre[-1][0])
            LastUncertainty_list.append(max(main.parameters_pre[-1][1],main.parameters_pre[-1][2]))
            Away_list.append(abs(main.parameters_pre[-1][0])-3*abs(max(main.parameters_pre[-1][1],main.parameters_pre[-1][2]))>0)
            deltaBICs.append(BIC0 - BICq)
            ax[0].set_title(r'LOOCV Test (Select %d, $\Delta BIC$ = %.3f)'%(i,BIC0-BICq))
            plt.tight_layout()
            plt.subplots_adjust(hspace=0)
            plt.savefig(outdir+'test%s_select%d.png'%(suffix,i),dpi=300)
            plt.savefig(outdir+'test%s_select%d.pdf'%(suffix,i),dpi=600)
            plt.show()
            print('BIC0 = %.3f, BICq = %.3f'%(BIC0,BICq))

            fig = plt.figure(figsize=(3.5*len(main.parameters),3.5*len(main.parameters)))
            corner.corner(main.samples, labels=[x for x in main.parameter_names],truths=main.params,fig=fig,show_titles=True,title_fmt='g',use_math_text=True)
            plt.tight_layout()
            plt.savefig(outdir+'test_corner%s_select%d.png'%(suffix,i),dpi=300)
            plt.savefig(outdir+'test_corner%s_select%d.pdf'%(suffix,i),dpi=600)
            plt.show()
        self.loocv_data = self.data.copy()
        self.loocv_data.drop('Valid',axis=1,inplace=True)
        self.loocv_data['BIC0'] = BIC0s
        self.loocv_data['BICq'] = BICqs
        self.loocv_data['Params'] = Params_list
        self.loocv_data['LastParameter'] = LastParam_list
        self.loocv_data['LastUncertainty'] = LastUncertainty_list
        self.loocv_data['3sigmaAway'] = Away_list
        self.loocv_data['deltaBIC'] = deltaBICs
        self.loocv_data.to_csv(outdir+'loocv_data%s.csv'%suffix,index=False)

    def Fit(self,outdir,suffix,title=None):
        """
        Perform a MCMC fit to the data.

        Parameters
        ----------
        outdir : str
            The directory where the output files are saved.
        suffix : str
            The suffix of the output files.
        title : str or None
            The title of the figure. If None, no title is given.

        Returns
        -------
        samples : 2D array
            The samples from the MCMC chain.

        Notes
        -----
        This method performs a MCMC fit to the data and saves the results to outdir.
        The prior parameters are saved in a file named prior_parameters.txt.
        The MCMC samples are saved in a file named samples.npy.
        The figure of the MCMC results is saved in a file named corner.pdf.
        The figure of the best fit model is saved in a file named Fit.pdf.
        """
        folder = os.path.exists(outdir)
        if not folder:
            os.makedirs(outdir)
        
        with open(outdir+'prior_parameters%s.txt'%suffix,'w') as f:
            f.write('period = '+str(self.period)+'\n')
            f.write('parameters = '+str(self.parameters)+'\n')
            f.write('parameter_names = '+str(self.parameter_names)+'\n')
            f.write('kind = '+str(self.kind)+'\n')
            f.write('rej_sigma = '+str(self.rej_sigma)+'\n')
            f.write('sampler = '+str(self.sampler)+'\n')
            f.write('sampler_parameters = '+str(self.sampler_parameters)+'\n')

        print('Fitting begins...')
        self.main = TTVFit(self.data.copy(),self.period,self.parameters,self.parameter_names,self.kind,rej_sigma=self.rej_sigma,multiprocessing=self.multiprocessing,limit=self.limit,sampler=self.sampler,sampler_parameters=self.sampler_parameters,lin_parameters=self.lin_parameters)
        self.main.runMethod(path=outdir+'Fit%s'%suffix)
        self.BIC = self.main.getBIC()
        self.emax = self.data['Epoch'].max()//100*100+100
        self.last_param = self.main.parameters_pre[-1][0]
        self.last_uncertainty = max(self.main.parameters_pre[-1][1],self.main.parameters_pre[-1][2])
        print('Fitting ends...')
        np.save(outdir+'samples%s.npy'%suffix,self.main.samples)

        fig = plt.figure(figsize=(3.5*len(self.main.parameters),3.5*len(self.main.parameters)))
        corner.corner(self.main.samples, labels=[x for x in self.main.parameter_names],truths=self.main.params,fig=fig,show_titles=True,title_fmt='g',use_math_text=True)
        plt.tight_layout()
        plt.savefig(outdir+'corner%s.png'%suffix,dpi=300)
        plt.savefig(outdir+'corner%s.pdf'%suffix,dpi=600)
        plt.show()

        fig,ax = plt.subplots(figsize = (20,6))
        self.main.plotFit(path=outdir+'Fit%s'%suffix,ax=ax)
        if title != None:
            params_equation = '$'+r' = %.6f$\pm$%.6f ['%(self.main.parameters_pre[-1][0],max(self.main.parameters_pre[-1][1],self.main.parameters_pre[-1][2]))+'$'
            params_equation = params_equation.join(self.parameter_names[-1].split('['))
            if self.kind != 'Linear':
                ax.set_title(title+' '+r'(%s, $BIC = $%.3f)'%(params_equation, self.BIC), fontsize=23)
            else:
                ax.set_title(title+' '+r'($BIC = $%.3f)'%(self.BIC), fontsize=23)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(outdir+'Fit%s.png'%suffix,dpi=300)
        plt.savefig(outdir+'Fit%s.pdf'%suffix,dpi=600)
        plt.show()
        if self.sampler == 'emcee':
            return self.main.samples
        else:
            return self.main.samples, self.main.results
            
    
    def loocv(self,outdir,suffix,param=None,title=None):
        """
        Perform a Leave-One-Out Cross Validation (LOOCV) test to the data.

        Parameters
        ----------
        outdir : str
            The directory where the output files are saved.
        suffix : str
            The suffix of the output files.
        param : list or None
            The last parameter of the fit and its uncertainty. If None, the last parameter of the fit is not shown.
        title : str or None
            The title of the figure. If None, no title is given.

        Notes
        -----
        This method performs a LOOCV test to the data and saves the results to outdir.
        The prior parameters are saved in a file named prior_parameters.txt.
        The figure of the LOOCV results is saved in a file named Test.pdf.
        """
        folder = os.path.exists(outdir) 
        if not folder:
            os.makedirs(outdir)

        with open(outdir+'prior_parameters%s.txt'%suffix,'w') as f:
            f.write('period = '+str(self.period)+'\n')
            f.write('parameters = '+str(self.parameters)+'\n')
            f.write('parameter_names = '+str(self.parameter_names)+'\n')
            f.write('kind = '+str(self.kind)+'\n')
            f.write('rej_sigma = '+str(self.rej_sigma)+'\n')
            f.write('sampler = '+str(self.sampler)+'\n')
            f.write('sampler_parameters = '+str(self.sampler_parameters)+'\n')

        print('LOOCV Test begins...')
        self.test(outdir,suffix)
        print('LOOCV Test ends...')
        fig,ax = plt.subplots(2,1,figsize = (20,6),sharex=True)
        self.emax = self.data['Epoch'].max()//100*100+100
        if param != None:
            self.last_param = param[0]
            if len(param)>2:
                self.last_uncertainty = max(param[1],param[2])
            else:
                self.last_uncertainty = param[1]
        ax[0].plot(list(range(self.emax)),[self.last_param]*self.emax,'-.', color='gray')
        ax[0].fill_between(list(range(0,self.emax)), [self.last_param+self.last_uncertainty]*self.emax,[self.last_param-self.last_uncertainty]*self.emax, alpha=0.3,color='lightgreen')
        ax[0].errorbar(self.loocv_data[self.loocv_data['3sigmaAway']==True]['Epoch'],self.loocv_data[self.loocv_data['3sigmaAway']==True]['LastParameter'],self.loocv_data[self.loocv_data['3sigmaAway']==True]['LastUncertainty'], fmt='s', color='lightskyblue', ecolor='dimgray',ms=5,elinewidth=1,capsize=3)
        ax[0].errorbar(self.loocv_data[self.loocv_data['3sigmaAway']==False]['Epoch'],self.loocv_data[self.loocv_data['3sigmaAway']==False]['LastParameter'],self.loocv_data[self.loocv_data['3sigmaAway']==False]['LastUncertainty'], fmt='d', color='lightpink', ecolor='dimgray',ms=5,elinewidth=1,capsize=3)

        ax[0].axhline(0, c='k', lw=1, ls='--')
        ax[0].tick_params(direction='in', which='both', labelsize=12)
        ax[0].xaxis.set_ticks_position('both')
        ax[0].yaxis.set_ticks_position('both')
        ax[0].set_ylabel(self.parameter_names[-1],fontsize=18)

        ax[1].plot(list(range(self.emax)),[10]*self.emax,'-.', color='lightpink')
        ax[1].plot(self.loocv_data['Epoch'],self.loocv_data['deltaBIC'],'-o', color='lightskyblue')
        ax[1].set_xlabel("Epoch",fontsize=18)
        ax[1].set_ylabel(r"$\Delta BIC$",fontsize=18)
        ax[1].tick_params(direction='in', which='both', labelsize=12)
        ax[1].xaxis.set_ticks_position('both')
        ax[1].yaxis.set_ticks_position('both')
        if title != None:
            ax[0].set_title(title+' LOOCV Test', fontsize=23)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.savefig(outdir+'Test%s.png'%suffix,dpi=300)
        plt.savefig(outdir+'Test%s.pdf'%suffix,dpi=600)
        plt.show()