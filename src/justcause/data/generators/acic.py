import os

import pandas as pd
import numpy as np

from .generator import DataGenerator
from ...utils import surface_plot, simple_comparison_mean

import scipy
from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler

# To make it work on MacOS
from mpl_toolkits.mplot3d import Axes3D
import matplotlib


matplotlib.use(config.PLOT_BACKEND)
import seaborn as sns
sns.set(style="darkgrid")
import matplotlib.pyplot as plt


# ToDo: Avoid this workaround
import config

np.random.seed(0)  # make sure to fix the seed for replication
exp_coeffs = np.random.normal(loc=0, scale=0.5, size=10)


def normal_polynomial(vars):
    """
    Calculate the averaged value of linear function with normal sampled coefficients for the input vars
    Usually returns results around 0, if inputs are normalized
    :param vars:
    :return:
    """
    mult = exp_coeffs*vars
    return np.sum(mult)/len(vars)

def exponential(vars):

    mult = exp_coeffs*vars
    return np.sum(np.exp(mult))

def random_poly(vars):
    coeffs = np.random.randn(len(vars))
    mult = coeffs*vars
    return np.sum(mult)

def interaction_poly(vars):
    """Simple multiplicative interaction between all covariates"""
    prod = np.prod(vars)
    return prod

def interaction_poly_with_coeffs(vars):
    coeffs = np.random.normal(loc=1, scale=0.5, size=len(vars))
    mult = coeffs*vars
    return np.prod(mult)

def f4(vars):
    # high interactions
    return 2*vars[0]*vars[3]*vars[5] + vars[0]*vars[3]*(1-vars[5]) + 3*vars[7]*vars[8] \
           + 4*vars[6]*(1- vars[9])*(1-vars[0])


class ACICGenerator(DataGenerator):

    def __init__(self, params):
        """

        :param params: dict containing 'random', 'homogeneous', 'deterministic', 'confounded'
        """
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        self.covariates_df = covariates_df.drop(columns=['sample_id'])
        self.covariates = covariates_df.values
        self.idx_dict = { name : self.covariates_df.columns.get_loc(name) for name in config.ACIC_SELECTED_VALUES}
        self.x = self.covariates
        super().__init__(params)


    def __str__(self):
        return self.make_data_name()

    def get_true_ite(self, data=None):
        return self.y_1 - self.y_0

    def load_training_data(self):
        # Get the counterfactual file with all the information
        fname = self.make_file_name(self.random, self.homogeneous, self.deterministic, self.confounded, counterfactual=True)

        if not os.path.isfile(fname):
            self.generate_file(self.random, self.homogeneous, self.confounded, self.deterministic)

        dataframe = pd.read_csv(fname)
        self.x = self.covariates
        self.t = dataframe['t'].values
        self.y = dataframe['y'].values
        self.y_cf = dataframe['y_cf'].values
        self.y_0 = dataframe['y_0'].values
        self.y_1 = dataframe['y_1'].values


    def get_num_covariates(self):
        return self.x.shape[1]

    def make_data_name(self):
        name = 'ACIC-'
        if self.random:
            name += 'rand-'

        if self.homogeneous:
            name += 'homo-'
        else:
            name += 'hetero-'

        if self.deterministic:
            name += 'det-'

        if self.confounded:
            name += 'conf-'

        return name


    @staticmethod
    def make_file_name(random, homogeneous, deterministic=False, conf=None, counterfactual=False):
        file = config.GENERATE_PATH
        file += 'random' if random else 'observ'
        file += '_homo' if homogeneous else '_hetero'
        if deterministic:
            file += '_det'
        if conf:
            file += '_conf'
        if counterfactual:
            file += '_cf'
        return file + '.csv'


    def treatment_assignment(self, covariates, num_parents=10, relation='weak', use_parents=None, *args):
        """

        :param covariates: covariate matrix of shape (n, k) with n instances and k features
        :param num_parents: number of parents on which treatment assignment should depend
        :param relation: strength of the relationship from parents to treatment
        :param use_parents: use predefined set of parent covariates
            parents should be the subset of the covariate matrix
        :return: binary treatment vector
        """
        np.random.seed(self.params['seed']) # make sure to fix the seed for replication
        if relation == 'random':
            # Completely random treatment assignment
            return np.random.random_integers(0,1, size=covariates.shape[0])
        else:
            # Confounded treatment assignment
            if use_parents is not None:
                confounders = minmax_scale(use_parents, feature_range=(0,10))
            else:
                ids = np.random.choice(covariates.shape[1], size=num_parents)
                confounders = minmax_scale(covariates[:, ids], feature_range=(0,10))

            if relation == 'weak':
                confounders = RobustScaler().fit_transform(confounders)
                func = np.array(list(map(normal_polynomial, confounders)))
                exp_poly = scipy.special.expit(RobustScaler().fit_transform(func.reshape(-1,1)).reshape(1, -1)[0])
                return np.random.binomial(1, p=exp_poly)
            if relation == 'strong':
                weight = covariates[:, self.idx_dict['dbwt']]
                treat = weight < np.median(weight) # Treatment only to low birth-weight samples
                return (treat + np.random.binomial(1, 0.1, size=len(treat)) > 0)

    def treatment_effect(self, covariates, relation='weak', homogeneous=True, num_parents=10, use_parents=None, *args):
        """Generate treatment effect based on subset of covariates

        Currently, the function is called with the parents selected in outcome_assignment(), thus use_parents
        always contains the relevant covariate-submatrix

        :param covariates: covariate matrix of shape (n, k) with n instances and k features
        :param relation: strength of the relationship from parents to treatment effect (strength of confounding)
        :param homogeneous: Whether or not treatment effect should be homogeneous / equal for all instances
        :param num_parents: number of parents on which outcome generation should depend
        :param use_parents: use predefined set of parent covariates
            parents should be the subset of the covariate matrix
        :return: treatment effect vector
        """
        np.random.seed(self.params['seed']) # make sure to fix the seed for replication
        if homogeneous:
            return np.full(len(covariates), 1.5) + np.random.normal(0, 0.1, size=len(covariates))
        else:
            if use_parents is not None:
                # Use unnormalized features to get more significant treatment effects
                confounders = use_parents
            else:
                ids = np.random.choice(covariates.shape[1], size=num_parents)
                confounders = covariates[:, ids]

            if relation == 'weak':
                confounders = RobustScaler().fit_transform(confounders)
                effect = np.array(list(map(normal_polynomial, confounders)))
                return minmax_scale(effect.reshape(-1, 1), feature_range=(1,2)).reshape(1, -1)[0]
            elif relation == 'strong':
                # Relation is deterministic and treatment effect is binary (either 1 or 0)
                # confounders = minmax_scale(confounders, feature_range=(-1.5,1.5))
                # exp_poly = scipy.special.expit(np.array(list(map(normal_polynomial, confounders))))

                weight = RobustScaler().fit_transform(covariates[:, self.idx_dict['dbwt']].reshape(-1,1)).reshape(1, -1)[0]
                return (weight < 0)+1

    def outcome_assignment(self,
                           covariates,
                           num_parents=10,
                           constant_base=False,
                           relation='weak',
                           use_parents=None,
                           homogeneous=True,
                           *args):
        """

        :param covariates: covariate matrix of shape (n, k) with n instances and k features
        :param num_parents: number of parents on which outcome generation should depend
        :param constant_base: Whether the y_0 surface ought to be constant
        :param relation: strength of the relationship from parents to treatment
        :param use_parents: use predefined set of parent covariates
            parents should be the subset of the covariate matrix
        :param homogeneous: Whether or not treatment effect should be homogeneous / equal for all instances
        :return: binary treatment vector
        """
        np.random.seed(self.params['seed']) # make sure to fix the seed for replication
        if use_parents is not None:
            confounders = use_parents
        else:
            ids = np.random.choice(covariates.shape[1], size=num_parents)
            confounders = covariates[:, ids]

        if constant_base:
            # Pathetic special case
            y_0 = np.zeros(covariates.shape[0])
        else:
            confounders_outcome = minmax_scale(confounders, feature_range=(1,2))
            y_0 = np.array(list(map(exponential, confounders_outcome)))
            y_0 = minmax_scale(y_0.reshape(-1,1), feature_range=(1,2)).reshape(1, -1)[0]

        # Use same confounders for treatment effect and outcome generation
        ite = self.treatment_effect(covariates,
                                     homogeneous=homogeneous,
                                     use_parents=confounders,
                                     relation=relation)

        y_1 = y_0 + ite
        return np.c_[y_1, y_0] # TODO: Change the order here and everywhere used


    def generate_data(self, random, homogeneous, confounded=False, deterministic=False):
        """Generate data from simplified parameters

        :param random: Whether treatment assignment should be random
        :param homogeneous: treatment effect homogeneous or not
        :param confounded: confounders present or not
        :param deterministic: Whether assignment ought to be deterministic
            deterministic assignment results in a no-overlap condition
        :return: t, ys, y, y_cf
            treatment vector, both outcomes, observed outcomes according to treatment, counterfactual outcomes
        """
        np.random.seed(self.params['seed']) # make sure to fix the seed for replication

        relation = 'random' if random else 'weak'
        outcome_relation = 'weak'

        if deterministic:
            # deterministic relates to both outcome and treatment assignment
            relation = 'strong'
            outcome_relation = 'strong'

        if confounded:
            # Use strong pre-selected values
            ids = np.random.choice(self.covariates.shape[1], size= 10 - len(self.idx_dict))
            ids = np.concatenate((ids, list(self.idx_dict.values())), axis=0)
            parents = self.covariates[:, ids]
        else:
            parents = None

        # Sample data
        t = self.treatment_assignment(self.covariates, relation=relation, use_parents=parents)
        ys = self.outcome_assignment(self.covariates, constant_base=False, homogeneous=homogeneous, relation=outcome_relation, use_parents=parents)
        y = np.array([row[int(1 - ix)] for row, ix in zip(ys, t)])
        y_cf = np.array([row[int(ix)] for row, ix in zip(ys, t)])

        return t, ys, y, y_cf
    def generate_file(self, random, homogeneous, confounded=False, deterministic=False):
        t, ys, y, y_cf = self.generate_data(random, homogeneous, confounded, deterministic)

        # Create Dataframes
        factual = pd.DataFrame(data=np.array([t, y]).T, columns=['t', 'y'])
        counterfactual = pd.DataFrame(data=np.array([t, y, y_cf, ys[:, 1], ys[:, 0]]).T, columns=['t', 'y', 'y_cf', 'y_0', 'y_1'])

        # Write file
        factual_file = self.make_file_name(random, homogeneous, deterministic, confounded, counterfactual=False)
        counterfactual_file = self.make_file_name(random, homogeneous, deterministic, confounded, counterfactual=True)
        factual.to_csv(factual_file)
        counterfactual.to_csv(counterfactual_file)

    def test_generation(self, random, homogeneous, confounded=False, deterministic=False):
        t, ys, y, y_cf = self.generate_data(random, homogeneous, confounded, deterministic)

        simple_comparison_mean(y, t)
        print('true ', np.mean(ys[:, 0] - ys[:, 1]))

        choice = np.random.choice(len(self.x), size=1000)
        surface_plot(self.y_1[choice], self.y_0[choice], self.y[choice], self.y_cf[choice], self.x[choice])


    def generate_all_files(self):

        self.generate_file(random=True, homogeneous=True)
        self.generate_file(random=True, homogeneous=False)
        self.generate_file(random=False, homogeneous=False) # Confounders possible
        self.generate_file(random=False, homogeneous=True)
        self.generate_file(random=False, homogeneous=False, confounded=True) # Confounders guaranteed
        self.generate_file(random=False, homogeneous=False, deterministic=True)


if __name__ == '__main__':

    dict = {
        'random' : False,
        'deterministic': False,
        'homogeneous' : False,
        'confounded' : True,
        'seed' : 0
    }

    import utils
    # a = ACICGenerator(dict)
    # a.test_generation(random=False, homogeneous=True)
    twins = ACICGenerator(dict)

    choice = np.random.choice(len(twins.x), size=1000)

    utils.surface_plot(twins.y_1[choice], twins.y_0[choice], twins.y[choice], twins.y_cf[choice], twins.x[choice],name='ACIC-hetero-conf')
    utils.plot_y_dist(twins.y[choice], twins.y_cf[choice], method_name='ACIC-hetero-conf')
    utils.ite_plot(twins.y_1[choice], twins.y_0[choice], method_name='ACIC-hetero-conf')




