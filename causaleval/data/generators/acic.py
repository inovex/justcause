import pandas as pd
import numpy as np

from causaleval.data.data_provider import DataProvider
from causaleval import config

import scipy
from sklearn.preprocessing import StandardScaler, minmax_scale, RobustScaler

# Playground imports
import seaborn as sns
import matplotlib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from causaleval.data.sets.ibm import SimpleIBMDataProvider


# To make it work on MacOS
import matplotlib
matplotlib.use("MacOSX")

import seaborn as sns
sns.set(style="darkgrid")

import matplotlib.pyplot as plt


class ACICGenerator(DataProvider):

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ACIC"

    def get_true_ite(self, data=None):
        super().get_true_ite(data)

    def get_train_generator_batch(self, batch_size=32):
        pass

    def get_training_data(self, size=None):
        super().get_training_data(size)

    def get_train_generator_single(self, random=False, replacement=False):
        pass

    def get_true_ate(self, subset=None):
        super().get_true_ate(subset)

    def get_num_covariates(self):
        pass

    def get_info(self):
        super().get_info()

    def get_test_data(self):
        super().get_test_data()

    def generate_data(self, param_dict):

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


        def treatment_assignment(covariates, num_parents=10, relation='weak', use_parents=None, *args):
            """

            :param covariates: covariate matrix of shape (n, k) with n instances and k features
            :param num_parents: number of parents on which treatment assignment should depend
            :param relation: strength of the relationship from parents to treatment
            :param use_parents: use predefined set of parent covariates
                parents should be the subset of the covariate matrix
            :return: binary treatment vector
            """
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
                    func = np.array(list(map(exponential, confounders)))
                    exp_poly = scipy.special.expit(RobustScaler().fit_transform(func.reshape(-1,1)).reshape(1, -1)[0])
                    return np.random.binomial(1, p=exp_poly)
                if relation == 'strong':
                    # TODO: Create a deterministic split based on a few covariates that results in a 50/50
                    # partition of the data --> Feature selection
                    sex = covariates[:, self.idx_dict['sex']]
                    return sex > 0 # Treatment only to one gender --> creates equal split with no overlap



        def treatment_effect(covariates, relation='weak', homogeneous=True, num_parents=10, use_parents=None, *args):
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
            if homogeneous:
                return np.full(len(covariates), 1.5)
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
                    confounders = minmax_scale(confounders, feature_range=(-1.5,1.5))
                    exp_poly = scipy.special.expit(np.array(list(map(normal_polynomial, confounders))))
                    return (exp_poly > 0.5).astype(int)

        def outcome_assignment(covariates,
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
            ite = treatment_effect(covariates,
                                         homogeneous=homogeneous,
                                         use_parents=confounders,
                                         relation=relation)

            y_1 = y_0 + ite
            return np.c_[y_1, y_0] # TODO: Change the order here and everywhere used

        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        covariates_df = covariates_df.drop(columns=['sample_id'])
        covariates = covariates_df.values

        interesting_columns = ['sex', 'md_route', 'dbwt', 'estgest']
        self.idx_dict = { name : covariates_df.columns.get_loc(name) for name in interesting_columns}

        self.param_dict = param_dict

        ids = np.random.choice(covariates.shape[1], size=10)
        confounders = covariates[:, ids]

        def test_confounding(x, y, y_cf, t):
            treated = y[t==1]
            control = y[t==0]
            simple_mean = np.mean(treated) - np.mean(control)
            print('simple: ' + str(simple_mean))


        def test_generation(random, homogeneous, conf=False, strength='weak'):
            np.random.seed(param_dict['seed'])
            relation = 'random' if random else 'weak'
            outcome_relation = strength

            if strength is 'strong':
                relation = 'strong'

            if conf:
                # Use strong pre-selected values
                ids = np.random.choice(covariates.shape[1], size=6)
                ids = np.concatenate((ids, list(self.idx_dict.values())), axis=0)
                parents = covariates[:, ids]
            else:
                parents = None

            # Sample data
            t = treatment_assignment(covariates, relation=relation, use_parents=parents)
            ys = outcome_assignment(covariates, constant_base=False, homogeneous=homogeneous, relation=outcome_relation, use_parents=parents)
            y_cf = np.array([row[int(ix)] for row, ix in zip(ys, t)])
            y = np.array([row[int(1 - ix)] for row, ix in zip(ys, t)])

            test_confounding(covariates, y, y_cf, t)
            print('true ', np.mean(ys[:, 0] - ys[:, 1]))

            # Plot distributions
            sns.distplot(y, color='red')
            sns.distplot(y_cf, color='green')
            sns.distplot(ys[:,0] - ys[:,1], color='blue')
            plt.show()




        def make_file_name(random, homogeneous, strength=None, conf=None):
            file = config.GENERATE_PATH
            file += 'random' if random else 'observ'
            file += '_homo_full' if homogeneous else '_hetero_full'
            if strength is 'strong':
                file += '_strong'
            if conf:
                file += '_conf'
            return file


        def generate_file(random, homogeneous, conf=False, strength='weak'):
            np.random.seed(param_dict['seed'])
            relation = 'random' if random else 'weak'

            if strength is 'strong':
                relation = 'strong'

            if conf:
                ids = np.random.choice(covariates.shape[1], size=10)
                parents = covariates[:, ids]
            else:
                parents = None

            # Sample data
            t = treatment_assignment(covariates, relation=relation, use_parents=parents)
            ys = outcome_assignment(covariates, constant_base=False, homogeneous=homogeneous, relation=relation, use_parents=parents)
            y = np.array([row[int(ix)] for row, ix in zip(ys, t)])
            y_cf = np.array([row[int(1 - ix)] for row, ix in zip(ys, t)])

            # Create Dataframes
            factual = pd.DataFrame(data=np.array([t, y]).T, columns=['t', 'y'])
            counterfactual = pd.DataFrame(data=np.array([t, y, y_cf, ys[:, 1], ys[:, 0]]).T, columns=['t', 'y', 'y_cf', 'y_0', 'y_1'])

            # Write file
            file_stub = make_file_name(random, homogeneous, strength, conf)
            factual_file = file_stub + '.csv'
            counterfactual_file = file_stub + '_CF.csv'
            factual.to_csv(factual_file)
            counterfactual.to_csv(counterfactual_file)

        test_generation(random=True, homogeneous=False)
        # generate_file(random=True, homogeneous=True)
        # generate_file(random=True, homogeneous=False)
        # generate_file(random=False, homogeneous=False)
        # generate_file(random=False, homogeneous=True)
        # generate_file(random=False, homogeneous=False, conf=True)
        # generate_file(random=False, homogeneous=False, strength='strong')


    def generate_data_powers(self, param_dict):
        # DGP after Powers et al.
        # All f's get 10 Standard normalized variables as input

        def f1(vars):
            return np.zeros(vars.shape[0])

        def f2(vars):
            # TODO: Select id=0 to be a variable that has variance within in this function
            return 5*(vars[:, 0] > 0) - 5

        def f3(vars):
            return 2*vars[:, 1] - 4

        def f4(vars):
            # high interactions
            return 2*vars[:, 0]*vars[:, 3]*vars[:, 5] + vars[:, 0]*vars[:, 3]*(1-vars[:, 5]) + 3*vars[:, 7]*vars[:, 8]
            + 4*vars[:, 6]*(1- vars[:, 9])*(1-vars[:, 0])

        def f5(vars):
            return np.sum(vars, axis=1) - 4

        def f6(vars):
            return 4*(vars[:, 0] > 0.8) + (vars[:, 2]>0) + 4*(vars[:, 3])*(vars[:, 8]) + 2*vars[:, 7]*vars[:, 8]

        def f7(vars):
            squares = np.power(vars[:, ::2], 2)
            normal = vars[:, 1::2]
            return 0.5*(np.sum(squares, axis=1) + np.sum(normal, axis=1))

        def f8(vars):
            return (1/np.sqrt(2))*(f4(vars) + f5(vars))

        f_list = [f1, f2, f3, f4, f5, f6, f7, f8]
        covariates_df = pd.read_csv(config.IBM_PATH_ROOT + '/' + 'covariates.csv')
        covariates_df = covariates_df.loc[:, covariates_df.var() > 0.3]
        covariates = covariates_df.drop(columns=['sample_id']).values







if __name__ == '__main__':

    a = ACICGenerator()
    a.generate_data({'seed': 10})





