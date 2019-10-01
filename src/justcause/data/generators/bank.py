import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale, StandardScaler, RobustScaler

from ..data_provider import DataProvider
from ... import utils

import config


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


## CLEANING CODE FROM: https://www.kaggle.com/aleksandradeis/bank-marketing-analysis

def get_dummy_from_bool(row, column_name):
    ''' Returns 0 if value in column_name is no, returns 1 if value in column_name is yes'''
    return 1 if row[column_name] == 'yes' else 0


def get_correct_values(row, column_name, threshold, df):
    ''' Returns mean value if value in column_name is above threshold'''
    if row[column_name] <= threshold:
        return row[column_name]
    else:
        mean = df[df[column_name] <= threshold][column_name].mean()
        return mean


def clean_data(df):
    '''
    INPUT
    df - pandas dataframe containing bank marketing campaign dataset

    OUTPUT
    df - cleaned dataset:
    1. columns with 'yes' and 'no' values are converted into boolean variables;
    2. categorical columns are converted into dummy variables;
    3. drop irrelevant columns.
    4. impute incorrect values
    '''

    cleaned_df = df.copy()

    # convert columns containing 'yes' and 'no' values to boolean variables and drop original columns
    bool_columns = ['default', 'housing', 'loan', 'y']
    for bool_col in bool_columns:
        cleaned_df[bool_col + '_bool'] = df.apply(lambda row: get_dummy_from_bool(row, bool_col), axis=1)

    cleaned_df = cleaned_df.drop(columns=bool_columns)

    # convert categorical columns to dummies
    cat_columns = ['job', 'marital', 'education', 'contact', 'month', 'poutcome']

    for col in cat_columns:
        cleaned_df = pd.concat([cleaned_df.drop(col, axis=1),
                                pd.get_dummies(cleaned_df[col], prefix=col, prefix_sep='_',
                                               drop_first=True, dummy_na=False)], axis=1)

    # drop irrelevant columns
    cleaned_df = cleaned_df.drop(columns=['pdays'])

    # impute incorrect values and drop original columns
    cleaned_df['campaign_cleaned'] = df.apply(lambda row: get_correct_values(row, 'campaign', 34, cleaned_df), axis=1)
    cleaned_df['previous_cleaned'] = df.apply(lambda row: get_correct_values(row, 'previous', 34, cleaned_df), axis=1)

    cleaned_df = cleaned_df.drop(columns=['campaign', 'previous'])

    return cleaned_df

class MarketingData(DataProvider):

    def __str__(self):
        return "Marketing-Simulation"

    def load_training_data(self):

        # del_col = ['y_bool', 'default_bool', 'loan_bool', 'housing_bool', 'day']
        #
        # data = pd.read_csv(path, sep=';')
        # cleaned_data = clean_data(data)
        # data = cleaned_data.iloc[:, :25].drop(columns=del_col) # only use customer data
        path = os.path.join(config.ROOT_DIR, 'datasets/banking/clean.csv')
        data = pd.read_csv(path)
        age, balance, duration = data['age'].values, data['balance'].values, data['duration'].values
        manager = data['job_management'].values
        edu_sec = data['education_secondary'].values
        edu_ter = data['education_tertiary'].values
        married = data['marital_married'].values

        scaled_balance = minmax_scale(balance.reshape(-1, 1), feature_range=(-100, 10000))[:, 0]
        standard_age = StandardScaler().fit_transform(age.reshape(-1, 1))[:, 0]

        self.y_0 = (65 - age)**2 + manager*150 + scaled_balance
        ite =  (65-age)*10 + edu_sec*200 + edu_ter*100 - married*150 + np.random.normal(100, 10)
        self.y_1 = self.y_0 + ite
        self.t = np.random.binomial(1, 1 - sigmoid(standard_age), size=len(age))
        self.y = (self.y_1*self.t) + self.y_0*(1-self.t)
        self.y_cf = self.y_1*(1 - self.t) + self.y_0*(self.t)
        self.x = data.values

if __name__ == '__main__':

    path = os.path.join(config.ROOT_DIR, 'datasets/banking/clean.csv')
    data = pd.read_csv(path)

    gen = MarketingData()
    # utils.plot_y_dist(gen.y, gen.y_cf)
    # utils.ite_plot(gen.y_1, gen.y_0)
    # utils.dist_plot(gen.y_0, method_name='marketing_y0')
    # utils.dist_plot(gen.y_1, method_name='marketing_y1')
    utils.confounder_outcome_plot(data['age'], gen.y_1 - gen.y_0, dataset='marketing-age')
    utils.confounder_outcome_plot(data['balance'], gen.y_1 - gen.y_0, dataset='marketing-balance')
    utils.confounder_outcome_plot(data['marital_married'], gen.y_1 - gen.y_0, dataset='marketing-married')
    utils.confounder_outcome_plot(data['job_management'], gen.y_1 - gen.y_0, dataset='marketing-manager')

    twins = gen
    choice = np.random.choice(len(twins.x), size=1000)
    # utils.surface_plot(twins.y_1[choice], twins.y_0[choice], twins.y[choice], twins.y_cf[choice], twins.x[choice],name='Marketing')
    utils.simple_comparison_mean(gen.y, gen.t)
    print(gen.get_train_ate())
