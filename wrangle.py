from sklearn.model_selection import train_test_split

from env import host, username, password
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def get_db_url(database):
    # this function uses my env.py file to get the url to access the Codeup SQL server
    # it takes the name of the database as an argument
    # it returns the url
    database = database
    url = f'mysql+pymysql://{username}:{password}@{host}/{database}'
    return url


def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):

        return pd.read_csv(fn, index_col=0)
    else:
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

def get_telco():
    url = get_db_url('telco_churn')
    query = ''' select * from customers
    join contract_types
    	using (contract_type_id)
    join internet_service_types
    	using (internet_service_type_id)
    join payment_types
    	using (payment_type_id)
        '''
    filename = 'telco.csv'
    df = check_file_exists(filename, query, url)

    return df

def wrangle_telco_not_encoded(df):
    '''
    This function will acquire the telco data from the codeup database,
    clean the data by dropping nulls, and convert the total_charges column
    to a float.
    '''
    # Acuire data from codeup database

    telco = df.drop(columns=['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    # Replace yes and no with 1 and 0
    telco['churn'] = telco['churn'].replace({'Yes': 1, 'No': 0})
    telco['partner'] = telco['partner'].replace({'Yes': 1, 'No': 0})
    telco['dependents'] = telco['dependents'].replace({'Yes': 1, 'No': 0})
    telco['phone_service'] = telco['phone_service'].replace({'Yes': 1, 'No': 0})
    telco['paperless_billing'] = telco['paperless_billing'].replace({'Yes': 1, 'No': 0})
    telco['senior_citizen'] = telco['senior_citizen'].replace({0: 'No', 1: 'Yes'})
    # Replace blank spaces with 0 due to the fact that the customers have not been charged yet
    telco['total_charges'] = telco['total_charges'].replace(' ', '0')
    telco['total_charges'] = telco['total_charges'].astype('float')

    return telco
def wrangle_telco_encoded(df):
    '''
    This function will acquire the telco data from the codeup database,
    clean the data by dropping nulls, and convert the total_charges column
    to a float.
    '''
    # Acuire data from codeup database

    telco = df.drop(columns=['customer_id', 'payment_type_id', 'internet_service_type_id', 'contract_type_id'])
    # Replace yes and no with 1 and 0
    telco['churn'] = telco['churn'].replace({'Yes': 1, 'No': 0})
    telco['partner'] = telco['partner'].replace({'Yes': 1, 'No': 0})
    telco['dependents'] = telco['dependents'].replace({'Yes': 1, 'No': 0})
    telco['phone_service'] = telco['phone_service'].replace({'Yes': 1, 'No': 0})
    telco['paperless_billing'] = telco['paperless_billing'].replace({'Yes': 1, 'No': 0})

    # Replace blank spaces with 0 due to the fact that the customers have not been charged yet
    telco['total_charges'] = telco['total_charges'].replace(' ', '0')
    telco['total_charges'] = telco['total_charges'].astype('float')
    # Create numeric and categorical dataframes
    num = telco.select_dtypes(include="number")
    char = telco.select_dtypes(include="object")
    # Create dummy variables for the object columns
    char_1 = pd.get_dummies(char, drop_first=True)
    # Concatenate the numeric and categorical dataframes
    telco_clean = pd.concat([num, char_1], axis=1)

    return telco_clean

def eval_results(p):
    alpha = .05
    if p < alpha:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")


def plot_churn_rate_by_internet_service_type():
    df = wrangle_telco_not_encoded()
    fiber_vs_dsl = df[df.internet_service_type != 'None']
    sns.barplot(x='internet_service_type', y='churn', data=fiber_vs_dsl)
    # Label the plot
    plt.title('Churn Rate by Internet Service Type')
    plt.xlabel('Internet Service Type')
    plt.ylabel('Churn Rate')
    plt.show()


def chi2_test(observed):
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')


def chi2_test_for_churn_and_internet_service_type():
    df = wrangle_telco_not_encoded()
    observed = pd.crosstab(df.churn, df.internet_service_type)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < 0.05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")


# split the data into train, validate, and test
def train_validate_test(df, strat):
    '''
    This function will take in a dataframe and return train, validate, and test dataframes split
    where 55% is in train, 25% is in validate, and 20% is in test.
    '''
    train_validate, test = train_test_split(df, test_size=0.2,
                                            random_state=123,
                                            stratify=df[strat])
    train, validate = train_test_split(train_validate, test_size=0.25,
                                       random_state=123,
                                       stratify=train_validate[strat])
    return train, validate, test