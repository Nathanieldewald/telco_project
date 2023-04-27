import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


def plot_churn_rate_by_internet_service_type(df):
    fiber_vs_dsl = pd.concat([df.internet_service_type_fiber_optic, df.internet_service_type_dsl, df.churn], axis=1)
    fiber_vs_dsl['fiber_dsl'] = np.where(fiber_vs_dsl['internet_service_type_fiber_optic'] == 1, 'fiber_optic',
                                     np.where(fiber_vs_dsl['internet_service_type_dsl'] == 1, 'dsl', 'neither'))
    # plot the churn rate for each internet service type vs not churning
    ax = sns.countplot(x='fiber_dsl', hue='churn', data=fiber_vs_dsl, order=['fiber_optic', 'dsl'])
    # Label the plot
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(ticks=[0, 1], labels=['Fiber Optic', 'DSL'])
    plt.title('Churn Count by Internet Service Type')
    plt.xlabel('Internet Service Type')
    plt.ylabel('Customer Count')
    plt.legend(['No Churn', 'Churn'])
    plt.show()


def eval_results(p):
    alpha = .05
    if p < alpha:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

def chi2_test(observed):
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('Observed\n')
    print(observed.values)
    print('---\nExpected\n')
    print(expected)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def chi2_test_short(observed):
    # Only prints chi2, p, and whether we reject or fail to reject the null hypothesis
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print('---\n')
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')

def chi2_test_for_churn_and_internet_service_type(df):
    observed = pd.crosstab(df.internet_service_type_fiber_optic, df.internet_service_type_dsl)
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

def chi2_test_for_churn_and_internet_service_type_short(df):
    # Only prints chi2, p, and whether we reject or fail to reject the null hypothesis
    observed = pd.crosstab(df.internet_service_type_fiber_optic, df.internet_service_type_dsl)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < 0.05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")
def plot_churn_rate_by_above_avg_monthly_charge(train):
    # plot the churn rate for each internet service type vs not churning
    # create a column that is true if the monthly charge is above the average monthly charge of $64.76
    train['above_avg_monthly_charge'] = train.monthly_charges > train.monthly_charges.mean()
    ax = sns.countplot(x='above_avg_monthly_charge', hue='churn', data=train)

    # Label the plot
    for container in ax.containers:
        ax.bar_label(container)
    plt.title('Churn Rate by Above Average Monthly Charge for Fiber Optic Customers')
    plt.xlabel('Above Average Monthly Charge of $64.76')
    plt.xticks(ticks=[0, 1], labels=['Below Average', 'Above Average'])
    plt.ylabel('Churn Rate')
    plt.show()

def chi2_test_for_churn_and_above_avg_monthly_charge(df):
    # Prints the chi2, p, whether or not we reject the null hypothesis, and the observed and expected values
    observed = pd.crosstab(df.above_avg_monthly_charge, df.churn)
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

def chi2_test_for_churn_and_above_avg_monthly_charge_short(df):
    # Only prints the chi2, p, and whether or not we reject the null hypothesis
    observed = pd.crosstab(df.above_avg_monthly_charge, df.churn)
    chi2, p, degf, expected = stats.chi2_contingency(observed)

    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < 0.05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")

def plot_churn_rate_by_contract_type(train):
    # create a new column that indicates whether or not the customer has a month-to-month contract
    train['contract_type_month_to_month'] = train['contract_type_One year'] + train['contract_type_Two year']
    train['contract_type_month_to_month'] = np.where(train['contract_type_month_to_month'] == 0, 1, 0)
    # combine the two year and one year contract types
    # create dataframe with contract type and churn
    contract_vs_churn = pd.concat([train['contract_type_month_to_month'], train['contract_type_One year'], train['contract_type_Two year'], train['churn']], axis=1)
    # create column with contract type month to month or long term contract (one or two year)
    contract_vs_churn['contract_type'] = np.where(contract_vs_churn['contract_type_month_to_month'] == 1, 'month_to_month',
                                              np.where(contract_vs_churn['contract_type_One year'] == 1, 'long_term', 'long_term'))
    ax = sns.countplot(x='contract_type', hue='churn', data=contract_vs_churn)
    # Label the plot
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(ticks=[0, 1], labels=['Long Term', 'Month to Month'])
    plt.title('Churn Count by Contract Type')
    plt.xlabel('Contract Type')
    plt.ylabel('Customer Count')
    plt.legend(['No Churn', 'Churn'])
    plt.show()

def chi2_test_for_churn_and_contract_type(train):
    # create a new column that indicates whether or not the customer has a month-to-month contract
    train['contract_type_month_to_month'] = train['contract_type_One year'] + train['contract_type_Two year']
    train['contract_type_month_to_month'] = np.where(train['contract_type_month_to_month'] == 0, 1, 0)
    # combine the two year and one year contract types
    # create dataframe with contract type and churn
    contract_vs_churn = pd.concat(
        [train['contract_type_month_to_month'], train['contract_type_One year'], train['contract_type_Two year'],
         train['churn']], axis=1)
    # create column with contract type month to month or long term contract (one or two year)
    contract_vs_churn['contract_type'] = np.where(contract_vs_churn['contract_type_month_to_month'] == 1,
                                                  'month_to_month',
                                                  np.where(contract_vs_churn['contract_type_One year'] == 1,
                                                           'long_term', 'long_term'))
    observed = pd.crosstab(contract_vs_churn.contract_type, contract_vs_churn.churn)
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


def chi2_test_for_churn_and_contract_type_short(train):
    # Only print chi2 test results, and whether or not we reject the null hypothesis
    # create a new column that indicates whether or not the customer has a month-to-month contract
    train['contract_type_month_to_month'] = train['contract_type_One year'] + train['contract_type_Two year']
    train['contract_type_month_to_month'] = np.where(train['contract_type_month_to_month'] == 0, 1, 0)
    # combine the two year and one year contract types
    # create dataframe with contract type and churn
    contract_vs_churn = pd.concat(
        [train['contract_type_month_to_month'], train['contract_type_One year'], train['contract_type_Two year'],
         train['churn']], axis=1)
    # create column with contract type month to month or long term contract (one or two year)
    contract_vs_churn['contract_type'] = np.where(contract_vs_churn['contract_type_month_to_month'] == 1,
                                                  'month_to_month',
                                                  np.where(contract_vs_churn['contract_type_One year'] == 1,
                                                           'long_term', 'long_term'))
    observed = pd.crosstab(contract_vs_churn.contract_type, contract_vs_churn.churn)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < 0.05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")
def plot_churn_rate_by_dependents(df):
    # plot churn rate by dependents
    ax = sns.countplot(x='dependents', hue='churn', data=df)
    for container in ax.containers:
        ax.bar_label(container)
    #label the plot
    plt.title('Churn Rate by Dependents')
    plt.xlabel('Dependents')
    plt.ylabel('Customer Count')
    plt.legend(['No Churn', 'Churn'])
    plt.xticks([0, 1], ['No Dependents', 'Dependents'])
    plt.show()

def chi2_test_for_churn_and_dependents(train):
    # Perform chi2 test and print results
    observed = pd.crosstab(train.dependents, train.churn)
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

def chi2_test_for_churn_and_dependents_short(train):
    # Only prints the chi2, p, and whether or not we reject the null hypothesis
    observed = pd.crosstab(train.dependents, train.churn)
    chi2, p, degf, expected = stats.chi2_contingency(observed)
    print(f'chi^2 = {chi2:.4f}')
    print(f'p     = {p:.4f}')
    if p < 0.05:
        print("We reject the null hypothesis")
    else:
        print("We fail to reject the null hypothesis")