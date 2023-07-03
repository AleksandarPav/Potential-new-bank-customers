from data_preparation import *
from visualizations import *
from ML import *
import pandas as pd
import time


def main():
    df_retail = get_dataframe('retail data df.pkl', 'Retail data.csv', ['Cocunut', 'CURRENT_WITH_BANK_DATE'],
                              ['CURRENT_ADDRESS_DATE', 'CURRENT_JOB_DATE'], ['EMPLOYMENT', 'GENDER', 'MARTIAL_STATUS',
                                                                             'EDUCATION'], ['AGE_AT_ORIGINATION'], True,
                              ['EDUCATION', 'EMPLOYMENT', 'GENDER', 'MARTIAL_STATUS'], 'Mortgage_YN', ['AGE'], True)
    df_potential = get_dataframe('potential customers df.pkl', 'Potential Customers.csv', ['Cocunut',
                                                                                           'CURRENT_WITH_BANK_DATE'],
                                 ['CURRENT_ADDRESS_DATE', 'CURRENT_JOB_DATE'], ['EMPLOYMENT', 'GENDER',
                                                                                'MARTIAL_STATUS', 'EDUCATION'])

    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.precision', 3,
    #                        ):
    #     print(df_retail.head())
    # with pd.option_context('display.max_rows', None,
    #                        'display.max_columns', None,
    #                        'display.precision', 3,
    #                        ):
    #     print(df_potential.head())

    mortgage_yn = df_retail['Mortgage_YN']
    age_at_origination = df_retail['AGE_AT_ORIGINATION']
    common_columns = df_potential.columns.intersection(df_retail.columns)
    df_potential = df_potential[common_columns]
    df_retail = df_retail[common_columns]
    df_retail['Mortgage_YN'] = mortgage_yn
    df_retail['AGE_AT_ORIGINATION'] = age_at_origination

    # train_mortgage_yn(df_retail)
    # train_age_at_origination(df_retail)
    # train_mortgage_probability(df_retail)
    scaler_mort, scaler_age = train_models(df_retail)

    mortgage_yn_predictions, age_at_origination_predictions, mortgage_probability_predictions =\
        analyze_potential_customers(df_potential, scaler_mort, scaler_age)
    print((mortgage_yn_predictions == 'Y').sum())
    indices = [index for index, value in enumerate(mortgage_yn_predictions) if value == 'Y']
    print(df_potential.iloc[indices])
    print(age_at_origination_predictions)
    print((mortgage_probability_predictions < 50).sum())


if __name__ == '__main__':
    main()
