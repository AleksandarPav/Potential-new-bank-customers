import os
from datetime import date, datetime
import pandas as pd
import pickle
import csv


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def make_dataframe(init_df, filename):
    if not isinstance(init_df, pd.DataFrame):
        print('The provided input is not a DataFrame!')
        return

    columns = init_df.columns[0].split(';')
    columns = [col.translate({ord(i): None for i in '"'}) for col in columns]
    df = pd.DataFrame(columns=columns)

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for k, row in enumerate(reader):
            values1 = row[list(row.keys())[0]].split(';')

            if len(list(row.keys())) > 1:
                values2 = row[list(row.keys())[1]][0].split(';')

                if len(row[list(row.keys())[1]]) > 1:
                    value3 = row[list(row.keys())[1]][1]
                    num = values1[-1] + '.' + values2[0]
                    num2 = values2[-1] + '.' + value3
                    # del values1[-1]
                    # del values2[0]
                    # values1.append(num)
                    # [values1.append(x) for x in values2]
                    # del values1[-1]
                    # values1.append(num2)
                    values1 = values1[:-1] + [num] + values2[1:-1] + [num2]

                else:
                    num = values1[-1] + '.' + values2[0]
                    # del values1[-1]
                    # del values2[0]
                    # values1.append(num)
                    # [values1.append(x) for x in values2]
                    values1 = values1[:-1] + [num] + values2[1:]

            # for i, x in enumerate(values1):
            #     if is_number(x):
            #         values1[i] = float(x)
            #     else:
            #         values1[i] = values1[i].translate({ord(i): None for i in '"'})
            values1 = [float(x) if is_number(x) else x.translate({ord(i): None for i in '"'}) for x in values1]

            df.loc[k] = values1

    with open(filename.lower().split('.')[0] + ' df.pkl', 'wb') as file:
        pickle.dump(df, file)


def remove_columns(df, columns):
    df.drop(columns, axis=1, inplace=True)
    return df


def transform_date_columns(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: (date.today() - datetime.strptime(x, '%Y-%m-%d').date()).days)

    return df


def get_dummies(df, columns):
    return pd.get_dummies(df, columns=columns, prefix=columns, drop_first=True, dtype=float)


def in_folder(filename, folder_path):
    return filename in os.listdir(folder_path)
