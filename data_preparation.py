from utils import *
from visualizations import *
import os


def get_dataframe(pkl_filename, csv_filename, columns_remove, columns_date, columns_dummy, columns_na=None, train=False,
                  columns_viz_group=None, column_viz_target=None, columns_viz_hist=None, visualize=False):
    if pkl_filename not in os.listdir('.'):
        init_df = pd.read_csv(csv_filename, on_bad_lines='skip')
        make_dataframe(init_df, csv_filename)
    with open(pkl_filename, 'rb') as file:
        df = pickle.load(file)

    if visualize:
        value_count_grouped(df, columns_viz_group, column_viz_target)
        histogram(df, columns_viz_hist)

    df = remove_columns(df, columns_remove)

    df = transform_date_columns(df, columns_date)

    index_bad = df[(df[columns_date[0]] < 0) | (df[columns_date[1]] < 0)].index
    df.drop(index_bad, inplace=True)

    df = get_dummies(df, columns_dummy)

    if train:
        df[columns_na[0]] = df[columns_na[0]].apply(lambda x: 0.0 if x == 'NA' else x)

    return df
