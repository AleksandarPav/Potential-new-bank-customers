import matplotlib.pyplot as plt


def value_count(df, columns):
    for col in columns:
        counts = df[col].value_counts()
        plt.figure(figsize=(15, 10))
        counts.plot(kind='bar')
        plt.title('Count of ' + col)
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.savefig('figures/' + 'Count of ' + col + '.png')


def value_count_grouped(df, columns_group, column_target):
    for col in columns_group:
        df_new = df.groupby([col])
        counts = df_new[column_target].value_counts()
        plt.figure(figsize=(15, 10))
        counts.plot(kind='bar')
        plt.title('Count of ' + column_target + ' grouped by ' + col)
        plt.xlabel(column_target + ', ' + col)
        plt.ylabel('Count')
        plt.savefig('figures/' + 'Count of ' + column_target + ' grouped by ' + col + '.png')


def histogram(df, columns):
    for col in columns:
        plt.figure(figsize=(15, 10))
        df[col].hist()
        plt.title('Histogram of ' + col)
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig('figures/' + 'Histogram of ' + col + '.png')


plt.show()
