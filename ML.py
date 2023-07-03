from data_preparation import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pickle


# def scale_data(df, columns_drop):
#     X = df.drop(columns_drop, axis=1)
#     y = df['Mortgage_YN']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     return X_train, X_test, y_train, y_test, scaler


def train_models(df):
    X = df.drop(['Mortgage_YN', 'AGE_AT_ORIGINATION'], axis=1)

    """
    mortgage
    """
    y_mort = df['Mortgage_YN']

    X_train, X_test, y_train_mort, y_test_mort = train_test_split(X, y_mort, test_size=0.2, random_state=42)

    scaler_mort = StandardScaler()
    X_train = scaler_mort.fit_transform(X_train)
    X_test = scaler_mort.transform(X_test)

    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train_mort.values)

    with open('models/logistic_model.pkl', 'wb') as file:
        pickle.dump(logistic_model, file)

    y_pred_yn = logistic_model.predict(X_test)

    accuracy = accuracy_score(y_test_mort, y_pred_yn)
    precision = precision_score(y_test_mort, y_pred_yn, pos_label='Y')
    recall = recall_score(y_test_mort, y_pred_yn, pos_label='Y')
    f1 = f1_score(y_test_mort, y_pred_yn, pos_label='Y')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    cm = confusion_matrix(y_test_mort, y_pred_yn)
    print('Confusion matrix:')
    print(cm)

    y_pred_prob = logistic_model.predict_proba(X_test)[:, 0] * 100

    print('Probability of taking a mortgage above 50%: ', (y_pred_prob > 50).sum(), ' / ', len(y_pred_prob))

    """
    age
    """
    y_age = df['AGE_AT_ORIGINATION']
    X_train, X_test, y_train_age, y_test_age = train_test_split(X, y_age, test_size=0.2, random_state=42)

    scaler_age = StandardScaler()
    X_train = scaler_age.fit_transform(X_train)
    X_test = scaler_age.transform(X_test)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train_age.values)

    with open('models/linear_model.pkl', 'wb') as file:
        pickle.dump(linear_model, file)

    y_pred_age = linear_model.predict(X_test)

    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(len(y_test_age)), y_test_age, 'r-', label='Real data')
    plt.plot(np.arange(len(y_test_age)), y_pred_age, 'b-', label='Predicted data')
    plt.legend()
    plt.xlabel('Person no.')
    plt.ylabel('Age')
    plt.savefig('figures/' + 'Real and predicted data.png')
    plt.show()

    mse = mean_squared_error(y_test_age, y_pred_age)
    rmse = np.sqrt(mse)
    print('MSE: ', mse)
    print('RMSE: ', rmse)

    return scaler_mort, scaler_age


# def train_mortgage_yn(df):
#     if in_folder('logistic_model_mortgage_yn.pkl', 'models/'):
#         return
#
#     X = df.drop(['Mortgage_YN', 'AGE_AT_ORIGINATION'], axis=1)
#     y = df['Mortgage_YN']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     logistic_model = LogisticRegression()
#     logistic_model.fit(X_train, y_train.values)
#
#     with open('models/logistic_model_mortgage_yn.pkl', 'wb') as file:
#         pickle.dump(logistic_model, file)
#
#     y_pred = logistic_model.predict(X_test)
#
#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, pos_label='Y')
#     recall = recall_score(y_test, y_pred, pos_label='Y')
#     f1 = f1_score(y_test, y_pred, pos_label='Y')
#     print("Accuracy:", accuracy)
#     print("Precision:", precision)
#     print("Recall:", recall)
#     print("F1 Score:", f1)
#
#     cm = confusion_matrix(y_test, y_pred)
#     print('Confusion matrix:')
#     print(cm)
#
#     return scaler
#
#
# def train_age_at_origination(df):
#     if in_folder('linear_model_age_at_origination.pkl', 'models/'):
#         return
#
#     X = df.drop(['Mortgage_YN', 'AGE_AT_ORIGINATION'], axis=1)
#     y = df['AGE_AT_ORIGINATION']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     linear_model = LinearRegression()
#     linear_model.fit(X_train, y_train.values)
#
#     with open('models/linear_model_age_at_origination.pkl', 'wb') as file:
#         pickle.dump(linear_model, file)
#
#     y_pred = linear_model.predict(X_test)
#
#     plt.figure(figsize=(15, 10))
#     plt.plot(np.arange(len(y_test)), y_test, 'r-', label='Real data')
#     plt.plot(np.arange(len(y_test)), y_pred, 'b-', label='Predicted data')
#     plt.legend()
#     plt.xlabel('Person no.')
#     plt.ylabel('Age')
#     plt.savefig('figures/' + 'Real and predicted data.png')
#     plt.show()
#
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = np.sqrt(mse)
#     print('MSE: ', mse)
#     print('RMSE: ', rmse)
#
#     return scaler
#
#
# def train_mortgage_probability(df):
#     if in_folder('logistic_model_mortgage_probability.pkl', 'models/'):
#         return
#
#     X = df.drop(['Mortgage_YN', 'AGE_AT_ORIGINATION'], axis=1)
#     y = df['Mortgage_YN']
#
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
#
#     logistic_model = LogisticRegression()
#     logistic_model.fit(X_train, y_train.values)
#
#     with open('models/logistic_model_mortgage_probability.pkl', 'wb') as file:
#         pickle.dump(logistic_model, file)
#
#     y_prob = logistic_model.predict_proba(X_test)[:, 0] * 100
#
#     print(y_prob)
#     print(sum(y_prob < 50))
#     print(sum(y_test == 'Y'))
#
#     return scaler


def analyze_potential_customers(df, scaler_mort, scaler_age):
    with open('models/logistic_model.pkl', 'rb') as file:
        logistic_model = pickle.load(file)
    with open('models/linear_model.pkl', 'rb') as file:
        linear_model = pickle.load(file)

    df_mort = scaler_mort.transform(df)
    df_age = scaler_age.transform(df)

    y_mortgage_yn = logistic_model.predict(df_mort)
    y_mortgage_probability = logistic_model.predict_proba(df_mort)[:, 0] * 100
    y_age_at_origination = linear_model.predict(df_age)

    return y_mortgage_yn, y_age_at_origination, y_mortgage_probability
