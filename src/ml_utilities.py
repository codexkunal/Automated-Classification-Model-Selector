import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# get the working directory of the main.py file
working_dir = os.path.dirname(os.path.abspath(__file__))

# get the parent directory
parent_dir = os.path.dirname(working_dir)


# step 1 : Read the Data
def read_data(file_name):
    file_path = f"{parent_dir}/data/{file_name}"
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        return df
    elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path)
        return df


# step 2 : Preprocess the data
def preprocess_data(df, target_column, scaler_type):
    global scaler, X_train
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numerical_columns = X.select_dtypes(include=['number']).columns
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns

    if len(numerical_columns) == 0:
        pass
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        num_imputer = SimpleImputer(strategy='mean')
        X_train[numerical_columns] = num_imputer.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = num_imputer.transform(X_test[numerical_columns])

        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()

        X_train[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
        X_test[numerical_columns] = scaler.transform(X_test[numerical_columns])

    if len(categorical_columns) == 0:
        pass
    else:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        X_train[categorical_columns] = cat_imputer.fit_transform(X_train[categorical_columns])
        X_test[categorical_columns] = cat_imputer.transform(X_test[categorical_columns])

        encoder = OneHotEncoder()
        X_train_encoded = encoder.fit_transform(X_train[categorical_columns])
        X_test_encoded = encoder.transform(X_test[categorical_columns])
        X_train_encoded = pd.DataFrame(X_train_encoded.toarray(),
                                       columns=encoder.get_feature_names(categorical_columns))
        X_test_encoded = pd.DataFrame(X_test_encoded.toarray(), columns=encoder.get_feature_names(categorical_columns))
        X_train = pd.concat([X_train.drop(columns=categorical_columns), X_train_encoded], axis=1)
        X_test = pd.concat([X_test.drop(columns=categorical_columns), X_test_encoded], axis=1)

    return X_train, X_test, y_train, y_test


# training
def train_model(X_train, y_train, model, model_name):
    model.fit(X_train, y_train)
    with open(f"{parent_dir}/trained_model/{model_name}.pkl", 'wb') as file:
        pickle.dump(model, file)
    return model


# evaluate
def evalute_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy = round(accuracy, 2)
    return accuracy
