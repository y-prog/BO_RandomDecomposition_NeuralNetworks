import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from proj_methods import  cleaning_methods

class data_preprocessor:
    def __init__(self):
        pass

    def preprocess_census_data(self):
        df = pd.read_csv(r'datasets\adult_tr.csv')
        clean_df = cleaning_methods(df)
        df = clean_df.remove_rows(col_name='threshold', wanted_rows_each_categ=200, seed=97)
        df['net'] = df.capital_gain - df.capital_loss
        cols = ['net', 'age', 'final_weight', 'education-num', 'capital_gain', 'capital_loss', 'hours_per_week']
        df[cols] = MinMaxScaler().fit_transform(df[cols])
        X = df[cols]
        y = df.threshold
        return train_test_split(X, y, test_size=0.25, random_state=42)

    def preprocess_creditcards_data(self):
        df = pd.read_csv(r'datasets\creditcard.csv')
        columns_to_normalize = df.columns.difference(['Class'])  # Modify this if you need to exclude any columns
        df[columns_to_normalize] = MinMaxScaler().fit_transform(df[columns_to_normalize])
        df = cleaning_methods(df).remove_rows(col_name='Class', wanted_rows_each_categ=200, seed=83)
        X = df.drop(['Class'], axis='columns')
        y = df['Class']
        train_test_df = train_test_split(X, y, test_size=0.25, random_state=42)
        return train_test_df

    def preprocess_titanic_data(self):
        df = pd.read_csv(r'datasets\train.csv')
        # Drop rows with missing values in specific columns
        preds_list = ['Sex', 'Age', 'Fare']
        df = df.dropna(subset=preds_list)
        y = df['Survived']  # store target variable
        # Encode categorical variable 'Sex' into dummy variables
        encode_cat_var = pd.get_dummies(df[preds_list[0]])
        # Drop the original 'Sex' column and concatenate the encoded columns with the DataFrame
        df = df[preds_list]
        df = df.drop([preds_list[0]], axis=1)
        X = pd.concat([encode_cat_var, df], axis=1)
        # Scale the features using Min-Max scaling
        X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
        train_test_df = train_test_split(X, y, test_size=0.25, random_state=42)
        return train_test_df

    def preprocess_realestate_data(self):
        df = pd.read_csv(r'datasets\RE_data.csv')
        scaler = MinMaxScaler()
        # Fit and transform the data
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        df = cleaning_methods(df).remove_outliers(.25, .75)
        df['tax_bin'] = df['TAX'].apply(lambda x: 1 if x > .228 else 0)
        df = cleaning_methods(df).remove_rows(col_name='tax_bin', wanted_rows_each_categ=200, seed=83)  # 125
        X = df[['INDUS', 'NOX', 'RAD', 'CRIM', 'CHAS']]
        y = df['tax_bin']
        train_test_df = train_test_split(X, y, test_size=0.25, random_state=42)
        return train_test_df