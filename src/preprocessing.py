from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

class FraudDataFrame:
    def __init__(self, df=None, filepath=None, test_size=0.2, random_state=42):
        if df is not None:
            self.df = df
        elif filepath is not None:
            self.df = pd.read_csv(filepath)
        else:
            raise ValueError
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.df.drop(columns=["target"]), 
            self.df["target"], 
            test_size=test_size,
            random_state=random_state,
            # stratify=self.df["target"]
        )

        self._preprocessor = None

    def get_data_splits(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load_preprocessor(self, preprocessor_filepath):
        self._preprocessor = joblib.load(preprocessor_filepath)

    def dump_preprocessor(self, preprocessor_filepath):
        if self._preprocessor is None:
            raise ValueError
        joblib.dump(self._preprocessor, preprocessor_filepath)
    
    def fit_preprocessor(self, X):
        if self._preprocessor is None:
            numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
            numerical_pipeline = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())
                    ])

            categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
            categorical_pipeline = Pipeline(steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore"))
                    ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numeric_features),
                    ("cat", categorical_pipeline, categorical_features)
                ]
            )

            self._preprocessor = preprocessor.fit(X)

        self._preprocessor.fit(X)    
    
    def transform_dataframe(self):
        if self._preprocessor is None:
            raise ValueError

        
        X_train_processed = self._preprocessor.transform(self.X_train)
        X_test_processed = self._preprocessor.transform(self.X_test)

        return X_train_processed, X_test_processed, self.y_train, self.y_test
    
    def transform(self, X):
        if self._preprocessor is None:
            raise ValueError

        return self._preprocessor.transform(X)