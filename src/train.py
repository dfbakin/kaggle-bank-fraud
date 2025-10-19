from catboost import CatBoostClassifier

def train_catboost_baseline(X_train, y_train, X_val=None, y_val=None, save_filename=None):
    model = CatBoostClassifier(
        iterations=50,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        use_best_model=True,
    )
    
    if X_val is not None and y_val is not None:
        eval_set = (X_val, y_val)
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
    )

    if save_filename is not None:
        model.save_model(save_filename)
    
    return model


def load_catboost_model(model_filepath):
    model = CatBoostClassifier()
    model.load_model(model_filepath)
    return model


if __name__ == "__main__":
    from preprocessing import FraudDataFrame

    df_loader = FraudDataFrame(filepath="data/train.csv")
    df_loader.fit_preprocessor(df_loader.X_train)
    df_loader.dump_preprocessor("models/preprocessor.joblib")

    X_train, X_val, y_train, y_val = df_loader.get_data_splits()
    X_train_processed = df_loader.transform(X_train)
    X_val_processed = df_loader.transform(X_val)

    model = train_catboost_baseline(
        X_train_processed,
        y_train,
        X_val=X_val_processed,
        y_val=y_val,
        save_filename="models/catboost_model.cbm"
    )
