import pandas as pd
import numpy as np

def create_submission(model, df_loader, test_filepath, submission_filepath):
    X_test = pd.read_csv(test_filepath)

    X_test_processed = preprocessor.transform(X_test)

    test_predictions = np.argmax(model.predict_proba(X_test_processed), axis=1)

    submission_df = pd.read_csv("data/sample_submission.csv")
    submission_df["prediction"] = test_predictions

    submission_df.to_csv(submission_filepath, index=False)


if __name__ == "__main__":
    from train import load_catboost_model
    from preprocessing import FraudDataFrame

    model = load_catboost_model("models/catboost_model.cbm")

    df_loader = FraudDataFrame(filepath="data/train.csv")
    df_loader.load_preprocessor("models/preprocessor.joblib")
    preprocessor = df_loader._preprocessor

    create_submission(
        model,
        df_loader,
        test_filepath="data/test.csv",
        submission_filepath="submissions/catboost_submission.csv"
    )
