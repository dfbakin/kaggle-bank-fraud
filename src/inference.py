import pandas as pd
import numpy as np

def create_submission(model, df_loader, test_filepath, submission_filepath):
    X_test = pd.read_csv(test_filepath)

    X_test_processed = df_loader.transform(X_test)

    test_predictions = np.argmax(model.predict_proba(X_test_processed), axis=1)

    submission_df = pd.read_csv("data/sample_submission.csv")
    submission_df["prediction"] = test_predictions

    submission_df.to_csv(submission_filepath, index=False)

    predicted_scores = model.predict_proba(X_test_processed)[:, 1]
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predicted_scores, shade=True, color="blue")
    plt.title("Density of predicted scores")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.savefig("data/predicted_scores_density.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(predicted_scores, shade=True, color="blue")
    plt.xscale("log")
    plt.title("Log density of predicted scores")
    plt.xlabel("Predicted score")
    plt.ylabel("Density")
    plt.savefig("data/predicted_scores_density_log.png")
    plt.close()


if __name__ == "__main__":
    from train import load_catboost_model
    from preprocessing import FraudDataFrame
    import seaborn as sns
    import json
    import matplotlib.pyplot as plt

    model = load_catboost_model("models/catboost_model.cbm")

    feature_importances = model.get_feature_importance(prettified=True)
    top_5_features = feature_importances[:5]
    top_5_features_dict = {
        feat: feat_imp
        for feat, feat_imp in zip(top_5_features["Feature Id"], top_5_features["Importances"])
    }

    with open("data/top_5_feature_importances.json", "w") as f:
        json.dump(top_5_features_dict, f)

    df_loader = FraudDataFrame(filepath="data/train.csv")
    df_loader.load_preprocessor("models/preprocessor.joblib")

    create_submission(
        model,
        df_loader,
        test_filepath="data/test.csv",
        submission_filepath="data/catboost_submission.csv"
    )

