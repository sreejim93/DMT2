import datetime as dt
import lightgbm as lgb
import numpy as np
import optuna
import pickle

def objective(trial, xtrain, ytrain, traingroup, xval, yval, valgroup, k, catcols):
    """Goal:
            Support optimizer
        ---------------------------------------------------------------------------
        Input:
            trial:
                ...
            xtrain/ytrain:
                Trainings data, pd.DataFrames
            traingroup:
                Group sizes, np.array
            xval/yval:
                Validation data, pd.Dataframes
            valgroup:
                Group sizes, np.array
            k:
                ...
            catcols:
                Categorical columns, list
        """
    # ---- Set parameters
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 1, 1500),
        "max_depth": trial.suggest_int("max_depth", 1, 50),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 10000, step = 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "subsample": trial.suggest_float("subsample", 0.1, 1),
                    "bagging_freq": trial.suggest_categorical("bagging_freq", [1]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
    }

    # ---- Set model
    model = lgb.LGBMRanker(**params, n_jobs=-1, objective="lambdarank",
                           n_estimators=1000)
    
    # ---- Split and get weights
    sample_weight = xtrain["rank_weight"].values / xtrain["rank_weight"].values # 1 now
    eval_sample_weight = [xval["rank_weight"].values / xval["rank_weight"].values] # 1 now
    xtrain = xtrain.drop(["rank_weight"], axis = 1)
    xval = xval.drop(["rank_weight"], axis = 1)

    # ---- Fit the model
    model.fit(X = xtrain, y = ytrain, group = traingroup,
        eval_at = [5], eval_set = [(xval, yval)], eval_group = [valgroup], categorical_feature = catcols, sample_weight = sample_weight,
        eval_sample_weight = eval_sample_weight, early_stopping_rounds=20, verbose=10)

    # ---- Get model score
    score = model.best_score_['valid_0'][f"ndcg@{k}"]

    return score

def run_optuna(Xtval, ytval, tval_groups, Xval, yval, val_groups, runs, k):
    """Goal:
            Get optimal parameters.
        ---------------------------------------------------------------------------
        Input:
            Xtval:
                X data of the training data, pd.DataFrame
            ytval:
                Y data of the training data, pd.DataFrame/pd.Series
            tval_groups:
                Group sizes, numpy array
            Xval, yval:
                Data of validation set, pd.DataFrame/pd.Series
            val_groups:
                Group sizes, numpy array
            runs:
                Number of iterations, int
            k:
                ...., int
        -----------------------------------------------------------------------------
        Output:

            
        """
    # ---- Set study name
    study_name = "Optimize_Lambda"

    # ---- Create study
    study = optuna.create_study(direction = "maximize", study_name = study_name)

    # ---- Get categorical columns
    catcols = []
    for col in Xtval.columns:
        try:
            if np.array_equal(Xtval[col], Xtval[col].astype(int)):
                if (len(np.unique(Xtval[col])) == (max(Xtval[col]) - min(Xtval[col]) + 1)) or (col in ["prop_country_id", "prop_id", "srch_destination_id", "visitor_location_country_id"]):
                    if col not in ['normalized_srch_adults_count', 'normalized_srch_length_of_stay', 'normalized_srch_room_count', "total_people", "size", "npromo", "nclicks", "nbookings", "srch_adults_count", "prop_starrating", "prop_review_score", "price_usd", "srch_children_count", "srch_room_count"]:
                        if ("_rate_percent_diff" not in col)  and ("competitors_" not in col):
                            catcols.append(col)
        except ValueError:
            pass
        
    # ---- Get objective
    def obj(trial):
        return objective(trial, Xtval, ytval, tval_groups, Xval, yval, val_groups, k, catcols)

    # ---- Optimize
    study.optimize(obj, n_trials = runs, show_progress_bar=True)

    # ---- Return best values and save them
    date = dt.datetime.now().strftime("%m-%d_%H%M")
    print(f"Optimal value: {study.best_value: .4f}")
    with open(f"optuna_run_{date}.pickle", 'wb') as f:
        pickle.dump(study, f)

    return study

