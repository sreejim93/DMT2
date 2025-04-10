from copy import deepcopy
import feature_engineering as fe
import filtimpute as fi
import lightgbm as lgb
import predict_position.main_position as mp
import numpy as np
import optimize as opt
import pandas as pd
import prepare4run as p4r
import matplotlib.pyplot as plt
import sklearn
import xgboost as xgb

# ---- Load datasets
train = pd.read_csv('./datasets/train.csv')
#test = pd.read_csv('../datasets/test.csv')
test = pd.read_csv('./testset.csv')
test = test.drop(["Unnamed: 0"], axis = 1)#
test_target = np.where(test.loc[:, 'booking_bool'] == 1, 5, np.where(test.loc[:, 'click_bool'] == 1, 1, 0))#
test["position"] = np.nan#
test["booking_bool"] = np.nan#
test["click_bool"] = np.nan#


# for col in train.columns:
#     if col not in ["price_usd", "prop_location_score2", "prop_location_score1", "prop_log_historical_price", "prop_starrating", "promotion_flag",
#                    "prop_review_score", "orig_destination_distance", "prop_brand_bool", "srch_id", "random_bool"]:
#         if col not in ["srch_destination_id", "prop_country_id", "visitor_location_country_id", "prop_id", 'click_bool', 'booking_bool', 'position', 'gross_bookings_usd']:
#             train = train.drop([col], axis = 1)
#             test = test.drop([col], axis = 1)

# ---- Concanate dfs and delete old --> faster and usefull
dfcombined = pd.concat([train, test], ignore_index = True)
del train
del test

# ---- Select columns
# dfcombined = dfcombined[["srch_query_affinity_score", "orig_destination_distance", "prop_location_score2", "visitor_hist_starrating", "visitor_hist_adr_usd", "srch_saturday_night_bool", "srch_children_count", "srch_room_count", "srch_adults_count", 
#                          "srch_length_of_stay", "promotion_flag", "prop_log_historical_price", "prop_location_score1", 
#                          "prop_id", "prop_review_score", "prop_starrating", "price_usd", "srch_id", "prop_brand_bool", "random_bool", 
#                          'click_bool', 'booking_bool', 'position', 'gross_bookings_usd', "srch_destination_id", "prop_country_id", "visitor_location_country_id"]]

# Categorical features with negative values are converted to positive
catcols = []
for col in dfcombined.columns:
    try:
        if np.array_equal(dfcombined[col], dfcombined[col].astype(int)):
            if dfcombined[col].min() < 0:
                dfcombined[col] = dfcombined[col] - dfcombined[col].min()
    except ValueError:
        pass

# ---- Filtering and imputation
print("Filtering and Imputing")
dfcombined = fi.filter_price(dfcombined)
dfcombined = fi.filtstarprice(dfcombined)
dfcombined = fi.filtreview(dfcombined)
# dfcombined = fi.imputeloc2(dfcombined)
# dfcombined = fi.imputesaff(dfcombined)
# dfcombined = fi.impoddist(dfcombined)
# dfcombined = fi.impcomp(dfcombined)

# ---- Convert dates
print("Converting dates.....")
if "date_time" in dfcombined.columns:
    dfcombined = fe.convert_dates(dfcombined)
    
# ---- Normalize price
print("Normalizing prices.....")
# dfcombined = fe.normalize_price(dfcombined)

# ---- Feature engineering
print("Feature engineering....")
dfcombined = fe.AddHotelInformation(dfcombined)
# dfcombined = fe.PeopleCount(dfcombined)
# dfcombined = fe.countrydiff(dfcombined)
#dfcombined = fe.getratios(dfcombined)

# --- Add target column to train_df, 0 if click_bool = 0 and 5 if booking_bool = 1, else 1
print("Setting target....")
dfcombined['target'] = np.where(dfcombined.loc[:, 'booking_bool'] == 1, 5, np.where(dfcombined.loc[:, 'click_bool'] == 1, 1, 0))

# ---- Get weights for training
print("Getting weights....")
dfcombined, popt = fe.rank_weights(dfcombined) # Also required for position prediction

# ---- Predict positions
print("Predicting position....") # Watch out---> this one does not work yet. Have to repair it first, so do not uncomment!!!
#dfcombined["pred_position"] = mp.get_mainscore(dfcombined, popt, 1)
dfcombined = fe.normalize_cols(dfcombined)
# ---- Split and delete old
print("Split dataframe....")
test_props = dfcombined.loc[dfcombined["booking_bool"].isna(), "prop_id"]
train = dfcombined[~dfcombined["booking_bool"].isna()]
test = dfcombined[dfcombined["booking_bool"].isna()].drop(["prop_id", "rank_weight", 'click_bool', 'booking_bool', 'position', 'gross_bookings_usd'], axis=1)
del dfcombined

# ---- Downsample
#train = p4r.downsample(train)
train = train.drop(["prop_id", 'click_bool', 'booking_bool', 'position', 'gross_bookings_usd'], axis=1)
cols2drop = []
for col in train.columns:
    if ("month" in col) or ("day" in col):
        cols2drop.append(col)
for col in train.columns:
    if "bool" in col:
        if col not in ["srch_saturday_night_bool", "random_bool", "prop_brand_bool"]:
            cols2drop.append(col)

train = train.drop(cols2drop, axis = 1)
test = test.drop(cols2drop, axis = 1)
# ---- Run optimizer
print("Start optimization....")
tval, val, tval_groups, val_groups = p4r.get_traineval(train, prop = 0.2) # Split train into train and validation groups
Xtval, Xval, ytval, yval = tval.drop(['target', "srch_id"], axis = 1), val.drop(['target', "srch_id"], axis = 1), tval["target"], val["target"]
#opt_results = opt.run_optuna(Xtval, ytval, tval_groups, Xval, yval, val_groups, runs = 100, k = 5)

# --- Insert --> drop some columns!!!
if "rank_weight" in Xval.columns:
    Xval = Xval.drop(["rank_weight"], axis = 1)
    Xtval = Xtval.drop(["rank_weight"], axis = 1)
if "rank_weight" in test.columns:
    test = test.drop(["rank_weight"], axis = 1)

# ---- Set parameters for LGBM 
#params = opt_results.best_params
params = {'num_leaves': 1398, 'max_depth': 46, 'min_data_in_leaf': 5000, 
          'reg_alpha': 1.3873408576611912, 'reg_lambda': 9.493210019923042, 
          'min_gain_to_split': 4.013088350318114, 'subsample': 0.900979078425516, 
          'bagging_freq': 1, 'colsample_bytree': 0.6627560865691828, 
          'learning_rate': 0.016513550304148843}

# Set parameters for XGBoost 
paramsXGB = {
        'boosting_type': 'gbtree',
        'metric': 'ndcg',
        'learning_rate': 0.02,
        'num_leaves': 50,
        'max_depth': 15
    }


# ---- Get X and y
Xtest, ytest = test.drop(['target'], axis = 1), test["target"]

# Correct Xtest
grouped_test = Xtest.groupby("srch_id", group_keys=False)
save_srch_ids = Xtest["srch_id"].values
cols = set(Xtest.columns) - set(Xtval.columns)
Xtest.drop(cols, axis=1, inplace=True)

# ---- Get categorical variables
catcols = []
for col in Xtval.columns:
    try:
        if np.array_equal(Xtval[col], Xtval[col].astype(int)):
            if (len(np.unique(Xtval[col])) == (max(Xtval[col]) - min(Xtval[col]) + 1)) or (col in ["prop_country_id", "prop_id", "srch_destination_id", "visitor_location_country_id"]):
                if col not in ["total_people", "size", "npromo", "nclicks", "nbookings", "srch_adults_count", "prop_starrating", "prop_review_score", "price_usd", "srch_children_count", "srch_room_count"]:
                    if ("_rate_percent_diff" not in col)  and ("competitors_" not in col):
                        catcols.append(col)
    except ValueError:
        pass

# ---- Fit model for lgb
print("Fit model....")
model = lgb.LGBMRanker(**params, n_jobs=-1, objective="lambdarank",
                           n_estimators=1000)
model.fit(Xtval, ytval, eval_at = [5], eval_set = [(Xval, yval)], eval_group = [val_groups], group = tval_groups,
    categorical_feature = catcols)

# ---- Fit model for xgb
#print("Fit model....")
#model = xgb.XGBRanker(**paramsXGB, n_jobs=-1, objective="rank:ndcg")
#model.fit(Xtval, ytval, eval_set = [(Xval, yval)], eval_group = [val_groups], group = tval_groups, eval_metric = "ndcg@5", verbose = True, early_stopping_rounds = 10)

# ---- Predict model
print("Predict test set ....")
testsizes = grouped_test.size().to_numpy()
predictions = model.predict(Xtest, group = testsizes)

idx_start = 0
ndcg = 0#
for group_size in testsizes:
    # Get batch
    batch = predictions[idx_start:idx_start+group_size]
    target =  np.array([test_target[idx_start:idx_start+group_size]])#
    prediction = np.array([batch])#
    ndcg += sklearn.metrics.ndcg_score(target, prediction, k=5, sample_weight=None, ignore_ties=False)#
    # Increase
    idx_start += group_size

ndcg /= grouped_test.ngroups#
print("NDCG5 is: ", ndcg)




lgb.plot_importance(model, figsize=(20,15), max_num_features = 10)
plt.show()

# --- To csv
Xtest['srch_id'] = save_srch_ids
grouped_test = Xtest.groupby('srch_id')
Xtest.drop(['srch_id'], axis=1, inplace=True)

#per grouped search predict the ranking
preds = []
for name, group in grouped_test:
    prediction = model.predict(group)
    preds.append(prediction)
preds = np.concatenate(preds)
Xtest['pred'] = preds
Xtest['srch_id'] = save_srch_ids
Xtest["prop_id"] = test_props
ranking = Xtest.sort_values(['srch_id', 'pred'], ascending=[True, False]).loc[:, ['srch_id', 'prop_id']]
ranking.to_csv('ranking.csv', index=False)
