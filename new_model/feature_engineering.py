import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

def convert_dates(df):
    """Goal:
        Function converts dates to days and months.
    ----------------------------------------------
    Input/output:
        df:
            pd.DataFrame"""
    
    # Add day and month columns
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['month'] = df['date_time'].dt.month
    df['day'] = df['date_time'].dt.day
    df['day_of_week'] = df['date_time'].dt.dayofweek

    # Drop date time column
    df.drop(['date_time'], axis=1, inplace=True)

    #One hot encode month, day and day of week
    df["month_old"] = df["month"]
    df = pd.get_dummies(df, columns=["month", 'day', 'day_of_week'])

    return df

def AddHotelInformation(df): #Should be normalised as well --> just decide between mean and median
    for col in ["price_usd", "prop_log_historical_price", "visitor_hist_starrating", "visitor_hist_adr_usd",
                "prop_location_score1", "prop_location_score2", 'position', 'orig_destination_distance']:
        for prop in ["mean", "max", "min", "std"]:
            if (col == "position") and (prop in ["min", "max"]):
                pass
            elif col == ("position"):
                df[prop + '_' + col + '_per_hotel'] = df.groupby(['prop_id'])[col].transform(prop)
            else:
                df[prop + '_' + col + '_per_hotel'] = df.groupby(['prop_id'])[col].transform(prop)

    #Mean price per hotel per month
    df['mean_price_per_hotel_per_month'] = df.groupby(['prop_id','month_old'])['price_usd'].transform('mean')
    # #Median price per hotel
    # df['median_price_per_hotel'] = df.groupby(['prop_id'])['price_usd'].transform('median')
    # #Median price per hotel per month
    # df['median_price_per_hotel_per_month'] = df.groupby(['prop_id','month_old'])['price_usd'].transform('median')
    #Price difference with mean price per hotel
    df['price_difference_mean_per_hotel'] = df['price_usd'] - df['mean_price_usd_per_hotel']
    #Price difference with mean price per hotel per month
    df['price_difference_mean_per_hotel_per_month'] = df['price_usd'] - df['mean_price_per_hotel_per_month']
    # #Price difference with median price per hotel
    # df['price_difference_median_per_hotel'] = df['price_usd'] - df['median_price_per_hotel']
    # #Price difference with median price per hotel per month
    # df['price_difference_median_per_hotel_per_month'] = df['price_usd'] - df['median_price_per_hotel_per_month']
    # Drop month
    df = df.drop(["month_old"], axis = 1)

    return df

def PeopleCount(df):
    df['total_people'] = df['srch_adults_count'] + df['srch_children_count']
    return df

def normalize_price(df):
    """Goal:
        Function normalize price in 3 ways
    ---------------------------------------------------
    Input/Output:
        df"""
    # Version 1
    df["price_usd_normalized1"] = df["price_usd"] / df["srch_length_of_stay"]
    # Version 2
    df["price_usd_normalized2"] = df["price_usd"] / (df["srch_length_of_stay"] * df["srch_adults_count"])
    # Version 3
    df["price_usd_normalized3"] = df["price_usd"] / (df["srch_length_of_stay"] * (df["srch_adults_count"] +  df["srch_children_count"]))

    return df

def normalize_cols(df):
    """Goal:
        Normalize columns by group.
    ------------------------------
    Input/Output:
        df:
            Dataframe, pd.DataFrame"""
    # --- Columns to normalize
    cols2normalize = ["price_usd"]

    # ---- Get categorical variables
    catcols = []
    for col in df.columns:
        try:
            if np.array_equal(df[col], df[col].astype(int)):
                if (len(np.unique(df[col])) == (max(df[col]) - min(df[col]) + 1)) or (col in ["srch_id", "prop_country_id", "prop_id", "srch_destination_id", "visitor_location_country_id"]):
                    if col not in ["total_people", "size", "npromo", "nclicks", "nbookings", "srch_adults_count", "prop_starrating", "prop_review_score", "price_usd", "srch_children_count", "srch_room_count"]:
                        if ("_rate_percent_diff" not in col)  and ("competitors_" not in col):
                            catcols.append(col)
        except ValueError:
            pass
    
    # --- Get columns to normalize
    cols2normalize = [col for col in df.columns if col not in catcols]
    cols2normalize = [col for col in cols2normalize if col not in ["rank_weight", 'click_bool', 'booking_bool', 'position', 'gross_bookings_usd', "target"]]

    # --- Normalize
    np.seterr(divide = 'ignore')
    for group in ["srch_id"]:
        # ---- Get median of group
        median = df.groupby(group)[cols2normalize].transform("median")
        # ---- Normalize
        df[["normalized_" + col for col in cols2normalize]] = (df[cols2normalize] - median) / median

    return df

def countrydiff(df):
    """Goal:
        Search within the same country?
    -----------------------------------
    Input/Output:
        df, pd.DataFrame"""
    
    df["within_country"] = (df["visitor_location_country_id"] - df["prop_country_id"]) == 0

    return df

def rank_weights(df):
    """Goal:
        Get weights based on rank.
    --------------------------------------
    Input/Output:
        df, pd.DataFrame"""
    
    # Initialize function
    def func(x, a, b, c):
        return a * (1 / (x**b)) + c
    
    # Sum positional outcomes
    summedrank = df.loc[df["random_bool"] == 0, :].groupby("position")["target"].mean() 

    # Fit predictpor
    popt, pcov = curve_fit(func, summedrank.index, summedrank, bounds=([0, 0, 0], [summedrank.max(), 1, 0.1 * summedrank.min()]))

    # Get weights
    df["rank_weight"] = 1 / func(df["position"], *popt)

    return df, popt

def getratios(df):
    # Get train and test data
    train = df[~df["booking_bool"].isna()].copy()
    test = df[df["booking_bool"].isna()].copy()
    del df

    # Evaluated?
    train["size"] = train.groupby(["srch_id", "click_bool"])["position"].transform("size")
    train["evaluated"] = train.groupby(["srch_id", "click_bool"])["position"].transform("max")
    train["evaluated"] = train.groupby(["srch_id"])["evaluated"].transform(lambda x: max(min(x), 10))
    train.loc[train["evaluated"] != (train["size"] - 1), "evaluated"] =  train["size"] # Correct
    train.loc[(train["position"] - train["evaluated"]) <= 0, "evaluated"] = 0
    train.loc[train["evaluated"] == 0, "evaluated"] =  1

    # nclicks, nbookings, evaluated
    train["nclicks"] = train.groupby("prop_id")["click_bool"].transform("sum")
    train["nbookings"] = train.groupby("prop_id")["booking_bool"].transform("sum")
    train["npromo"] = train.groupby("prop_id")["promotion_flag"].transform("sum")

    # Ratios 1
    #train["ratio"] = train["nclicks"] / train.groupby("prop_id").transform("size")

    train["ratio2"] = train["nbookings"] / train["nclicks"]
    train.loc[train["nclicks"] < 30, "ratio2"] = np.nan

    # Ratios 2
    # train["ratio3"] = train["nclicks"] / train.groupby("prop_id")["evaluated"].transform("sum")
    # train["ratio4"] = train["nbookings"] / train.groupby("prop_id")["evaluated"].transform("sum")

    # Drop
    train = train.drop(["size"], axis = 1)
    # Concat
    df = pd.concat([train, test], ignore_index = True)
    del train
    del test

    # Fill
    for col in ["ratio2"]: #"ratio", "ratio2", "ratio3", "ratio4"
        df[col] = df[col].fillna(df.groupby("prop_id")[col].transform("mean"))
        #df["missingness_" + col] = df[col].isna()
        #df[col] = df[col].fillna(0)
    # df["evaluated"] = df['evaluated'].fillna(1)

    # Drop evaluated
    df = df.drop(["evaluated", "nclicks", "npromo", "nbookings"], axis = 1)
    
    return df
