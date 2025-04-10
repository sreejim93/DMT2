from copy import deepcopy
import numpy as np
import pandas as pd

def filter_price(df):
    """Goal:
        Function filters and imputes price. 5 x median range.
        -----------------------------------------------------
        Input/Output:
            df, pd.DataFrame"""
    #---- Set zeros to NaN
    df["price_usd_miss"] = 0 # Save miss
    df["price_usd_miss"] = df["price_usd_miss"].where(df["price_usd"] != 0, 1)
    df["price_usd"] = df["price_usd"].where(df["price_usd"] != 0, np.nan)
    # ---- Get median of group
    median = df.groupby("srch_id")["price_usd"].transform("median")
    # ---- Impute NaNs
    df["price_usd"] = df["price_usd"].where(~df["price_usd"].isna(), median)
    # ---- Get relative difference
    diff = df["price_usd"] / median
    # ---- Adapt
    df["price_usd_miss"] = df["price_usd_miss"].where(((diff > 0.2) & (diff < 5)), 1)
    df["price_usd"] = df["price_usd"].where(((diff > 0.2) & (diff < 5)), median)

    return df

def filtstarprice(df):
    """Goal:
        Function calculates differences in starrating and price, also gets
            new users.
    -----------------------------------------------------------------------
        Input/Output:
            df, pd.DataFrame"""
    # ------ Difference in starrating
    # Get difference between visitor and hotel rating
    df["diff starrating"] = df["visitor_hist_starrating"] - df["prop_starrating"]
    # Save nans in a bool
    df["bool miss visitor starrating"] = df["diff starrating"].isna()
    # Fill nans with zero --> no difference
    df["diff starrating"] = df["diff starrating"].fillna(value = 0)

    # ------ Get new users
    bool_newusers = df["visitor_hist_starrating"].isna() + df["visitor_hist_adr_usd"].isna()
    bool_newusers = np.where(bool_newusers == 1, 0, bool_newusers)
    bool_newusers = np.where(bool_newusers == 2, 1, bool_newusers)
    df["bool new users"] = bool_newusers

    # ------ Get difference between visitor and hotel price
    df["diff hotel price"] = df["visitor_hist_adr_usd"] - df["price_usd"]
    # Save nans in a bool
    df["bool miss visitor hotel price"] = df["diff hotel price"].isna()
    # Fill nans with zero --> no difference
    df["diff hotel price"] = df["diff hotel price"].fillna(value = 0)
    # Remove
    # df = df.drop("visitor_hist_starrating", axis = 1)
    # df = df.drop("visitor_hist_adr_usd", axis = 1)

    return df

def filtreview(df):
    """Goal:
        Impute review score.
    -------------------------
    Input/output:
        df, pd.DataFrame"""
    # Bool missing
    df["bool review unknown"] = df["prop_review_score"].isna() # Checked it, and everything before is also nan. So not randomly missing
    df["bool review absent"] = np.where(df["prop_review_score"] == 0, 1, 0)

    # Fill nans with zero, no information seems related to missingness?
    df["prop_review_score"] = df["prop_review_score"].fillna(0)

    return df

def imputeloc2(df):
    """Goal:
        Impute location score 2
    ----------------------------------
    Input/output:
        df, pd.DataFrame"""
    # ----- Save missing values in bool
    df["bool miss prop_location_score2"] = df["prop_location_score2"].isna()

    # ----- Impute
    # First round, try on worst score for the hotel --> worst 25%
    df["prop_location_score2"] = df.groupby("prop_id")["prop_location_score2"].transform(lambda group: group.fillna(group.quantile(0.25)))
    # Second round, impute based on destination id --> worst 25%
    df["prop_location_score2"] = df.groupby("srch_destination_id")["prop_location_score2"].transform(lambda group: group.fillna(group.quantile(0.25)))
    # Third round, impute based on country --> worst 25%
    df["prop_location_score2"] = df.groupby("prop_country_id")["prop_location_score2"].transform(lambda group: group.fillna(group.quantile(0.25)))

    return df

def imputesaff(df):
    """Goal:
        Impute search affinity score.
    --------------------------------
    Input/output:
        df, pd.DataFrame"""
    # Save missing values to bool
    df["bool miss srch_query_affinity_score"] = df["srch_query_affinity_score"].isna()

    # Impute with 100 times the minimum log value, as zeros will reach towards -infinity
    df["srch_query_affinity_score"] = df["srch_query_affinity_score"].fillna(df["srch_query_affinity_score"].min() * 100)

    return df

def impoddist(df):
    """Goal:
        Impute destination distance.
    ------------------------------------------
    Input/output:
        df, pd.DataFrame"""

    # Save missing values to bool
    df["bool miss orig_destination_distance"] = df["orig_destination_distance"].isna()

    # Save raw
    rawdest = deepcopy(df["orig_destination_distance"])

    # ---- Impute
    # First round: Impute by median of visitor loaction and srch destination
    df["orig_destination_distance"] = df.groupby(["visitor_location_country_id", "srch_destination_id"])["orig_destination_distance"].transform(lambda group: group.fillna(group.median()))
    # Second round: Impute by median of visitor location and hotel country
    df["orig_destination_distance"] = df.groupby(["visitor_location_country_id", "prop_country_id"])["orig_destination_distance"].transform(lambda group: group.fillna(group.median()))
    # Third round: Impute by median of srch destination location
    df["orig_destination_distance"] = df.groupby(["srch_destination_id"])["orig_destination_distance"].transform(lambda group: group.fillna(group.median()))
    # Fourth round: Impute by median of hotel country
    df["orig_destination_distance"] = df.groupby(["prop_country_id"])["orig_destination_distance"].transform(lambda group: group.fillna(group.median()))
    # Fifth round: Impute by overall median
    df["orig_destination_distance"] = df["orig_destination_distance"].fillna(rawdest.median())

    return df


def impcomp(df):
    """Goal:
        Impute and aggregate competitors.
        ---------------------------------
        Input/Output:
            df, pd.DataFrame"""

    # # ---- Initialize
    # df["competitors_sum_lower"] = 0
    # df["competitors_sum_higher"] = 0
    # df["competitors_sum_equal"] = 0
    # df["competitors_sum_NaN"] = 0

    # Impute with zero
    for icomp in range(1, 9):
        # Get  columns
        col1 = "comp" + str(icomp) + "_rate"
        col2 = "comp" + str(icomp) + "_inv"
        col3 = "comp" + str(icomp) + "_rate_percent_diff"
        # # Fill nans with zeros
        # df["competitors_sum_lower"] = df["competitors_sum_lower"] + (df[col1] == -1)
        # df["competitors_sum_higher"] = df["competitors_sum_higher"] + (df[col1] == 1)
        # df["competitors_sum_equal"] = df["competitors_sum_equal"] + (df[col1] == 0)
        # df["competitors_sum_NaN"] = df["competitors_sum_NaN"] + (df[col1].isna())

        # Mean, max, min, std
        for col in [col1, col2, col3]:
            for prop in ["mean", "max", "min", "std"]:
                df[prop + '_' + col + '_per_hotel'] = df.groupby(['prop_id'])[col].transform(prop)

        # Drop
        #df = df.drop([col1, col2, col3], axis = 1)

    # Get final sum
    #df["competitors_pen"] = (df["competitors_sum_lower"] + df["competitors_sum_NaN"] + df["competitors_sum_equal"]) - 2 * df["competitors_sum_higher"]
    
    return df