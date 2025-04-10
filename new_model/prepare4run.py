from sklearn.model_selection import GroupShuffleSplit

def downsample(train):
    """Goal:
        Downsample the training set, remove samples that have not been evaluated.
    -----------------------------------------------------------------------------
    Input/output:
        train, pd.DataFrame"""
    # Evaluated?
    train["size"] = train.groupby(["srch_id", "click_bool"])["position"].transform("size")
    train["evaluated"] = train.groupby(["srch_id", "click_bool"])["position"].transform("max")
    train["evaluated"] = train.groupby(["srch_id"])["evaluated"].transform(lambda x: max(min(x), 10))
    train.loc[train["evaluated"] != (train["size"] - 1), "evaluated"] =  train["size"] # Correct
    train.loc[(train["position"] - train["evaluated"]) <= 0, "evaluated"] = 0
    train.loc[train["evaluated"] == 0, "evaluated"] =  1
    # Downsample
    train = train[train["evaluated"] == 1]
    train = train.drop(["evaluated", "size"], axis = 1)

    return train


def train2val(df, prop):
    """Goal:
        Split training set into a train and validation set.
    -------------------------------------------------------
    Input:
        df:
            DataFrame, pd.DataFrame
        prop:
            Test size [-], float
    --------------------------------------------------------
    Output:
        tval and val:
            Training and validation set, pd.DataFrame"""
    # ---- Split
    tval_idx, val_idx = next(GroupShuffleSplit(n_splits = 1, test_size = prop, random_state = 1).split(
            df, groups=df["srch_id"]))

    # Get train and validation set
    tval, val = df.iloc[tval_idx, :], df.iloc[val_idx, :]

    return tval.copy(), val.copy()


def get_traineval(train, prop):
    """Goal:
        Function retrieves train and validation set.
    -----------------------------------------------
    Input:
        train:
            Training set, pd.DataFrame
        prop:
            Train size proportion, float
    -----------------------------------------------
    Output:
        tval, val:
            pd.Dataframes
        tval_groups, val_groups:
            The sizes of the groups, np.arrays"""
    # Get x and y
    tval, val = train2val(train, prop = prop)
    # Get group sizes
    tval_groups = tval.groupby("srch_id", group_keys=False).size().to_numpy()
    val_groups = val.groupby("srch_id", group_keys=False).size().to_numpy()

    return tval, val, tval_groups, val_groups