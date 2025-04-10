from copy import deepcopy
from collections import Counter, OrderedDict
import graphviz
import itertools as it
import json
import shap
from sklearn import metrics, model_selection, tree
from sklearn.utils import class_weight
from skopt import gp_minimize
from skopt.plots import plot_convergence
from skopt.utils import use_named_args
import math
import matplotlib.pyplot as plt
import numpy as np
from os import path, remove
import pandas as pd
import statistics
import xgboost as xgb

class OrderedCounter(Counter, OrderedDict):
    """Goal:
        Counter that remembers the order elements are first seen"""
    def __repr__(self):
         return "%s(%r)" % (v___class__.__name__, OrderedDict(self))
    def __reduce__(self):
        return v___class__, (OrderedDict(self),)
        
def training_prediction(X, y, proc_predict, stratisfy = True):
    """Goal:
            Splits dataset in train-test and prediction data,
                stratisfied and random
        ----------------------------------------------------
        Input:
            X: 
                The X data, df
            y:
                The y data, pd.Series/df
            proc_predict:
                proc_predict, float
            stratisfy
                Additional variable in case of regression,
                    bool.
        -----------------------------------------------------
        Output:
            tuple:
                X_train_test/X_predict:
                    Train and test/prediction X data, df
                y_train_test/y_predict:
                    Train and test/prediction y data, df
                index_train_test/index_predict:
                    Indices of the train and test/prediction data, np array
    """
    # Create split
    if proc_predict == None: # No prediction split
        X_train_test, y_train_test = X, y
        X_predict, y_predict, index_train_test, index_predict = [], [], [], []
    elif stratisfy:
        X_train_test, X_predict, y_train_test, y_predict = model_selection.train_test_split(
            X, y, test_size = proc_predict, train_size = (1 - proc_predict), 
            random_state = 1, shuffle = True, stratify = y)
        index_train_test, index_predict = X_train_test.index.values, X_predict.index.values # Saving indices
    else:
        X_train_test, X_predict, y_train_test, y_predict = model_selection.train_test_split(
            X, y, test_size = proc_predict, train_size = (1 - proc_predict), 
            random_state = 1, shuffle = True)
        index_train_test, index_predict = X_train_test.index.values, X_predict.index.values # Saving indices

    return (X_train_test, X_predict, y_train_test, y_predict, index_train_test, index_predict)

def train_splitter (X_train_test, y_train_test, n_splits, stratisfy = True):
    """Goal:
            Creates stratified k-fold split settings and performs k splits on train-test.
        ---------------------------------------------------------------------------------
        Input:
            X_train_test, y_train_test, n_splits
        ----------------------------------------------------------------------------------
        Output:
            folds:
                The folds, dict
            indices:
                The indices of the folds, dict
            skf_1:
                The splitter, ..."""
    # Initialize folds and indices
    folds, indices = {}, {}
    i_1 = 1
    
    # Initialize k_fold model (random_state different than prediction)
    if stratisfy:
        skf_1 = model_selection.StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 42) 
    else:
        skf_1 = model_selection.KFold(n_splits = n_splits, shuffle = True, random_state = 42) 
    
    # Create folds
    for train_index, test_index in skf_1.split(X_train_test,y_train_test):
        X_train, X_test  = X_train_test.iloc[train_index], X_train_test.iloc[test_index]
        y_train, y_test = y_train_test.iloc[train_index], y_train_test.iloc[test_index]
        # Set folds
        folds['fold_{}'.format(i_1)] = {}
        folds['fold_{}'.format(i_1)]['X_train'] = X_train
        folds['fold_{}'.format(i_1)]['X_test'] = X_test
        folds['fold_{}'.format(i_1)]['y_train'] = y_train
        folds['fold_{}'.format(i_1)]['y_test'] = y_test
        # Set indices
        indices['index_{}'.format(i_1)] = {}
        indices['index_{}'.format(i_1)]['train'] = train_index
        indices['index_{}'.format(i_1)]['test'] = test_index
        # Increase fold
        i_1 += 1
        
    return (folds, indices, skf_1)

def optimize_model(model_settings, param_grid, folds, indices, cv, n_splits, 
    considered_model, scoring_prediction, direction, regression = False):
    """Goal:
            Optimize model by the use of gridsearch
        -------------------------------------------
        Input:
            model_settings:
                The settings for the model, class
            param_grid:
                The parameters and their values, dict
            folds:
                The folds, dict
            indices:
                The indices of the folds, dict
            cv:
                The splitting model, ..
            n_splits:
                The number of splits
            considered_model:
                The type of model, string
            scoring_prediction:
                The evaluation metric, dict
            direction:
                Direction to minimize, dict
        --------------------------------------------
        Output:
            folds_op, indices_op, models_op,
         train_results, test_results, rank, scoring)
    """
    # ---- Initialize variables
    final_model_cvresults_train, final_model_cvresults_test = {}, {}
    train_results, test_results = {}, {}
    folds_op, indices_op = {}, {}
    scoring = []

    # Set all scoring methods
    for scoring_method, value in scoring_prediction.items():
        if value == True:
            scoring.append(scoring_method)
            final_model_cvresults_train[scoring_method], final_model_cvresults_test[scoring_method] = [], []
            train_results[scoring_method], test_results[scoring_method] = {}, {}
    
    #  ---- Load the different versions of the considered model and the number of combinations
    models_op, number_of_combinations, optweights = create_models_optimalization (model_settings, considered_model, param_grid, regression)

    # Create new train and test splits
    for i_2 in range (1, (n_splits + 1)): # Create new train and test sets
        print("Split_{}".format(i_2))

        # Select train data and get its indices
        X_train = folds['fold_{}'.format((i_2))]['X_train']
        y_train = folds['fold_{}'.format((i_2))]['y_train']
        index = indices['index_{}'.format(i_2)]['train']

        # Initialize splits
        folds_op['fold_{}'.format(i_2)] = {}
        indices_op['index_{}'.format(i_2)] = {}

        # Creation of new split
        i_3 = 1
        for train_index_op, test_index_op in cv.split(X_train, y_train):
            # Get new train and test data
            X_train_op, X_test_op  = X_train.iloc[train_index_op], X_train.iloc[test_index_op]
            y_train_op, y_test_op = y_train.iloc[train_index_op], y_train.iloc[test_index_op]
            
            # Store splits
            folds_op['fold_{}'.format(i_2)][i_3] = {}
            indices_op['index_{}'.format(i_2)][i_3] = {}
            folds_op['fold_{}'.format(i_2)][i_3]['X_train'] = X_train_op
            folds_op['fold_{}'.format(i_2)][i_3]['X_test'] = X_test_op
            folds_op['fold_{}'.format(i_2)][i_3]['y_train'] = y_train_op
            folds_op['fold_{}'.format(i_2)][i_3]['y_test'] = y_test_op
            indices_op['index_{}'.format(i_2)][i_3]['train'] = index[train_index_op]
            indices_op['index_{}'.format(i_2)][i_3]['test'] = index[test_index_op]
            i_3 += 1

            # Set early stopping rounds
            if considered_model == "xgboost":
                X_train_op = np.array(X_train_op)
                y_train_op = np.array(y_train_op)
                X_test_op = np.array(X_test_op)
                y_test_op = np.array(y_test_op)
                # Get eval set
                X_train_op, X_eval, y_train_op, y_eval, index_train, index_eval = training_prediction(X_train_op, y_train_op, 0.1)


            ## Get current balance
            if not regression:
                ys = np.unique(y_train_op)
                weights_original = class_weight.compute_class_weight(class_weight = 'balanced', classes = ys, y = y_train_op.flatten())
                weights_original = {num: weights_original[num] for num in ys}

                ## Loop through combinations
                for combination in models_op:
                    # Get model
                    model = models_op[combination]
                    # Get weights
                    weights = deepcopy(weights_original)
                    optweight = optweights[combination]
                    if optweight.keys() != []:
                        for key, val in weights.items():
                            weights[key] = val * optweight[key]
                    # Set weights
                    w = [weights[kind] for kind in y_train_op.flatten()]
                # Fit model
                if considered_model == "xgboost":
                    if not regression:
                        model.min_child_weight = min(weights.values())
                        model_fitted = model.fit(X_train_op, y_train_op, sample_weight = w, eval_set = [(X_eval, y_eval)], verbose = model_settings.verbosity)
                    else:
                        model.min_child_weight = 5
                        model_fitted = model.fit(X_train_op, y_train_op, eval_set = [(X_eval, y_eval)], verbose = model_settings.verbosity)
                else:
                    model.class_weight = weights
                    model_fitted = model.fit(X_train_op, y_train_op)

                # Predict
                predicted_train = model_fitted.predict(X_train_op) # Classification train
                y_score_train = model_fitted.predict_proba(X_train_op) # Probability train for class labeled as 1
                predicted_test = model_fitted.predict(X_test_op) # Classification test
                y_score_test = model_fitted.predict_proba(X_test_op) # Probability test for class labeled as 1
                
                # Get scores
                score_train, score_test = scoring_results(y_train_op, y_test_op, predicted_train, 
                    predicted_test, y_score_train, y_score_test, scoring_prediction, weights_original)
                
                for score in scoring:
                    final_model_cvresults_train[score].append(score_train[score])
                    final_model_cvresults_test[score].append(score_test[score])
    
    # Get means and std over splits
    for score in scoring:
        cvresults_train = np.array(final_model_cvresults_train[score])
        cvresults_test = np.array(final_model_cvresults_test[score])
        train_reshape = np.reshape(cvresults_train, (n_splits**2, number_of_combinations))
        test_reshape = np.reshape(cvresults_test, (n_splits**2,number_of_combinations))
        diff_reshape = ((test_reshape-train_reshape)/train_reshape)*100
        
        # Calculate means and stds
        train_results[score]["mean"] = np.mean(train_reshape, axis = 0)
        train_results[score]["std"] = np.std(train_reshape, axis = 0)
        test_results[score]["mean"] = np.mean(test_reshape, axis = 0)
        test_results[score]["std"] = np.std(test_reshape, axis = 0)
        test_results[score]["diff_mean"] = np.mean(diff_reshape, axis = 0)
        test_results[score]["diff_std"] = np.std(diff_reshape, axis = 0)
    
    # Rank results
    rank = ranking(train_results, test_results, models_op, scoring, direction)

    return(folds_op, indices_op, models_op, train_results, test_results, rank, scoring, optweights)

def optimize_model_bayesian(predefined_model, space, params, cv, scoring_prediction, n_splits, folds, direction, considered_model, verbosity, regression = False):
    """Goal:
        Function optimizes model settings by Bayesian optimization."""
    # Initialize scoring
    scoring = []
    for scoring_method, value in scoring_prediction.items():
        if value == True:
            scoring.append(scoring_method)
    # Set model
    reg = predefined_model
    
    # Run optimizer
    @use_named_args(space)
    def objective(**params):
        # Initialize lists
        test_values = []
        train_values = []
        diff_values = []

        # Get weights
        Kclass_weights = {}
        for key in params.keys():
            if "class" in key:
                Kclass_weights[int(key.replace("class", ""))] = params[key]
        # Set file name and parameters
        filename = 'Bayesian.json'
        reg.set_params(**params)
        # Continue
        for i_2 in range (1, (n_splits + 1)):
            # Get train sets
            X_train = folds['fold_{}'.format((i_2))]['X_train']
            y_train = folds['fold_{}'.format((i_2))]['y_train']
            
            # Split train set in new train and testsets
            for train_index_op, test_index_op in cv.split(X_train, y_train):
                X_train_op, X_test_op  = X_train.iloc[train_index_op], X_train.iloc[test_index_op]
                y_train_op, y_test_op = y_train.iloc[train_index_op], y_train.iloc[test_index_op]

                # Get eval set
                X_train_op, X_eval, y_train_op, y_eval, index_train, index_eval = training_prediction(X_train_op, y_train_op, 0.1, not regression)

                ## Get current balance
                if not regression:
                    ys = np.unique(y_train_op)
                    weights_original = class_weight.compute_class_weight(class_weight = 'balanced', classes = ys, y = y_train_op.values.flatten())
                    weights_original = {num: weights_original[num] for num in ys}
                    # Adapt weights
                    weights = deepcopy(weights_original)
                    if Kclass_weights.keys() != []:
                        for weight, val in weights_original.items():
                            weights[weight] = val * Kclass_weights[weight]
                    # Set weights
                    w = [weights[kind] for kind in y_train_op.values.flatten()]

                # Fit model
                if considered_model == "xgboost":
                    if not regression:
                        reg.min_child_weight = min(weights.values())
                        model_fitted = reg.fit(X_train_op, y_train_op.values.flatten(), sample_weight = w, eval_set = [(X_eval, y_eval.values.flatten())], verbose = verbosity)
                    else:
                        reg.min_child_weight = 5
                        model_fitted = reg.fit(X_train_op, y_train_op.values.flatten(),  eval_set = [(X_eval, y_eval.values.flatten())], verbose = verbosity)
                elif considered_model == "decision_tree":
                    reg.class_weight = weights
                    model_fitted = reg.fit(X_train_op, y_train_op)
                # Get predictions
                predicted_train = model_fitted.predict(X_train_op) # Classification train
                predicted_test = model_fitted.predict(X_test_op) # Classification test
                if not regression:
                    y_score_test = model_fitted.predict_proba(X_test_op) # Probability test for class labeled as 1
                    y_score_train = model_fitted.predict_proba(X_train_op) # Probability train for class labeled as 1
                else:
                    weights_original = None
                    y_score_test = None
                    y_score_train = None
                # Get results
                score_train, score_test = scoring_results(y_train_op, y_test_op, predicted_train, 
                    predicted_test, y_score_train, y_score_test, {score: scoring_prediction[score]}, weights_original)
                test_values.append(score_test[score])
                train_values.append(score_train[score])
                diff_values.append(((score_test[score]-score_train[score])/score_train[score])*100)
        
        # Average over splits
        if direction[score] == False:
            test_value = - np.mean(test_values)
            train_value = - np.mean(train_values)
            diff_value = - np.mean(diff_values)
        elif direction[score] == True:
            test_value = np.mean(test_values)
            train_value = np.mean(train_values)
            diff_value = np.mean(diff_values)
        
        std_train = np.std(train_values)
        std_test = np.std(test_values)
        std_diff = np.std(diff_values)
        
        # Write test_values, train_values and differences to json
        # Read JSON file
        with open(filename) as fp:
            listObj = json.load(fp)
        print(len(listObj))
        
        listObj.append({
          "train": train_value, "std_train": std_train,
          "test": test_value, "std_test": std_test,
          "diff": diff_value, "std_diff": std_diff
        })
        
        with open(filename, 'w') as fp:
            json.dump(listObj, fp, indent=4, separators=(',',': '))
        
        return test_value
    
    # Run optimizer for all scoring
    for score in scoring:
        # The list of hyper-parameters we want to optimize. For each one we define the
        # bounds, the corresponding scikit-learn parameter name, as well as how to
        # sample values from that dimension (`'log-uniform'` for the learning rate)
        
        # Initialize json file
        filename = 'Bayesian.json'
        listObj = []
        if path.isfile(filename) is False:  
            pass
        else:
            remove(filename)
            
        with open(filename, 'w') as f_obj:
            json.dump(listObj, f_obj)
        
        # Search
        res_gp = gp_minimize(objective, space, n_calls=500, random_state=0, verbose = False)

        # Plot convergence
        plot_convergence(res_gp)
        plt.show()
        
        # Create pandas frame
        a = pd.DataFrame(data = res_gp.x_iters, columns = [col.name for col in space])
        b = pd.read_json(filename)
        c = pd.concat([a,b], axis = 1)
        
        # Return mean and std test scores and diff_scores
        test = c["test"].to_numpy()
        std_test = c["std_test"].to_numpy()
        diff = c["diff"].to_numpy()
        std_diff = c["std_diff"].to_numpy()
        
        # Get rank and write to excel
        rank = np.lexsort((std_diff, std_test, diff, test))
        c = c.reindex(rank)
        c.to_excel("bayesian_{}".format(score) + ".xlsx")
    return(scoring) 
                                       
def create_models_optimalization (model_settings, considered_model, param_grid, bayesian = False, results = False, row = False, regression = False):
    """Goal:
            Creates models for optimalization
        --------------------------------------
        Input:
            model_settings, considered_model,
                param_grid
            bayesian:
                Bayesian?, bool
            results:
                ....
            row: ....
        -------------------------------------
        Output:
            models_op:
                Models, dict
            number_of_combinations:
                Number of models, int"""
    # Initialize
    models_op, optweights = {}, {}
    error = 0
    
    # Set combinations
    if bayesian:
        allnames = param_grid
        length_allnames = len(allnames)
        list_combinations = results.iloc[int(row)].values[1 : (length_allnames + 1)]
        number_of_combinations = 1
    else:
        # Get parameter names
        allnames = list(param_grid.keys())
        # Get all combinations
        combinations = it.product(*(param_grid[name] for name in allnames))
        list_combinations = list(it.chain(*combinations))
        # Number of parameters and combinations
        length_allnames = len(allnames)
        length_combinations = len(list_combinations)
        number_of_combinations = int(length_combinations / length_allnames)
    
    # Set models
    for i_2 in range (0, int(number_of_combinations)): # For all possible combinations
        # Get Combinations
        start = 0 + i_2 * length_allnames
        end = start + length_allnames
        combination_local = list_combinations[start : end]
        
        if considered_model == "decision_tree":
            # Initialize model parameters
            v_criterion = model_settings.v_criterion
            v_splitter = model_settings.v_splitter
            v_min_weight_fraction_leaf = model_settings.v_min_weight_fraction_leaf
            v_max_features = model_settings.v_max_features
            v_random_state = model_settings.v_random_state
            v_max_leaf_nodes = model_settings.v_max_leaf_nodes
            v_min_impurity_decrease = model_settings.v_min_impurity_decrease
            v_ccp_alpha = model_settings.v_ccp_alpha
            v_max_depth = model_settings.v_max_depth
            v_min_samples_split = model_settings.v_min_samples_split
            v_min_samples_leaf = model_settings.v_min_samples_leaf
            
            # Change values of optimalized parameters   
            for j_1 in range(0,length_allnames):
                if allnames[j_1] == 'criterion':
                    v_criterion = combination_local[j_1]
                elif allnames[j_1] == 'splitter':
                    v_splitter = combination_local[j_1]
                elif allnames[j_1] == 'min_weight_fraction_leaf':
                    v_min_weight_fraction_leaf = combination_local[j_1]
                elif allnames[j_1] == 'max_features':
                    v_max_features = combination_local[j_1]
                elif allnames[j_1] == 'random_state':
                    v_random_state = combination_local[j_1]
                elif allnames[j_1] == 'max_leaf_nodes':
                    v_max_leaf_nodes = combination_local[j_1]
                elif allnames[j_1] == 'min_impurity_decrease':
                    v_min_impurity_decrease = combination_local[j_1]
                elif allnames[j_1] == 'ccp_alpha':
                    v_ccp_alpha = combination_local[j_1]
                elif allnames[j_1] == 'max_depth':
                    v_max_depth = int(combination_local[j_1])
                elif allnames[j_1] == 'min_samples_split':
                    v_min_samples_split = combination_local[j_1]
                elif allnames[j_1] == 'min_samples_leaf':
                    v_min_samples_leaf = combination_local[j_1]
                else:
                    error += 1 # Increase error by one
            
            # Define model
            models_op["combination_{}".format(i_2)] = tree.DecisionTreeClassifier(criterion = v_criterion,
                    splitter = v_splitter, min_weight_fraction_leaf = v_min_weight_fraction_leaf,
                    max_features = v_max_features, random_state = v_random_state, 
                    max_leaf_nodes = v_max_leaf_nodes, min_impurity_decrease = v_min_impurity_decrease,
                    ccp_alpha = v_ccp_alpha, max_depth = v_max_depth, min_samples_split = v_min_samples_split,
                    min_samples_leaf = v_min_samples_leaf
                    )
                    
        elif considered_model == "xgboost":
            # Initialize model parameters
            v_early_stopping_rounds = model_settings.early_stopping_rounds
            v_booster = model_settings.booster
            v_verbosity = model_settings.verbosity
            v_validate_parameters = model_settings.validate_parameters
            v_disable_default_eval_metric = model_settings.disable_default_eval_metric
            v_eta = model_settings.eta
            v_gamma = model_settings.gamma
            v_max_depth = model_settings.max_depth
            v_max_delta_step = model_settings.max_delta_step
            v_subsample = model_settings.subsample
            v_sampling_method = model_settings.sampling_method
            v_colsample_bytree = model_settings.colsample_bytree
            v_colsample_bylevel = model_settings.colsample_bylevel
            v_colsample_bynode = model_settings.colsample_bynode
            v_reg_lambda = model_settings.reg_lambda
            v_alpha = model_settings.alpha
            v_tree_method = model_settings.tree_method
            v_sketch_eps = model_settings.sketch_eps
            v_refresh_leaf = model_settings.refresh_leaf
            v_process_type = model_settings.process_type
            v_grow_policy = model_settings.grow_policy
            v_max_leaves = model_settings.max_leaves
            v_max_bin = model_settings.max_bin
            v_predictor = model_settings.predictor
            v_num_parallel_tree = model_settings.num_parallel_tree
            v_num_class = model_settings.num_class
            #v_monotone_constraints = model_settings.monotone_constraints
            #v_interaction_constraints = model_settings.interaction_constraints
            #v_n_estimators = model_settings.n_estimators
            
            # Additional hist and gpu_hist
            v_single_precision_histogram = model_settings.single_precision_histogram
            # Additional gpu_hist
            v_deterministic_histogram = model_settings.deterministic_histogram
            # Additional dart
            v_sample_type = model_settings.sample_type
            v_normalize_type = model_settings.normalize_type
            v_rate_drop = model_settings.rate_drop
            v_one_drop = model_settings.one_drop
            v_skip_drop = model_settings.skip_drop
            
            ## gblinear
            # v_reg_lambda = model_settings.reg_lambda
            # v_alpha = model_settings.alpha
            # v_updater = model_settings.updater
            v_feature_selector = model_settings.feature_selector
            v_top_k = model_settings.top_k
            ## Tweedie regression
            v_tweedie_variance_power = model_settings.tweedie_variance_power
            
            # Learning task parameters
            v_objective = model_settings.objective
            v_base_score = model_settings.base_score
            v_eval_metric = model_settings.eval_metric
            v_seed = model_settings.seed
            v_seed_per_iteration = model_settings.seed_per_iteration
            v_use_label_encoder = model_settings.use_label_encoder
            v_missing = model_settings.missing
            
            # Change values of optimalized parameters   
            for j_1 in range(0,length_allnames):
                if allnames[j_1] == 'booster':
                    v_booster = combination_local[j_1]
                elif allnames[j_1] == 'eta':
                    v_eta = float(combination_local[j_1])
                elif allnames[j_1] == 'gamma':
                    v_gamma = float(combination_local[j_1])
                elif allnames[j_1] == 'max_depth':
                    v_max_depth = int(combination_local[j_1])
                elif allnames[j_1] == 'max_delta_step':
                    v_max_delta_step = float(combination_local[j_1])
                elif allnames[j_1] == 'subsample':
                    v_subsample = float(combination_local[j_1])
                elif allnames[j_1] == 'sampling_method':
                    v_sampling_method = combination_local[j_1]
                elif allnames[j_1] == 'colsample_bytree':
                    v_colsample_bytree = float(combination_local[j_1]) 
                elif allnames[j_1] == 'colsample_bylevel':
                    v_colsample_bylevel = float(combination_local[j_1])
                elif allnames[j_1] == 'colsample_bynode':
                    v_colsample_bynode = float(combination_local[j_1])
                elif allnames[j_1] == 'reg_lambda':
                    v_reg_lambda = float(combination_local[j_1])
                elif allnames[j_1] == 'alpha':
                    v_alpha = float(combination_local[j_1])
                elif allnames[j_1] == 'tree_method':
                    v_tree_method = combination_local[j_1]
                elif allnames[j_1] == 'sketch_eps':
                    v_sketch_eps = combination_local[j_1]
                elif allnames[j_1] == 'refresh_leaf':
                    v_refresh_leaf = combination_local[j_1]
                elif allnames[j_1] == 'refresh_leaf':
                    v_refresh_leaf = combination_local[j_1]
                elif allnames[j_1] == 'process_type':
                    v_process_type = combination_local[j_1]
                elif allnames[j_1] == 'grow_policy':
                    v_grow_policy = combination_local[j_1]
                elif allnames[j_1] == 'max_leaves':
                    v_max_leaves = combination_local[j_1]  
                elif allnames[j_1] == 'max_bin':
                    v_max_bin = combination_local[j_1]
                elif allnames[j_1] == 'predictor':
                    v_predictor = combination_local[j_1]   
                elif allnames[j_1] == 'num_parallel_tree':
                    v_num_parallel_tree = combination_local[j_1]
                # elif allnames[j_1] == 'monotone_constraints':
                    # v_num_monotone_constraints = combination_local [j_1]
                # elif allnames[j_1] == 'interaction_constraints':
                    # v_interaction_constraints = combination_local [j_1]
                # elif allnames[j_1] == 'n_estimators':
                    # v_n_estimators = combination_local [j_1]
                elif allnames[j_1] == 'single_precision_histogram':
                    v_single_precision_histogram = combination_local[j_1]
                elif allnames[j_1] == 'deterministic_histogram':
                    v_deterministic_histogram = combination_local[j_1]
                elif allnames[j_1] == 'sample_type':
                    v_sample_type = combination_local[j_1]
                elif allnames[j_1] == 'normalize_type':
                    v_normalize_type = combination_local[j_1]
                elif allnames[j_1] == 'rate_drop':
                    v_rate_drop = combination_local[j_1]
                elif allnames[j_1] == 'one_drop':
                    v_one_drop = combination_local[j_1]
                elif allnames[j_1] == 'skip_drop':
                    v_skip_drop = combination_local[j_1]
                elif allnames[j_1] == 'feature_selector':
                    v_feature_selector= combination_local[j_1]
                elif allnames[j_1] == 'top_k':
                    v_top_k= combination_local[j_1]
                elif allnames[j_1] == 'tweedie_variance_power':
                    v_tweedie_variance_power= combination_local[j_1]
                else:
                        error += 1 # Increase error by one
            
            # Define model
            if regression:
                models_op["combination_{}".format(i_2)] = xgb.XGBRegressor(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                    seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                    booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                    disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                    max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                    colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                    reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                    refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                    predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, early_stopping_rounds = v_early_stopping_rounds)
            else:
                if v_booster == "gbtree":
                    if v_tree_method == "hist":
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, single_precision_histogram = v_single_precision_histogram,
                        early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                    elif v_tree_method == "gpu_hist":
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, single_precision_histogram = v_single_precision_histogram, 
                        deterministic_histogram = v_deterministic_histogram, early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                    else:
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                elif v_booster == "dart":
                    if v_tree_method == "hist":
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, single_precision_histogram = v_single_precision_histogram, 
                        sample_type = v_sample_type, normalize_type = v_normalize_type, rate_drop = v_rate_drop, one_drop = v_one_drop,
                        skip_drop = v_skip_drop, early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                    elif v_tree_method == "gpu_hist":
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, single_precision_histogram = v_single_precision_histogram,
                        deterministic_histogram = v_deterministic_histogram, sample_type = v_sample_type, normalize_type = v_normalize_type, rate_drop = v_rate_drop, 
                        one_drop = v_one_drop, skip_drop = v_skip_drop, early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                    else:
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters,
                        disable_default_eval_metric = v_disable_default_eval_metric, eta = v_eta, gamma = v_gamma, max_depth = v_max_depth,
                        max_delta_step = v_max_delta_step, subsample = v_subsample, sampling_method = v_sampling_method,
                        colsample_bytree = v_colsample_bytree, colsample_bylevel = v_colsample_bylevel, colsample_bynode = v_colsample_bynode,
                        reg_lambda = v_reg_lambda, alpha = v_alpha, tree_method = v_tree_method, sketch_eps = v_sketch_eps, 
                        refresh_leaf = v_refresh_leaf, process_type = v_process_type, grow_policy = v_grow_policy, max_leaves = v_max_leaves, max_bin = v_max_bin,
                        predictor = v_predictor, num_parallel_tree = v_num_parallel_tree, sample_type = v_sample_type, normalize_type = v_normalize_type, rate_drop = v_rate_drop, 
                        one_drop = v_one_drop, skip_drop = v_skip_drop, early_stopping_rounds = v_early_stopping_rounds, v_num_class = v_num_class)
                elif v_booster == "gblinear":
                        models_op["combination_{}".format(i_2)] = xgb.XGBClassifier(objective = v_objective, base_score = v_base_score, eval_metric = v_eval_metric,
                        seed = v_seed, seed_per_iteration = v_seed_per_iteration, use_label_encoder = v_use_label_encoder, missing = v_missing,
                        booster = v_booster, verbosity = v_verbosity, validate_parameters = v_validate_parameters, 
                        disable_default_eval_metric = v_disable_default_eval_metric, reg_lambda = v_reg_lambda, alpha = v_alpha, updater = v_updater, 
                        feature_selector = v_feature_selector, top_k = v_top_k, tweedie_variance_power = v_tweedie_variance_power, early_stopping_rounds = v_early_stopping_rounds, 
                        v_num_class = v_num_class)
        
        # Save weights
        optweights["combination_{}".format(i_2)] = {}
        for j_1 in range(0,length_allnames):
            if "class" in allnames[j_1]:
                optweights["combination_{}".format(i_2)][int(allnames[j_1].replace("class", ""))] = combination_local[j_1]

    return(models_op, number_of_combinations, optweights)

def scoring_results(y_train, y_test, predicted_train, predicted_test, 
    y_score_train, y_score_test, scoring_prediction, weights):
    """Goal:
            Get results.
        --------------------------------------
        Input:
            y_train, 
                y_test
            predicted train, predicted test:
                Classifications
            y_score_train, y_score_test:
                Probabilities
            scoring_predictions:
                Evaluation metric, dict
            """
    # Initialize function variables
    scores_training, scores_testing = {}, {}
    error_2 = 0
    
    # Initializing lists
    for key, value in scoring_prediction.items():
        # Scoring
        if key == "neg_mean_squared_error":
            if value:
                scores_training[key] = metrics.mean_squared_error(y_train, predicted_train)
                scores_testing[key] = metrics.mean_squared_error(y_test, predicted_test)
        elif key == "neg_mean_absolute_error":
            if value:
                scores_training[key] = metrics.mean_absolute_error(y_train, predicted_train)
                scores_testing[key] = metrics.mean_absolute_error(y_test, predicted_test)
        elif key == "accuracy":
            if value:
                scores_training[key] = metrics.accuracy_score(y_train, predicted_train)
                scores_testing[key] = metrics.accuracy_score(y_test, predicted_test)
        elif key == "balanced_accuracy":
            if value:
                sample_weight_train = [weights[y] for y in y_train.values.flatten()]
                sample_weight_test = [weights[y] for y in y_test.values.flatten()]
                scores_training[key] = metrics.balanced_accuracy_score(y_train, predicted_train, sample_weight = sample_weight_train)
                scores_testing[key] = metrics.balanced_accuracy_score(y_test, predicted_test, sample_weight = sample_weight_test)
        elif key == "top_k_accuracy":
            if value:
                scores_training[key] = metrics.top_k_accuracy_score(y_train, y_score_train)
                scores_testing[key] = metrics.top_k_accuracy_score(y_test, y_score_test)
        elif key == "average_precision":
            if value:
                scores_training[key] = metrics.average_precision_score(y_train, y_score_train)
                scores_testing[key] = metrics.average_precision_score(y_test, y_score_test)
        elif key == "neg_brier_score":
            if value:
                scores_training[key] = metrics.brier_score_loss(y_train, y_score_train)
                scores_testing[key] = metrics.brier_score_loss(y_test, y_score_test)
        elif key == "f1_score":
            if value:
                scores_training[key] = metrics.f1_score(y_train, predicted_train)
                scores_testing[key] = metrics.f1_score(y_test, predicted_test)
        elif key == "f1_micro":
            if value:
                scores_training[key] = metrics.f1_score(y_train, predicted_train)
                scores_testing[key] = metrics.f1_score(y_test, predicted_test)
        elif key == "f1_macro":
            if value:
                scores_training[key] = metrics.f1_score(y_train, predicted_train, average = "macro", labels = np.unique(y_train))
                scores_testing[key] = metrics.f1_score(y_test, predicted_test, average = "macro", labels = np.unique(y_train))
        elif key == "f1_weighted":
            if value:
                scores_training[key] = metrics.f1_score(y_train, predicted_train, average = "weighted", labels = np.unique(y_train))
                scores_testing[key] = metrics.f1_score(y_test, predicted_test, average = "weighted", labels = np.unique(y_train))
        elif key == "f1_samples":
            if value:
                scores_training[key] = metrics.f1_score(y_train, predicted_train)
                scores_testing[key] = metrics.f1_score(y_test, predicted_test)
        elif key == "neg_log_loss":
            if value:
                scores_training[key] = metrics.log_loss(y_train, predicted_train)
                scores_testing[key] = metrics.log_loss(y_test, predicted_test)
        elif key == "precision":
            if value:
                scores_training[key] = metrics.precision_score(y_train, predicted_train)
                scores_testing[key] = metrics.precision_score(y_test, predicted_test)
        elif key == "recall":
            if value:
                scores_training[key] = metrics.recall_score(y_train, predicted_train)
                scores_testing[key] = metrics.recall_score(y_test, predicted_test)
        elif key == "jaccard":
            if value:
                scores_training[key] = metrics.jaccard_score(y_train, predicted_train)
                scores_testing[key] = metrics.jaccard_score(y_test, predicted_test)
        elif key == "roc_auc":
            if value:
                scores_training[key] = metrics.roc_auc_score(y_train, y_score_train)
                scores_testing[key] = metrics.roc_auc_score(y_test, y_score_test)
        elif key == "roc_auc_ovr":
            if value:
                scores_training[key] = metrics.roc_auc_score(y_train, y_score_train)
                scores_testing[key] = metrics.roc_auc_score(y_test, y_score_test)
        elif key == "roc_auc_ovo":
            if value:
                scores_training[key] = metrics.roc_auc_score(y_train, y_score_train)
                scores_testing[key] = metrics.roc_auc_score(y_test, y_score_test)
        elif key == "roc_auc_ovr_weighted":
            if value:
                scores_training[key] = metrics.roc_auc_score(y_train, y_score_train, multi_class = "ovr", average = "weighted", labels = np.unique(y_train))
                scores_testing[key] = metrics.roc_auc_score(y_test, y_score_test, multi_class = "ovr", average = "weighted", labels = np.unique(y_train))
        elif key == "roc_auc_ovo_weighted":
            if value:
                scores_training[key] = metrics.roc_auc_score(y_train, y_score_train)
                scores_testing[key] = metrics.roc_auc_score(y_test, y_score_test)
        else:
            error_2 +=1
        
    return(scores_training, scores_testing)

def ranking (train_results, test_results, models_op, scoring, direction):
    """Goal:
            Function ranks the results.
        -------------------------------------------------------------
        Input:
            train_results, test_results, models_op, scoring, direction
        --------------------------------------------------------------
        Output:
            rank"""
    # Initialize variables
    rank, models = {}, []
    for cb, value in models_op.items():
        models.append(value)

    # Sort arrays
    for score in scoring:
        std = test_results[score]["std"]
        diff_std = test_results[score]["diff_std"]

        if direction[score] == False:
            mean = np.multiply(test_results[score]["mean"], -1) # Change order
            diff_mean = np.multiply(test_results[score]["diff_mean"], -1)
            rank[score] = np.lexsort((diff_std, std, diff_mean, mean))
        elif direction[score] == True:
            mean = test_results[score]["mean"]
            diff_mean = test_results[score]["diff_mean"]
            rank[score] = np.lexsort((diff_std, std, diff_mean, mean))
    
    # Set and write results
    for score in scoring:
        tr_0 = [models[i] for i in rank[score]]
        tr_1 = train_results[score]["mean"][rank[score]]
        tr_2 = train_results[score]["std"][rank[score]]
        tr_3 = test_results[score]["mean"][rank[score]]
        tr_4 = test_results[score]["std"][rank[score]]
        tr_5 = test_results[score]["diff_mean"][rank[score]]
        tr_6 = test_results[score]["diff_std"][rank[score]]
        
        d_op = {'models': tr_0, "mean_train": tr_1, "std_train": tr_2, 'mean_test': tr_3, 
            'std_test': tr_4, 'diff_mean': tr_5, 'diff_std': tr_6}
        df_op = pd.DataFrame(data = d_op)
        df_op.to_excel("optimal_{}".format(score) + ".xlsx")
    return(rank)
    

def testing(data, model, scoring_prediction, scoring, n_splits, considered_model, y_name, headers, target, strategy, 
            verbose = None, optweights = None, regression = False):
    """Goal:
        Evaluates the chosen model on the test sets"""
    # Initialize variables
    d_results_1, d_results_2 = {}, {}
    testing_train, testing_test = {}, {}
    for score in scoring:
        testing_test [score] = []
        testing_train[score] = []

    # Set iterations
    if strategy == "prediction":
        n_splits = 1

    # Get data
    for i_3 in range (1, (n_splits + 1)):
        if strategy == "testing":
            # Unpack train and test sets
            X_train, y_train  = data['fold_{}'.format((i_3))]['X_train'], data['fold_{}'.format((i_3))]['y_train']
            X_predict, y_predict = data['fold_{}'.format((i_3))]['X_test'], data['fold_{}'.format((i_3))]['y_test']
        else:
            X_train, y_train = data[0], data[1]
            X_predict, y_predict = data[2], data[3]
        
        if considered_model == "xgboost":
            # Get eval set
            X_train, X_eval, y_train, y_eval, index_train, index_eval = training_prediction(X_train, y_train, 0.1, not regression)

        ## Get current balance
        if not regression:
            ys = np.unique(y_train)
            weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = ys, y = y_train.values.flatten())
            weights_original = {num: weights[num] for num in ys}
            # Manipulate weights
            weights = deepcopy(weights_original)
            if optweights != []:
                for key, val in weights.items():
                    weights[key] = val * optweights[key]
            # Set weights
            w = [weights[kind] for kind in y_train.values.flatten()]
        # Fit and predict
        if considered_model == "xgboost":
            if not regression:
                model.min_child_weight = min(weights.values())
                model_fitted = model.fit(X_train, y_train.values.flatten(), sample_weight = w, eval_set = [(X_eval, y_eval.values.flatten())], verbose = verbose)
            else:
                model.min_child_weight = 5
                model_fitted = model.fit(X_train, y_train.values.flatten(), eval_set = [(X_eval, y_eval.values.flatten())], verbose = verbose)
        else:
            model.class_weight = weights
            model_fitted = model.fit(X_train, y_train)
            
        # Get scores 
        predicted_train = model_fitted.predict(X_train) # Classification train
        predicted_test = model_fitted.predict(X_predict) # Classification test
        if not regression:
            y_score_train = model_fitted.predict_proba(X_train) # Probability train for class labeled as 1
            y_score_test = model_fitted.predict_proba(X_predict) # Probability test for class labeled as 1
            # Write confusion matrix
            df_confusion = pd.DataFrame(data = metrics.confusion_matrix(y_predict, predicted_test, 
                normalize = "true"))    
            df_confusion_2 = pd.DataFrame(data = metrics.confusion_matrix(y_predict, predicted_test, 
                normalize = None))
            writer_confusion = pd.ExcelWriter("confusion{}".format(i_3) + ".xlsx")
            df_confusion_2.to_excel(writer_confusion, sheet_name = "Absolute")
            df_confusion.to_excel(writer_confusion, sheet_name = "Normalized")
            writer_confusion.save()
        else:
            weights_original = None
            y_score_train = None
            y_score_test = None

        #    ---- Return figures (1)
        # Return figure of tree
        if considered_model == "decision_tree":
            tree.plot_tree(model_fitted, feature_names= headers, class_names = [str(t) for t in target])
            plt.show()
        
        # Return Shapley plot
        if ("f1_score" in scoring or "f1_weighted" in scoring) and strategy != "testing":
            shap.initjs()
            explainer = shap.TreeExplainer(model_fitted)
            shap_values = explainer.shap_values(X_predict, check_additivity = False)
            shap.summary_plot(shap_values, features=X_predict, feature_names=X_predict.columns, max_display=None, plot_type=None, 
                                color=None, axis_color='#333333', title=None, alpha=1, show=True, sort=True, color_bar=True,
                                plot_size='auto', layered_violin_max_num_bins=20, class_names=target, class_inds=target,
                                color_bar_label='Feature value', auto_size_plot=None, use_log_scale=False)
            # Plot per label
            for label in target:
                shaps = shap_values[label]
                shap.summary_plot(shaps, features=X_predict, feature_names=X_predict.columns, title = str(label), show = True)
        
        # Return feature importance
        if considered_model == 'xgboost' and strategy != "testing":
            xgb.plot_importance(model_fitted, show_values = False, xlabel = '', max_num_features = 10, importance_type = 'gain')
            plt.show()
        
        # ---- Get results
        score_train, score_test = scoring_results(y_train, y_predict, predicted_train, 
            predicted_test, y_score_train, y_score_test, scoring_prediction, weights_original)
        for score in scoring:
            # Append results
            testing_train[score].append(score_train[score])
            testing_test [score].append(score_test[score])
            if strategy == "testing":
                if i_3 == n_splits:
                    # Set final results
                    d_results_1["train_" + score] = testing_train[score]
                    d_results_1["test_" + score] = testing_test [score]
                    d_results_2["mean_train_" + score] = statistics.mean(testing_train[score])
                    d_results_2["std_train_" + score] = statistics.stdev(testing_train[score])
                    d_results_2["mean_test_" + score] = statistics.mean(testing_test[score])
                    d_results_2["std_test_" + score] = statistics.stdev(testing_test[score])
            else:
                d_results_1["train_" + score] = testing_train[score]
                d_results_1["test_" + score] = testing_test [score]
        
    # Export to excel
    if strategy == "testing":
        df_results_1 = pd.DataFrame(data = d_results_1)
        df_results_2 = pd.DataFrame(data = d_results_2, index = [0])
        writer = pd.ExcelWriter("results.xlsx")
        df_results_1.to_excel(writer, sheet_name = "Sheet1")
        df_results_2.to_excel(writer, sheet_name = "Sheet2")
        writer.save()
    else:
        plt.plot(y_predict, predicted_test, "o")
        plt.show()
        df_results_1 = pd.DataFrame(data = d_results_1)
        writer = pd.ExcelWriter("results_prediction.xlsx")
        df_results_1.to_excel(writer, sheet_name = "Sheet1")
        writer.save()

