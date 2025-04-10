import math
from sklearn import tree
from sklearn import metrics
from skopt.space import Real, Integer
import xgboost as xgb
import numpy as np

class Settings_general():
    """A class to store all general settings."""
    def __init__(self):
        """Goal:
            Initialize the settings."""
        
        # --------- Target and variables
        self.y_name = "mood"
        self.nvars = 11

        # --------- Classification
        # ------ Labels
        self.labels = None

        # ------ Multiclass classification
        self.num_class = None # Number of classes

        # ------ Set proc_traintest and proc_predict to None if no prediction set is required or if an external prediction set is provided. self.proc_traintest = None # proportion of data in train-testset
        self.proc_predict = None # % proportion of data in predictset
        self.n_splits = 10 # Number of splits for k-fold analysis
        
        # ------ Set model considered ("decision_tree", "xgboost")
        self.considered_model = "xgboost"
        
        # ------ Scoring methods
        ## Method(s) used during gridsearch
        ## Binary
        # self.scoring = {'neg_log_loss': metrics.make_scorer(metrics.log_loss, 
            # greater_is_better = False, needs_proba = True)
            # } # Log loss, binary and multiclass
        # self.scoring = {'auc': metrics.make_scorer(metrics.roc_auc_score, 
            # greater_is_better = True, needs_proba = True)
            # } # Auc, binary
        # self.scoring = {'f1_score': metrics.make_scorer(metrics.f1_score, 
        #   greater_is_better = True, needs_proba = False), 'auc': metrics.make_scorer(metrics.roc_auc_score, 
        #   greater_is_better = True, needs_proba = True) 
        #   } # Precision recall curve, binary
        # Multi-class
        self.scoring =  {'f1_score': metrics.make_scorer(metrics.f1_score, 
                             average = 'balanced', labels = self.labels, greater_is_better = True, needs_proba = False), 
                        'auc': metrics.make_scorer(metrics.roc_auc_score, multi_class = "ovr", average = "weighted", labels = self.labels,
                                           greater_is_better = True, needs_proba = True)}
        # Direction used by GridSearch to sort for scoring method.
        # False means descending, True means ascending  
        # self.direction = {'neg_log_loss': True}  # Log loss
        #self.direction = {'roc_auc': False} # Auc
        self.direction = {'balanced_accuracy': False, 'f1_macro': False, 'f1_weighted': False, 'f1_score': False,
                          'roc_auc': False, 'roc_auc_ovr_weighted': False, "neg_mean_squared_error": True,
                          "neg_mean_absolute_error": True} # F1_score

        # Method(s) used during testing and during prediction on the prediction set, see 
        # https://scikit-learn.org/stable/modules/model_evaluation.html
        # True will be used and False not.
        # Note: f1 and roc_auc scores are not yet specified in functions. This should be implemented in "functions" before use.
        self.scoring_prediction = {"accuracy": False, "balanced_accuracy": False, "top_k_accuracy": False, "average_precision": False,
            "neg_brier_score": False, "f1_score": False, "f1_micro": False, "f1_macro": False, "f1_weighted": False, "f1_samples": False,
            "neg_log_loss": False, "precision": False, "recall": False, "jaccard": False, "roc_auc": False, "roc_auc_ovr": False,
            "roc_auc_ovo": False, "roc_auc_ovr_weighted": False, "roc_auc_ovo_weighted": False, "neg_mean_squared_error": True,
            "neg_mean_absolute_error": True
            }
        
class Settings_decisiontree():
    """Goal:
        A class to store all settings of the decision tree program."""
    def __init__(self):
        """Goal:
            Initialize the settings of the decision tree."""
        
        # ------ Attributes, set gridsearch attributes to []
        self.v_criterion = 'gini'
        self.v_splitter = 'best'
        self.v_min_weight_fraction_leaf = 0.0
        self.v_max_features = None
        self.v_random_state = 1
        self.v_max_leaf_nodes = None
        self.v_min_impurity_decrease = []
        self.v_ccp_alpha = 0.0
        self.v_max_depth = []
        self.v_min_samples_split = 2
        self.v_min_samples_leaf = 1
        
        # For sake of simplicity non-gridsearch attributes as tree_class instance (no change required)
        self.tree_class = tree.DecisionTreeClassifier(criterion = self.v_criterion,
            splitter = self.v_splitter, min_weight_fraction_leaf = self.v_min_weight_fraction_leaf,
            max_features = self.v_max_features, random_state = self.v_random_state, 
            max_leaf_nodes = self.v_max_leaf_nodes, min_impurity_decrease = self.v_min_impurity_decrease,
            ccp_alpha = self.v_ccp_alpha, max_depth = self.v_max_depth, min_samples_split = self.v_min_samples_split,
            min_samples_leaf = self.v_min_samples_leaf
            ) # Pre-specified settings for gridsearch
            
        # ------ Gridsearch attributes
        # Parameter values to be evaluated by gridsearch
        self.param_grid = {'max_depth': [1, 2, 3, 10], 
            'min_impurity_decrease': [0.0, 0.005, 0.01, 0.1]
            }
        
        ## Bayesian attributes
        # Parameter values to be evaluated by Bayesian optimization
        self.params = ['max_depth', 'min_impurity_decrease']
        self.space  = [Integer(1, 10, name='max_depth'), Real(0, 0.1, "uniform", name='min_impurity_decrease')]
            
        # ------ Optimal model, define yourself if you would not like to adhere to the optimal settings for the first scoring method
        self.model = tree.DecisionTreeClassifier(max_depth = 2, criterion = 'gini', splitter = 'best', min_weight_fraction_leaf = 0.0,
            max_features = None, random_state = 1, max_leaf_nodes = None, min_impurity_decrease = 0.01, ccp_alpha = 0.0,
            min_samples_leaf = 1, min_samples_split = 2)
            
class Settings_xgboost():
    """Goal:
        A class to store all settings of the XGBoost model."""
    def __init__(self):
        """Goal:
            Initialize the settings."""
        # Initialize general settings
        set_gen = Settings_general()
        # ------ Attributes, set gridsearch attributes and non-considered attributes to []
        # General
        self.booster = "gbtree" # "gbtree", "dart", "gblinear"
        self.verbosity = 0 # 0(silent), 1(warning), 3(debug)
        self.validate_parameters = True # Check whether parameters are used or not
        self.disable_default_eval_metric = 1 # 1 means disabling default metric
        
        # gbtree and dart 
        self.eta = [] # learning rate, between 0 and 1 (typical: 0.01-0.2)
        self.gamma = [] # min_split_loss, between 0 and infinity (might be interesting)
        self.max_depth = [] # Between 0 and infinity (typical: 3-10)
        self.max_delta_step = [] # Between 0 and infinity
        self.subsample = [] # Percentage of samples considered
        self.sampling_method = 'uniform' # Method of sampling, 'uniform'/'gradient_based'
        self.colsample_bytree = [] # subsample columns for each tree, between 0 and 1
        self.colsample_bylevel = [] # subsample columns for each level, between 0 and 1
        self.colsample_bynode = [] # subsample columns for each node, between 0 and 1
        self.reg_lambda = [] # l2 regularization term on weights
        self.alpha = [] # l1 regularization term on weights
        self.tree_method = "auto" # "approx"/"hist"/"gpu_hist"/"auto"/"exact"
        self.sketch_eps = 0.03 # Only if tree_method = approx
        self.refresh_leaf = 1 # 1 updates leaf and node stats, 0 only node stats
        self.process_type = "default" # 
        self.grow_policy = "depthwise" # If tree_method is hist or gpu_hist
        self.max_leaves = 0 # Only if grow_policy = lossguide
        self.max_bin = 256 # Only if hist or gpu_hist
        self.predictor = "auto" #
        self.num_parallel_tree = 1 #
        self.num_class = set_gen.num_class
        #self.monotone_constraints = [] #
        #self.interaction_constraints = [] #
        #self.n_estimators = []
        
        # Additional hist and gpu_hist
        self.single_precision_histogram = "false"
        # Additional gpu_hist
        self.deterministic_histogram = "true"
        # Additional dart
        self.sample_type = "uniform"
        self.normalize_type = "tree"
        self.rate_drop = 0.0 # Between 0 and 1
        self.one_drop = 0 
        self.skip_drop = 0.0 # Between 0 and 1
        
        ## gblinear
        # self.reg_lambda = [] # l2 regularization
        # self.alpha = [] # l1 regularization
        # self.updater = [] #
        self.feature_selector = "cyclic"
        self.top_k = 0 
        ## Tweedie regression
        self.tweedie_variance_power = 1.5 # Between 1 and 2
        
        # Learning task parameters
        self.objective = "reg:squarederror"#"reg:absoluteerror" #"reg:squarederror" #"multi:softmax" #"binary:logistic" 
        self.base_score = 0 #0.5 
        self.eval_metric = "rmse" #"mae"  # "mlogloss" #"logloss" 
        # self.eval_metric = "auc"
        self.seed = 42 
        self.seed_per_iteration = "false"
        self.use_label_encoder = False
        self.missing = np.nan
        self.early_stopping_rounds = 10
        # https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html
        
        # # For sake of simplicity non-gridsearch attributes as xgboost instance (no change required)
        if set_gen.labels == None: # Regression
            self.clf_xgb = xgb.XGBRegressor(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds)
        else:
            if self.booster == "gbtree":
                if self.tree_method == "hist":
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, single_precision_histogram = self.single_precision_histogram, early_stopping_rounds = self.early_stopping_rounds,
                        num_class = self.num_class)
                elif self.tree_method == "gpu_hist":
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, single_precision_histogram = self.single_precision_histogram, 
                        deterministic_histogram = self.deterministic_histogram, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
                else:
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
            if self.booster == "dart":
                if self.tree_method == "hist":
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, single_precision_histogram = self.single_precision_histogram, 
                        sample_type = self.sample_type, normalize_type = self.normalize_type, rate_drop = self.rate_drop, one_drop = self.one_drop,
                        skip_drop = self.skip_drop, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
                elif self.tree_method == "gpu_hist":
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, single_precision_histogram = self.single_precision_histogram,
                        deterministic_histogram = self.deterministic_histogram, sample_type = self.sample_type, normalize_type = self.normalize_type, rate_drop = self.rate_drop, 
                        one_drop = self.one_drop, skip_drop = self.skip_drop, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
                else:
                    self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                        disable_default_eval_metric = self.disable_default_eval_metric, eta = self.eta, gamma = self.gamma, max_depth = self.max_depth,
                        max_delta_step = self.max_delta_step, subsample = self.subsample, sampling_method = self.sampling_method,
                        colsample_bytree = self.colsample_bytree, colsample_bylevel = self.colsample_bylevel, colsample_bynode = self.colsample_bynode,
                        reg_lambda = self.reg_lambda, alpha = self.alpha, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                        refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                        predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, sample_type = self.sample_type, normalize_type = self.normalize_type, rate_drop = self.rate_drop, 
                        one_drop = self.one_drop, skip_drop = self.skip_drop, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
            if self.booster == "gblinear":
                self.clf_xgb = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                        seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                        booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters, 
                        disable_default_eval_metric = self.disable_default_eval_metric, reg_lambda = self.reg_lambda, alpha = self.alpha, updater = self.updater, 
                        feature_selector = self.feature_selector, top_k = self.top_k, tweedie_variance_power = self.tweedie_variance_power, early_stopping_rounds = self.early_stopping_rounds, 
                        num_class = self.num_class)
        
        # ------ Gridsearch attributes
        # Parameter values to be evaluated by gridsearch
        self.param_grid = {'max_depth': [1, 2, 3, 4, 5], "eta": [0.1, 0.3, 0.5, 0.7, 1.0], 'subsample': [0.5, 0.7, 1.0],
                           "gamma": [0, 10], "max_delta_step": [0.5, 10], "reg_lambda": [1, 5], 
                           "class0": [1], "class1": [1], "class2": [1]}
        
        ## Bayesian attributes
        # Parameter values to be evaluated by Bayesian optimization
        self.params = ['max_depth', "eta", 'subsample', "gamma",
                        "max_delta_step", "reg_lambda",
                        "class0", "class1", "class2", "colsample_bytree", "colsample_bylevel",
                         "alpha", "colsample_bynode"]
                        
        self.space  = [Integer(3, 10, name='max_depth'), Real(0.01, 0.3, "log-uniform", name='eta'), Real(0.5, 1.0, "uniform", name='subsample'),
          Real(0, 20, "uniform", name="gamma"), Real(0, 10, "uniform", name="max_delta_step"), Real(1, 10, "uniform", name="reg_lambda"),
          Real(1, 2, "uniform", name="class0"), Real(1, 2, "uniform", name="class1"), Real(1, 2, "uniform", name="class2"),
          Real(0.5, 1, "uniform", name="colsample_bytree"), Real(0.5, 1, "uniform", name="colsample_bylevel"), Real(0, 100, "uniform", name="alpha"),
          Real(math.sqrt(set_gen.nvars) / set_gen.nvars, 1, "uniform", name="colsample_bynode")]
        
        ## Optimal model, define yourself if you would not like to adhere to the optimal settings for the first scoring method###
        # # Absolute error
        # self.model = xgb.XGBRegressor(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
        #             seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
        #             booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
        #             disable_default_eval_metric = self.disable_default_eval_metric, eta = 0.3, gamma = 7.47863269814187, max_depth = 3,
        #             max_delta_step = 2.80012381949565, subsample = 1, sampling_method = self.sampling_method,
        #             colsample_bytree = 1, colsample_bylevel = 1, colsample_bynode = 1,
        #             reg_lambda = 10, alpha = 8.32742685017453, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
        #             refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
        #             predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
        # self.optweights  = {0: 1, 1: 1, 2: 1}
        # mean-squared
        self.model = xgb.XGBRegressor(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
                    seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
                    booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
                    disable_default_eval_metric = self.disable_default_eval_metric, eta = 0.3, gamma = 7.14198416065053, max_depth = 3,
                    max_delta_step = 0, subsample = 1, sampling_method = self.sampling_method,
                    colsample_bytree = 0.964642530923503, colsample_bylevel = 0.5, colsample_bynode = 0.306410956715352,
                    reg_lambda = 10, alpha = 0, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
                    refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
                    predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
        self.optweights  = {0: 1, 1: 1, 2: 1}
        # # balanced-accuracy
        # self.model = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
        #             seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
        #             booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
        #             disable_default_eval_metric = self.disable_default_eval_metric, eta = 0.01, gamma = 11.3083559973573, max_depth = 4,
        #             max_delta_step = 6.7885033513493, subsample = 1, sampling_method = self.sampling_method,
        #             colsample_bytree = 0.548499718297719, colsample_bylevel = 0.764242354831908, colsample_bynode = 0.437802624976385,
        #             reg_lambda = 5.897832060941, alpha = 16.787495947288, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
        #             refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
        #             predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
        # self.optweights  = {0: 2, 1: 2, 2: 2}
        # # f1-macro
        # self.model = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
        #             seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
        #             booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
        #             disable_default_eval_metric = self.disable_default_eval_metric, eta = 0.045632935304887, gamma = 3.07049736887377, max_depth = 4,
        #             max_delta_step = 8.37482016548168, subsample = 1, sampling_method = self.sampling_method,
        #             colsample_bytree = 0.905579794453456, colsample_bylevel = 0.540241351597251, colsample_bynode = 0.301511344577764,
        #             reg_lambda = 10, alpha = 7.45826476857336, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
        #             refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
        #             predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
        # self.optweights  = {0: 1, 1: 1.2932245424338, 2: 1}

        # # f1-weighted
        # self.model = xgb.XGBClassifier(objective = self.objective, base_score = self.base_score, eval_metric = self.eval_metric,
        #             seed = self.seed, seed_per_iteration = self.seed_per_iteration, use_label_encoder = self.use_label_encoder, missing = self.missing,
        #             booster = self.booster, verbosity = self.verbosity, validate_parameters = self.validate_parameters,
        #             disable_default_eval_metric = self.disable_default_eval_metric, eta = 0.01, gamma = 8.09533536613185, max_depth = 4,
        #             max_delta_step = 10, subsample = 1, sampling_method = self.sampling_method,
        #             colsample_bytree = 0.580171146372335, colsample_bylevel = 0.819560021928965, colsample_bynode = 0.339635599852816,
        #             reg_lambda = 1, alpha = 0, tree_method = self.tree_method, sketch_eps = self.sketch_eps, 
        #             refresh_leaf = self.refresh_leaf, process_type = self.process_type, grow_policy = self.grow_policy, max_leaves = self.max_leaves, max_bin = self.max_bin,
        #             predictor = self.predictor, num_parallel_tree = self.num_parallel_tree, early_stopping_rounds = self.early_stopping_rounds, num_class = self.num_class)
        # self.optweights  = {0: 1.41731860123474, 1: 2, 2: 1.18343555434691}