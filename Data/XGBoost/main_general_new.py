import functions_general2 as fun
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from settings_general import Settings_general
from settings_general import Settings_decisiontree
from settings_general import Settings_xgboost

# Loading data
source_excel = "C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\mining1\\Dataset7dayremoval_numeric\\trainNielssss.csv"
#df = pd.read_excel (source_excel, header = 0)
df_original = pd.read_csv (source_excel, header = 0)
# Tijdelijk !!!
df = df_original.drop(["date", "id", "activity", "sleep hours", "diff sp500", "circumplex.valence"], axis = 1)
df = df.rename(columns = {'Rain[0.1mm]': 'Rain'})

# Tijdelijk !!!

# Initialize settings
gen_settings = Settings_general()
dt_settings = Settings_decisiontree()
xgb_settings = Settings_xgboost()
pd.set_option("display.max_rows", None, "display.max_columns", None) # Set display properties

# Seperate x from y data
y = df.pop(gen_settings.y_name).to_frame()
X = df 

# Make a numpy array consisting of the headers
headers = df.columns.tolist()
if gen_settings.labels != None:
    regression = False
    target = np.array(gen_settings.labels)
else:
	regression = True
	target = None


# ---- Seperate prediction and train-test data
# Separate train-test data from prediction data
if not regression:
	(X_train_test, X_predict, y_train_test, y_predict, index_train_test, index_predict) = fun.training_prediction(X,y, gen_settings.proc_predict)
else:
	(X_train_test, X_predict, y_train_test, y_predict, index_train_test, index_predict) = fun.training_prediction(X,y, gen_settings.proc_predict, stratisfy = False)

# ---- Seperate train-test data in train and test data
if not regression:
	folds, indices, cv = fun.train_splitter(X_train_test, y_train_test, gen_settings.n_splits)
else:
	folds, indices, cv = fun.train_splitter(X_train_test, y_train_test, gen_settings.n_splits, stratisfy = False)

# ---- Optimalization
while True:
	print("\nWould you like to use optimalization procedures?: (yes/no)")
	answer_4 = input("\t")
	if answer_4.title() == "Yes":
		print("\nWhich procedure would you like to use?: (gridsearch/bayesian)")
		while True:
			answer_5 = input("\t")
			if answer_5.title() == "Gridsearch":
				if gen_settings.considered_model == "decision_tree":
					folds_op, indices_op, models_op, train_results, test_results, rank, scoring, optweights = fun.optimize_model(
						dt_settings, dt_settings.param_grid, 
						folds, indices, cv, gen_settings.n_splits,
						gen_settings.considered_model, gen_settings.scoring_prediction,
						gen_settings.direction
						)
				elif gen_settings.considered_model == "xgboost":
					folds_op, indices_op, models_op, train_results, test_results, rank, scoring, optweights = fun.optimize_model(
						xgb_settings, xgb_settings.param_grid, 
						folds, indices, cv, gen_settings.n_splits,
						gen_settings.considered_model, gen_settings.scoring_prediction,
						gen_settings.direction, regression
						)
			elif answer_5.title() == "Bayesian":
				if gen_settings.considered_model == "decision_tree":
					scoring = fun.optimize_model_bayesian(predefined_model = dt_settings.tree_class, space = dt_settings.space, params = dt_settings.params, cv = cv,
						scoring_prediction = gen_settings.scoring_prediction, n_splits = gen_settings.n_splits, folds = folds, 
						direction = gen_settings.direction, considered_model = "decision_tree", verbosity = False)
				elif gen_settings.considered_model == "xgboost":
					scoring = fun.optimize_model_bayesian(predefined_model = xgb_settings.clf_xgb, space = xgb_settings.space, params = xgb_settings.params, cv = cv,
						scoring_prediction = gen_settings.scoring_prediction, n_splits = gen_settings.n_splits, folds = folds, 
						direction = gen_settings.direction, considered_model = "xgboost", verbosity = xgb_settings.verbosity, regression = regression)
			else:
				print("\nYou did not type a method, try again:")
			break
		# Make choices for model evaluation on test set and/or prediction set
		print("\nThe settings of the final model which will be evaluated on test" 
			+ " and/or prediction data can be set to the optimal settings"
			+ " or to manual settings. Would you like to use manual settings? (yes/no): ")
		
		while True:
			answer = input("\t")
			if answer.title() == "No":
				while True:
					print("The scores evaluated are: ", scoring)
					answer_3 = input("For which score would you like to use the optimal settings?: ")
					print("In the excel files the scores for each combinations can be found")
					answer_cb = input("\tWhich combination would you like to use? (1st combination is 0): ")
					if str(answer_3.lower()) in scoring:
						# Get rank needed
						if answer_5.title() == "Bayesian":
							bayesian_results = pd.read_excel("bayesian_{}".format(answer_3.lower()) + ".xlsx", header = 0)
							if gen_settings.considered_model == "xgboost":
								models_op, number_of_combinations, optweights = fun.create_models_optimalization (xgb_settings, gen_settings.considered_model, xgb_settings.params, 
									bayesian = True, results = bayesian_results, row = answer_cb, regression = regression)
							else:
								models_op, number_of_combinations, optweights = fun.create_models_optimalization (dt_settings, gen_settings.considered_model, dt_settings.params, 
									bayesian = True, results = bayesian_results, row = answer_cb)
							# Get optimal model
							model = models_op["combination_{}".format(answer_cb)]
							optweights = optweights["combination_{}".format(answer_cb)]
						else:
							rank_needed = rank[answer_3][int(answer_cb)]
							model = models_op["combination_{}".format(rank_needed)]
							optweights = optweights["combination_{}".format(rank_needed)]
						break
					else:
						print("\nYou did not type a score present in the list of scores, try again:")
				break
			elif answer.title() == "Yes":
				while True:
					answer_2 = input ("Did you already change these parameters? (yes/no): ")
					if answer_2.title() == "Yes":
						if gen_settings.considered_model == "decision_tree":
							model = dt_settings.model
						elif gen_settings.considered_model == "xgboost":
							model = xgb_settings.model
						break
					elif answer_2.title() == "No":
						print("\nYou can change the settings of the model parameters in 'settings_general'")
						exit()
					else:
						print("\nYou did not type 'yes' or 'no', try again:")
				break
			else:
				print("You did not type 'yes' or 'no', try again:")
			
		break
	elif answer_4.title() == "No":
		scoring = []
		for scoring_method, value in gen_settings.scoring_prediction.items():
			if value == True:
				scoring.append(scoring_method)
		if gen_settings.considered_model == "decision_tree":
			model = dt_settings.model
			optweights = dt_settings.optweights
		elif gen_settings.considered_model == "xgboost":
			model = xgb_settings.model
			optweights = xgb_settings.optweights
		break


# ---- Evaluate the model on the test set
if gen_settings.considered_model == "xgboost":
	fun.testing(folds, model, gen_settings.scoring_prediction, scoring, 
		gen_settings.n_splits, gen_settings.considered_model, gen_settings.y_name, headers, target, 
		strategy = "testing", verbose = xgb_settings.verbosity, optweights=optweights, regression = regression)
else:
	fun.testing(folds, model, gen_settings.scoring_prediction, scoring, 
		gen_settings.n_splits, gen_settings.considered_model, gen_settings.y_name, headers, target, strategy = "testing", 
		optweights=optweights)

# ---- Evaluate the model on the prediction set
source_excel = "C:\\Users\\niels\\OneDrive\\Documenten\\GitHub\\mining1\\Dataset7dayremoval_numeric\\predictNielssss.csv"
df_original = pd.read_csv(source_excel, header = 0)
# Tijdelijk !!!
df = df_original.drop(["date", "id", "activity", "sleep hours", "diff sp500", "circumplex.valence"], axis = 1)
df = df.rename(columns = {'Rain[0.1mm]': 'Rain'})

# Seperate x from y data
y_predict = df.pop(gen_settings.y_name).to_frame()
X_predict = df 
# Predict
fun.testing([X_train_test, y_train_test, X_predict, y_predict], model, gen_settings.scoring_prediction, scoring, 
	gen_settings.n_splits, gen_settings.considered_model, gen_settings.y_name, headers, target, 
	strategy = "prediction", optweights=optweights, regression = regression)