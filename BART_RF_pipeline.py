import pandas as pd#, pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier
import gpboost as gpb
import shap, copy
import os

# if binary, use classification, if continuous, use regression
def rf_interaction(x,y, binary):
    if binary:
        rf = RandomForestClassifier(n_estimators = 50, random_state = 12)
    else:
        rf = RandomForestRegressor(n_estimators = 50, random_state = 12)
    rf.fit(x, y.values.ravel())
    explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")
    shap_vals = np.abs(explainer.shap_interaction_values(x))
    interaction_matrix = pd.DataFrame(shap_vals.mean(0),columns=list(x), index = list(x))
    interation_matrix_self_interact_removed=interaction_matrix.copy()
    for i in np.arange(interaction_matrix.shape[0]):
        interation_matrix_self_interact_removed.iloc[i,i]=0
    return interation_matrix_self_interact_removed

class GPBoostClassifier:
    def __init__(self,scan=False,use_coords=False,random_state=42,boosting_type='gbdt'):
        self.scan=scan
        self.use_coords=use_coords
        self.random_state=random_state
        self.boosting_type=boosting_type
    
    def fit(self,X,y,groups,coords=None):
        data_train = gpb.Dataset(X, y)
        self.gp_model = gpb.GPModel(group_data=groups, likelihood="gaussian", gp_coords=coords.values if self.use_coords else None,cov_function="exponential")
        params = {'learning_rate': 1e-1, 'min_data_in_leaf': 10, #'objective': "binary",
                  'verbose': 0}
        if self.boosting_type!='gbdt':
            assert self.boosting_type in ['rf','dart']
            params['boosting']=self.boosting_type
        params['n_jobs']=1
        num_boost_round = 600
        
        if self.scan:
            param_grid_small = {'learning_rate': [0.1,0.01,0.001], 'min_data_in_leaf': [20,50,100],
                                'max_depth': [5,10,15], 'max_bin': [255,1000], 'use_nesterov_acc': [False,True]}
            opt_params = gpb.grid_search_tune_parameters(param_grid=param_grid_small,
                                                         params=params,
                                                         num_try_random=15,
                                                         folds=list(GroupShuffleSplit(random_state=42).split(X,y,groups)),
                                                         gp_model=self.gp_model,
                                                         use_gp_model_for_validation=True,
                                                         train_set=data_train,
                                                         verbose_eval=1,
                                                         num_boost_round=num_boost_round,#50 
                                                         early_stopping_rounds=10,
                                                         seed=1,
                                                         metrics='root_mean_squared_error') 

            params=opt_params['best_params']

        self.gpm = gpb.train(params=params,
                    train_set=data_train,
                    gp_model=self.gp_model,
                    num_boost_round=num_boost_round,
                    
                   )
        return self

    def interaction_extract(self,x,y,group):
        explainer = shap.TreeExplainer(self.gpm, feature_perturbation="tree_path_dependent")
        shap_vals = np.abs(explainer.shap_interaction_values(x, group))
        interaction_matrix = pd.DataFrame(shap_vals.mean(0),columns=list(x), index = list(x))
        interation_matrix_self_interact_removed=interaction_matrix.copy()
        for i in np.arange(interaction_matrix.shape[0]):
            interation_matrix_self_interact_removed.iloc[i,i]=0
        return interation_matrix_self_interact_removed

# store the matrix
def get_interaction_from_matrix(interaction_matrix):
    interaction_matrix = copy.deepcopy(interaction_matrix)
    interaction_matrix = interaction_matrix.where(np.triu(np.ones(interaction_matrix.shape)).astype(np.bool_))
    interaction_matrix = interaction_matrix.stack().reset_index()
    interaction_matrix.columns = ['x', 'y', 'value']
    interaction_matrix = interaction_matrix.sort_values(by = "value", ascending = False)
    return list(zip(interaction_matrix.iloc[0:2, 0], interaction_matrix.iloc[0:2, 1]))    

def rf_gpboost_pipeline(foldername):
    if "Binary" in foldername:
        binary = True
    else:
        binary = False
    # go to specified folder in the same directory as this script
    os.chdir(foldername)
    # create an empty dataframe to store results, with following columns: filename, rf_interaction, gpboost_interaction
    results = pd.DataFrame(columns = ["filename", "rf_interaction", "gpboost_interaction"])
    # for each file in the folder
    for filename in os.listdir():
        # read csv file as dataframe
        data = pd.read_csv(filename).iloc[:, 1:].iloc[:, : -1] 
        x = data.iloc [:, : -1] 
        y = data.iloc [:, -1 :] 
        group = pd.read_csv(filename).iloc[:, 1:]["batch.assignment"]
        # get rf interaction
        rf_interaction_matrix = rf_interaction(x,y, binary)
        rf_interaction = get_interaction_from_matrix(rf_interaction_matrix)
        # get gpboost interaction
        gpboost_interaction_matrix = GPBoostClassifier().fit(x,y,group).interaction_extract(x,y,group)
        gpboost_interaction = get_interaction_from_matrix(gpboost_interaction_matrix)
        # add results to the dataframe
        results = results.append({"filename": filename, "rf_interaction": rf_interaction, "gpboost_interaction": gpboost_interaction}, ignore_index = True)
    # save results to a csv file
    results.to_csv("results.csv")


if __name__ == "__main__":
    folders = [f for f in os.listdir() if os.path.isdir(f)]
    for folder in folders:
        rf_gpboost_pipeline(folder)