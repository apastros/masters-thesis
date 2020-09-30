import pathlib
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import json

# Import data from csv to DataFrame
data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'Cleaned_Scaled_Data.csv'

print("Importing data from ",data_filepath,"...")
specObj_data = pd.read_csv(data_filepath,low_memory=False)
specObj_data_df = pd.DataFrame(specObj_data)

# Set X and y
print("Analyzing data with the following features:")
specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
        'cModelMag_z', 'z','g-r','r-i','i-z','u-g']
X = pd.DataFrame(specObj_data_df[specObj_features],columns=specObj_features)

y = specObj_data_df['class']

print(specObj_features)

# Segregate into training and testing data
print("Performing cross fold validation and reporting the mean scores...")

model_mean_accuracy = {"Model Name": "Mean cross val score"}

# Trains model through k cross folds and appends the model's name and mean accuracy to the dictionary model_mean_accuracy
def train_model(model,X,y,k):
    model_name = type(model).__name__
    print("Utilizing the following model: ",model_name)
    start_time = timer()
    pipeline = make_pipeline(StandardScaler(),model)
    score = cross_val_score(pipeline,X,y,cv=k)
    model_mean_accuracy[model_name] = np.mean(score)
    print("Model evaluated in ",start_time,"seconds")
    return

# Models to be tested
model_list = [RandomForestClassifier(random_state=1),
              SVC(random_state=1),
              xgb.XGBClassifier(random_state=1),
              GradientBoostingClassifier(random_state = 1),
              LGBMClassifier(random_state=1),
              CatBoostClassifier(task_type="GPU",verbose=False)]

# Running through the list of models
for model in model_list:
    train_model(model,X,y,10)

# Outputting 
with open("model_evaluation.json",'w') as outfile:
    json.dump(model_mean_accuracy, outfile)
    