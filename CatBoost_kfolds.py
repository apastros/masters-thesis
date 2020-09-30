import pathlib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
# from sklearn.metrics import accuracy_score
import pandas as pd
from timeit import default_timer as timer

# Time the function
start_time = timer()

# Import data from csv to DataFrame
# specObj_data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass10000_apastros.csv'
specObj_data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass_apastros.csv'

print("Importing data from ",specObj_data_filepath,"...")
specObj_data = pd.read_csv(specObj_data_filepath,low_memory=False)
specObj_data_df = pd.DataFrame(specObj_data)

# Set X and y
print("Analyzing data with the following features:")
specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
       'cModelMag_z', 'z']
X = pd.DataFrame(specObj_data_df[specObj_features],columns=specObj_features)
y = pd.DataFrame(specObj_data_df['class'],columns=['class'])
print(specObj_features)

# Segregate into training and testing data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Get time of fit
time_start_fit = timer()
# Train model on training data 
print("Training model on data...")
model = CatBoostClassifier(random_state = 1,task_type='GPU',verbose=False)
scores = cross_val_score(model, X, y, cv =10)
print("Model fit in ", timer()-time_start_fit, "seconds")

# Compare predictions with true values and print accuracy
print("Finding error...")
mean_accuracy = scores.mean()
std_accuracy = scores.std()
print("Accuracy is ",mean_accuracy, "% +/- ",std_accuracy)

# End timing the function
print("Elapsed time is ",timer() - start_time,"seconds")