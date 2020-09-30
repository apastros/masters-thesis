import pathlib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from timeit import default_timer as timer
from sklearn.preprocessing import StandardScaler

# Time the function
start_time = timer()

# Import data from csv to DataFrame
# data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass10000_apastros.csv'
# data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass_apastros.csv'
data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'Cleaned_Scaled_Data.csv'

print("Importing data from ",data_filepath,"...")
specObj_data = pd.read_csv(data_filepath,low_memory=False)
specObj_data_df = pd.DataFrame(specObj_data)

# Set X and y
print("Analyzing data with the following features:")
# specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
#         'cModelMag_z', 'z']
# specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
#         'cModelMag_z', 'z','g-r','r-i','i-z','u-g']
specObj_features = ['z']
X = pd.DataFrame(specObj_data_df[specObj_features],columns=specObj_features)

y = specObj_data_df['class']

print(specObj_features)


# Segregate into training and testing data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Get time of fit
time_start_fit = timer()
# Train model on training data 
print("Training model on data...")
model = RandomForestClassifier(random_state = 1)
model.fit(X_train,y_train.values.ravel())
print("Model fit in ", timer()-time_start_fit, "seconds")

# Get time of predict
time_start_predict = timer()
# Use model to predict
print("Using model to create predictions...")
predictions = model.predict(X_test)
print("Model predicted in ",timer()-time_start_predict, "seconds")

# Compare predictions with true values and print accuracy
print("Finding error...")
accuracy = accuracy_score(y_test,predictions,normalize=True)
print("Accuracy is ",accuracy*100, "%")

# Obtain confusion matrix
print("The following is the confusion matrix: \n",confusion_matrix(y_test,predictions))

# End timing the function
print("Elapsed time is ",timer() - start_time,"seconds")