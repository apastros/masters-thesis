import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass_apastros.csv'
# data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'SpecPhotoObjectsClass10000_apastros.csv'

print("Importing data from ",data_filepath,"...")
specObj_data = pd.read_csv(data_filepath,low_memory=False)
specObj_data_df = pd.DataFrame(specObj_data)

# Set X and y
specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
       'cModelMag_z', 'z','class']
X = pd.DataFrame(specObj_data_df[specObj_features],columns=specObj_features)
derived_columns = ['g-r','r-i','i-z','u-g','class']
X_derived = [X['cModelMag_g']-X['cModelMag_r'],X['cModelMag_r']-X['cModelMag_i'],X['cModelMag_i']-X['cModelMag_z'],X['cModelMag_u']-X['cModelMag_g'],X['class']]
X_derived = pd.DataFrame(X_derived).transpose()
X_derived.columns = derived_columns
X = X.drop('class',axis=1)
X = pd.concat([X,X_derived],axis=1,sort=False)

print("Cleaning data...")

# Remove errors from data
def no_outliers(cModelMag,max_deviations = 100):
    mean = np.mean(cModelMag)
    standard_deviation = np.std(cModelMag)
    distance_from_mean = abs(cModelMag - mean)
    not_outlier = distance_from_mean < max_deviations * standard_deviation
    cModelMag = cModelMag[not_outlier]
    return(cModelMag)

for i in X:
    if i != 'class':
        X[i] = no_outliers(X[i])
X.dropna(inplace=True)
specObj_features = specObj_features[:-1]
y = X['class']
X = X.drop(['class'],axis=1)

print("Exporting data to csv...")
X_df = pd.DataFrame(X,columns=['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
       'cModelMag_z', 'z','g-r','r-i','i-z','u-g'])
to_csv = pd.concat([X_df,y],axis=1,sort=False)
new_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'Cleaned_Scaled_Data.csv'
to_csv = to_csv.dropna()
to_csv.to_csv(new_filepath)


g_r = [X_df['g-r']]
r_i = [X_df['r-i']]
i_z = [X_df['i-z']]
u_g = [X_df['u-g']]

plt.scatter(g_r,r_i,s=1)
plt.xlabel('g-r')
plt.ylabel('r-i')
plt.title('g-r vs r-i')
plt.show()

plt.scatter(g_r,i_z,s=1)
plt.xlabel('g-r')
plt.ylabel('i-z')
plt.title('g-r vs i-z')
plt.show()

plt.scatter(g_r,u_g,s=1)
plt.xlabel('g-r')
plt.ylabel('u-g')
plt.title('g-r vs u-g')
plt.show()

plt.scatter(r_i,i_z,s=1)
plt.xlabel('r-i')
plt.ylabel('i-z')
plt.title('r-i vs i-z')
plt.show()

plt.scatter(r_i,u_g,s=1)
plt.xlabel('r-i')
plt.ylabel('u-g')
plt.title('r-i vs u-g')
plt.show()

plt.scatter(i_z,u_g,s=1)
plt.xlabel('i-z')
plt.ylabel('u-g')
plt.title('i-z vs u-g')
plt.show()

# plt.hist(z)