import pathlib
import pandas as pd
import matplotlib.pyplot as plt

# Import data from csv to DataFrame
data_filepath = pathlib.Path.cwd() / 'Thesis Data' / 'Cleaned_Scaled_Data.csv'

print("Importing data from ",data_filepath,"...")
specObj_data = pd.read_csv(data_filepath,low_memory=False)
specObj_data_df = pd.DataFrame(specObj_data)

# Set X and y
print("Analyzing data with the following features:")
specObj_features = ['cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
        'cModelMag_z', 'z','g-r','r-i','i-z','u-g']
X_df = pd.DataFrame(specObj_data_df[specObj_features],columns=specObj_features)

y = specObj_data_df['class']

print(specObj_features)



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