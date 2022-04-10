
# coding: utf-8

# In[1]:


#install packages
get_ipython().system('pip install pandas')
get_ipython().system('pip install numpy')
get_ipython().system('pip install scipy')
get_ipython().system('pip install sklearn')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install missingno')


# In[2]:


#import
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import missingno as msno
import scipy.stats as stats
import seaborn


# In[4]:


#load excel file
data = pd.read_csv('C:/Users/blien/Documents/WGU/D206/Medical/medical_data.csv',dtype={'locationid':np.int64}) 
data.head()


# In[5]:


#removing unnecessary column at beginning
del data[data.columns[0]]


# In[6]:


#Gather information on dataset
data.info()


# In[7]:


#renaming columns to better describe them
data.rename(columns = {'Item1':'Timely admission',
'Item2':'Timely treatment',
'Item3':'Timely visits',
'Item4':'Reliability',
'Item5':'Options',
'Item6':'Hours of treatment',
'Item7':'Courteous staff',
'Item8':'Active listening from Doctor'},
inplace=True)


# In[8]:


#testing for duplicates across all rows
med_duplicates = data.duplicated()
print(med_duplicates)


# In[9]:


#checking to see if there are duplicates
data.duplicated().sum()


# In[10]:


#testing to see which rows are duplicated
data[med_duplicates]


# In[11]:


#identify how many records are null in each field
data_nulls = data.isnull().sum()
print(data_nulls)


# In[12]:


#store these missing values
missing_values = data.isnull().any(axis= 1)
data[missing_values]


# In[13]:


#identify whether fields have any NA values 
data.isna().any()


# In[14]:


#looking over the data types of our variables
data.dtypes


# In[15]:


# Examine columns for misspellings in categorical variables
data['Area'].unique()


# In[16]:


data['Marital'].unique()


# In[67]:


data['Education'].unique()


# In[17]:


data['Employment'].unique()


# In[18]:


data['Services'].unique()


# In[19]:


data['Age'].unique()


# In[20]:


data['State'].unique()


# In[21]:


data['Soft_drink'].unique()


# In[22]:


data['Complication_risk'].unique()


# In[23]:


#reexpressing variables from (T/F) to numeric
data['CR_numeric'] = data['Complication_risk']
dict_CR = {'CR_numeric':{'Low': 0, 'Medium': 1, 'High': 2, 'unknown': np.NaN}}
data.replace(dict_CR, inplace = True)
data['CR_numeric'].unique()


# In[24]:


data['SoftD_numeric'] = data['Soft_drink']
dict_SoftD = {'SoftD_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_SoftD, inplace = True)
data['SoftD_numeric'].unique()


# In[25]:


data['ReAd_numeric'] = data['ReAdmis']
dict_ReAd = {'ReAd_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_ReAd, inplace = True)
data['ReAd_numeric'].unique()


# In[26]:


data['HB_numeric'] = data['HighBlood']
dict_HB = {'HB_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_HB, inplace = True)
data['HB_numeric'].unique()


# In[27]:


data['Stroke_numeric'] = data['Stroke']
dict_Stroke = {'Stroke_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Stroke, inplace = True)
data['Stroke_numeric'].unique()


# In[28]:


data['Art_numeric'] = data['Arthritis']
dict_Art = {'Art_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Art, inplace = True)
data['Art_numeric'].unique()


# In[29]:


data['Diabetes_numeric'] = data['Diabetes']
dict_Diabetes = {'Diabetes_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Diabetes, inplace = True)
data['Diabetes_numeric'].unique()


# In[30]:


data['Hyper_numeric'] = data['Hyperlipidemia']
dict_Hyper = {'Hyper_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Hyper, inplace = True)
data['Hyper_numeric'].unique()


# In[31]:


data['BP_numeric'] = data['BackPain']
dict_Back = {'BP_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Back, inplace = True)
data['BP_numeric'].unique()


# In[32]:


data['AR_numeric'] = data['Allergic_rhinitis']
dict_AR = {'AR_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_AR, inplace = True)
data['AR_numeric'].unique()


# In[33]:


data['RE_numeric'] = data['Reflux_esophagitis']
dict_RE = {'RE_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_RE, inplace = True)
data['RE_numeric'].unique()


# In[34]:


data['Ast_numeric'] = data['Asthma']
dict_Ast = {'Ast_numeric':{'No': 0, 'Yes': 1, 'unknown': np.NaN}}
data.replace(dict_Ast, inplace = True)
data['Ast_numeric'].unique()


# In[35]:


data.info()


# In[36]:


# Create intial histograms of important variables
data.hist(['Children', 'Age', 'Income', 'SoftD_numeric', 'Overweight','Anxiety','Initial_days'])


# In[37]:


#print which variables have nulls
print(data_nulls)


# In[38]:


#perfoming the imputation on our variables
data['Children'] = data['Children'].fillna(data['Children'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Income'] = data['Income'].fillna(data['Income'].median())
data['Anxiety'] = data['Anxiety'].fillna(data['Anxiety'].median())
data['Initial_days'] = data['Initial_days'].fillna(data['Initial_days'].median())
data['Overweight'] = data['Overweight'].fillna(data['Overweight'].median())
data['Soft_drink'] = data['Soft_drink'].fillna(data['Soft_drink'].mode()[0])
data['SoftD_numeric'] = data['SoftD_numeric'].fillna(data['SoftD_numeric'].median())


# In[39]:


#testing if there are any remaining null values
data.isnull().sum()


# In[40]:


#creating histograms from our null positive variables
data.hist(['Children', 'Age', 'Income', 'SoftD_numeric', 'Overweight','Anxiety','Initial_days'])


# In[42]:


df_stats = data


# In[43]:


#calculting z score for income variable
df_stats['Z_Score_Income'] = stats.zscore(df_stats['Income'])
df_stats[['Income','Z_Score_Income']].head


# In[44]:


#plotting z score in a histogram
plt.hist(df_stats['Z_Score_Income'])


# In[45]:


#calculating z score for Children variable
df_stats['Z_Score_Children']=stats.zscore(df_stats['Children'])
df_stats[['Children','Z_Score_Children']].head


# In[46]:


#plotting z score in a histogram
plt.hist(df_stats['Z_Score_Children'])


# In[47]:


#showing our variables in a boxplot
df_stats.boxplot(['Children', 'Income', 'Initial_days'])


# In[48]:


#taking a closer look at children variable in a boxplot
boxplot_children=seaborn.boxplot(x='Children',data=data)


# In[49]:


#taking a closer look at income variable in a boxplot
boxplot_income=seaborn.boxplot(x='Income',data=data)


# In[50]:


#drop z scores from clean CSV
data.drop(columns=data.columns[-2:], axis=1,  inplace=True)


# In[101]:


#export clean data to CSV file
data.to_csv('C:/Users/blien/Documents/WGU/D206/Medical/medical_data_output.csv')


# In[51]:


#define features for the PCA
PCA_med = data[['Initial_days', 'Doc_visits', 'TotalCharge', 'Additional_charges', 'Income', 'Hours of treatment', 'Timely admission', 'Timely treatment']]


# In[52]:


#normalize the data
med_normalized = ((PCA_med-PCA_med.mean())/PCA_med.std())


# In[53]:


#select # of components to extract
pca = PCA(n_components = PCA_med.shape[1])


# In[54]:


#Use PCA application and convert the dataset of 8 variables into 8 components
pca.fit(med_normalized)


# In[56]:


#create list of PC names
pca_med = pd.DataFrame(pca.transform(med_normalized), columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6', 'PC7', 'PC8'])


# In[57]:


#load components
loadings = pd.DataFrame(pca.components_.T, columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5','PC6', 'PC7', 'PC8'], index = med_normalized.columns)


# In[58]:


print(loadings)


# In[59]:


#Extract the eigenvalues
cov_matrix = np.dot(med_normalized.T, med_normalized)/ PCA_med.shape[0]
eigenvalues = [np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)) for eigenvector in pca.components_]


# In[60]:


#plot the eigenvalues
plt.plot(eigenvalues)
plt.xlabel('Number of Components')
plt.ylabel('Eigen Values')
plt.show()

