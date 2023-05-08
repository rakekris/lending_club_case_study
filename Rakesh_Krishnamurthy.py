#!/usr/bin/env python
# coding: utf-8

# # Lending Club Case Study Introduction
# 
# The case study is for a consumer finance company which specialises in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:
# 
# If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
# 
# If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company

# # Data understanding
# 
# The case study provides a loan.csv which consists of 39717 records with 111 columns in a comma seperated file. 
# #####  Below are the insights gathered looking at the data
# 1. There are columns with all the values in a row as marked as na, which add no value to the analysis   
# 2. The id or member id can be considered as unique identification column as it has no duplicates 
# 3. The driving column identifed so far are "loan_status","
# 4. It's also noticed that the Data Dictionary has 116 columns but the loan.csv has 111 columns 

# In[1]:


#import python libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from scipy.stats import norm


# In[2]:


#quick take on the data
pd.set_option('display.max_columns', 120)
pd.set_option('display.max_rows', 120)
df = pd.read_csv("G://My Drive//Education//Master//artifacts"
                 "//upgrad//lending club case study//data//loan//loan.csv", low_memory=False)
print(df.shape)
print(df.info())
print(df.dtypes)
df.describe()
#total rows : 39717


# In[3]:


#check missing values in data

#check the nulls row wise

null_rows = df.isnull().all(axis=1).sum()
print('Count of empty Rows: {}'.format(null_rows))
# Review: There are 0 rows with missing values

#check the nulls column wise
null_columns = df.isnull().all(axis=0).sum()
print('Count of Columns with missing values: {}'.format(null_columns))
#Review: There are 54 columns with missing values


# # Data Cleaning and Manipulation

# In[4]:


### drop the variables which have only nan values or all values are missing

df = df.dropna(axis=1, how='all')

print("data shape is {}".format(df.shape))
df.describe()


# In[5]:


#check for variables with nulls

null_percent = df.isnull().mean() * 100
print(null_percent)


# In[6]:


#dropping variables which have more > 50% null as they unlikely to provide meaningful insights, including them in analsysis might mislead 

columns_to_drop = ["mths_since_last_delinq","mths_since_last_record","next_pymnt_d"]
df.drop(labels = columns_to_drop, axis =1, inplace=True)
print("data shape is {}".format(df.shape))
df.describe()


# In[7]:


#dropping additional variables where mean,min, max are zeros or same as they unlikely to provide meaningful insights

columns_to_drop = ["collections_12_mths_ex_med","policy_code","acc_now_delinq", "chargeoff_within_12_mths", "delinq_amnt","tax_liens"]
df.drop(labels = columns_to_drop, axis =1, inplace=True)
print("data shape is {}".format(df.shape))
df.describe()


# In[8]:


# Check for duplicate rows across the data set based on column: id

duplicate_id_count = len(df[df.duplicated(subset='id', keep=False)])
duplicate_mem_id_count = len(df[df.duplicated(subset='id', keep=False)])

print("No of duplicate ids: {}".format(duplicate_id_count))

print("No of duplicate members ids: {}".format(duplicate_mem_id_count))

#review : There are no duplicate ids and member ids


# In[9]:


# dropping variables which are not useful for the analysis

columns_to_drop = ["id","url","desc"]
df.drop(labels = columns_to_drop, axis =1, inplace=True)
print("data shape is {}".format(df.shape))
df.describe()


# ### Data typing/casting before analysis

# In[10]:


### remove % and cast int_rate to float, cast issue_d and earliest_cr_line into date


df['int_rate'] = df['int_rate'].map(lambda x: str(x).rstrip('%')).astype(float)
df['revol_util'] = df['revol_util'].map(lambda x: str(x).rstrip('%')).astype(float)
df['issue_d'] = pd.to_datetime(df['issue_d'], format ='%b-%y')
df['last_pymnt_d'] = pd.to_datetime(df['last_pymnt_d'], format ='%b-%y')
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format ='%b-%y')
df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format ='%b-%y')
print("data shape is {}".format(df.shape))
df.describe()


# In[11]:


print("pre cleaning :")
print(df.pub_rec_bankruptcies.isnull().sum())
print("post cleaning :")
df.pub_rec_bankruptcies.fillna('Not Known',inplace=True)
print(df.pub_rec_bankruptcies.isnull().sum())


# In[12]:


#checking min and max for all the date variables

df[['earliest_cr_line','last_credit_pull_d','last_pymnt_d','issue_d']].agg(['min','mean','max'])

#review : seems like the earliest_cr_line has a future date , logic needs to be reviewed


# In[13]:


#logic to handle dates such as earliest_cr_line which are beyond 1970
def date_correction_beyond_1970(date):
    if date > pd.to_datetime('2012-01-01'):
        return date - pd.DateOffset(years=100)
    else:
        return date

#correcting earliest_cr_line as it has unrealistic dates due to a datetime limitation  
df['earliest_cr_line_corrected'] = df['earliest_cr_line'].apply(date_correction_beyond_1970)    


# In[14]:


# check outliers for the annual_inc using box plot, based on the min, quantiles and max calculated in previous cells

fig, ax = plt.subplots()


ax.hist(df['annual_inc'], bins=10, density=True, alpha=0.6)

mean = df['annual_inc'].mean()
std_dev = df['annual_inc'].std()


x_axis = np.linspace(df['annual_inc'].min(), df['annual_inc'].max(), 100)


curve = norm.pdf(x_axis, mean, std_dev)


ax.plot(x_axis, curve, color='red')


ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Distribution of annual_inc')


plt.show()
df['annual_inc'].describe()
#review, the graph looks skewed based on the annual_inc


# In[15]:


#removing the outliers using Interquartile ranges  as data is skewed

# calculate the first quartile (Q1)
Q1 = df['annual_inc'].quantile(0.25)

# calculate the third quartile (Q3)
Q3 = df['annual_inc'].quantile(0.75)

# calculate the interquartile range (IQR)
IQR = Q3 - Q1

# determine the lower bound
lower_bound = Q1 - (1.5 * IQR)

# determine the upper bound
upper_bound = Q3 + (1.5 * IQR)

# remove outliers from DataFrame
df = df[(df['annual_inc'] >= lower_bound) & (df['annual_inc'] <= upper_bound)]


# In[16]:


# distribution after outlier removal
fig, ax = plt.subplots()

# plot histogram for 'annual_inc'
ax.hist(df['annual_inc'], bins=10, density=True, alpha=0.6)

# calculate mean and standard deviation for 'Variable'
mean = df['annual_inc'].mean()
std_dev = df['annual_inc'].std()

# create range of values for the x-axis
x_axis = np.linspace(df['annual_inc'].min(), df['annual_inc'].max(), 100)

# generate normal distribution curve
curve = norm.pdf(x_axis, mean, std_dev)

# plot normal distribution curve
ax.plot(x_axis, curve, color='red')

# add axis labels and title
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Distribution of annual_inc')

# show plot
plt.show()

df['annual_inc'].describe()

#review : the distribution looks normal post outlier removal


# # Identify key variable for the case study and perform univariate analysis

# In[17]:


# Based on the data dictionary shared , loan_status is the key variable to perform the analysis

count = df['loan_status'].value_counts()
count.plot(kind='bar', title = 'loan_status')
print("printing the % share of each loan status")
print(df.loan_status.value_counts(normalize=True)*100)
print("printing the count of each loan status")
print(count)
print("data shape is {}".format(df.shape))


# # Data Analysis

# In[18]:


#derived a new variable to represent the loan_status in numeric format; 
# loan_status_num:  fully paid = 0 , Charged off = 1 and current = 2

df["loan_status_num"] = [1 if x=="Charged Off" else 2 if x=="Current" else 0 for x in df['loan_status']]

count = df['loan_status_num'].value_counts()
count.plot(kind='bar', title = 'loan_status_num')

print(df.loan_status_num.value_counts(normalize=True)*100)


print(count)
print("data shape is {}".format(df.shape))


# In[19]:


#derive new variables extracted from date fields
df['issue_yr'] = df['issue_d'].dt.year
df['issue_month'] = df['issue_d'].dt.month

df['earliest_cr_line_yr'] = df['earliest_cr_line_corrected'].dt.year
df['earliest_cr_line_month'] = df['earliest_cr_line_corrected'].dt.month


# In[20]:


# Correlaration matrix for all the numeric variables
f = plt.figure(figsize=(29, 29))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=20, rotation=90)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=20)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=20)
plt.title('Correlation Matrix', fontsize=20);

#review : There is very less correlation between the variables except loan_amnt, funded_amnt and funded_amnt_inv


# In[21]:


### Get correlation matrix to compare the correlation efficient between charged_off and other variables
df_corr=df.select_dtypes(['number']).corr()
df_corr


# In[22]:


#Generic function to generate box plot out of a given variables with loan_status_num
def generate_box_plot_by_var_loan_status_num(df,x_col,y_col):
    palette = sns.color_palette(['#2ca02c', '#ff7f0e', '#1f77b4'])
    colors = ['#2ca02c', '#ff7f0e', '#1f77b4']
    labels = ['Fully Paid', 'Charged Off', 'Current']
    legend_dict = dict(zip(labels, colors))
    legend_handles = [plt.Rectangle((0,0), 1, 1, color=color, label=label) for label, color in legend_dict.items()]
    sns.boxplot(x=x_col, y=y_col, data=df, palette=palette)
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.show()


# In[23]:


x_col = 'loan_status_num'
y_col = 'total_rec_prncp'

generate_box_plot_by_var_loan_status_num(df=df,x_col=x_col,y_col=y_col)


# In[24]:


x_col = 'loan_status_num'
y_col = 'total_acc'


generate_box_plot_by_var_loan_status_num(df=df,x_col=x_col,y_col=y_col)


# In[25]:


x_col = 'loan_status_num'
y_col = 'loan_amnt'

generate_box_plot_by_var_loan_status_num(df=df,x_col=x_col,y_col=y_col)


# In[26]:


x_col = 'loan_status_num'
y_col = 'int_rate'

generate_box_plot_by_var_loan_status_num(df=df,x_col=x_col,y_col=y_col)


# # bivariate analysis

# In[27]:


#Generic function to generate a bar plot out of a given variable with loan_status
def generate_bar_plot_by_var_vs_loan_status(df,col_vs_loan_status):
    #create a pivot with a given variable and loan_status
    col_vs_loan_status_df = df.groupby([col_vs_loan_status, 'loan_status']).loan_status.count().unstack().fillna(0).reset_index()
    col_vs_loan_status_df['Total'] = col_vs_loan_status_df['Charged Off'] + col_vs_loan_status_df['Current'] + col_vs_loan_status_df['Fully Paid'] 
    col_vs_loan_status_df['Chargedoff_percent'] = col_vs_loan_status_df['Charged Off'] / col_vs_loan_status_df['Total']
    col_vs_loan_status_df.columns.set_names(['row_id'], inplace = True)
    
    #barplot generation based on the pivot
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.set_title('{} vs Charged off'.format(col_vs_loan_status),fontsize=20,color = 'b')
    ax1=sns.barplot(y='Chargedoff_percent', x=col_vs_loan_status, data=col_vs_loan_status_df, width=0.2)
    ax1.set_ylabel(col_vs_loan_status,fontsize=12,color='w')
    ax1.set_xlabel(col_vs_loan_status,fontsize=12,color = 'b')
    ax1.set_ylabel('Chargedoff ratio',fontsize=12,color = 'b')
    plt.show()

    return col_vs_loan_status_df.sort_values('Chargedoff_percent', ascending=False)


# In[28]:


#term versus loan_status
col_vs_loan_status='term'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review : customers who took 60 month term loan defaulted twice than who did 36 months term


# In[29]:


#grade versus loan_status
col_vs_loan_status='grade'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review: customers whose grade increased from ‘A’ through ‘G’ , there is a linear increase in the default cases, G being the highest


# In[30]:


#sub_grade versus loan_status
col_vs_loan_status='sub_grade'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review : customers whose grade increased from ‘A-X’ through ‘G-X’ , there is (close to) linear increase in the default cases, F5 being the highest, similar to grade


# In[31]:


#analysis by home ownership

col_vs_loan_status = 'home_ownership'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review : based on the below there is no conclusive evidence that home ownership would help us decide whether customer would default or not


# In[32]:


#analysis by verification status
col_vs_loan_status = 'verification_status'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

# There seems to be not much difference in defaulties based on the verfication status hence it would not be driving factor.


# In[33]:


#analysis by issue year

col_vs_loan_status = 'issue_yr'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review : highest defaults during recession year


# In[34]:


#analysis by purpose

col_vs_loan_status = 'purpose'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#review : customers whose purpose of loan was small_business tend to default


# In[35]:


#analysis by address state 


col_vs_loan_status = 'addr_state'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

#although customers from NE stands who defaulted, there is only few data points which cannot be conclusive. 
#But cusotmers from NV , close 21.6 % have defaulted


# In[36]:


# analysis by pub_rec_bankruptcies

col_vs_loan_status = 'pub_rec_bankruptcies'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

# customers who have public record bankruptcies are likely to default


# In[37]:


# analysis by pub_rec

col_vs_loan_status = 'pub_rec'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)

# similar to cutomers with bankruptcies: cutomers with derogatory public records are likely to default


# In[38]:


# create a decade bins (derived variable) for the earliest_cr_line_yr field
df['earliest_cr_line_decade'] = pd.cut(df['earliest_cr_line_yr'], bins=range(1900, 2030, 10), labels=range(1900, 2020, 10))


col_vs_loan_status = 'earliest_cr_line_decade'
generate_bar_plot_by_var_vs_loan_status(df=df,col_vs_loan_status=col_vs_loan_status)
#review : There is no effective measure that we can derive from the below analysis


# In[ ]:




