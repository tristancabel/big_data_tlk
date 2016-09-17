
# import
import numpy as np
import pandas as pd

data = pd.read_csv("train.csv", index_col="Loan_ID")

##1 – Boolean Indexing
#
#What do you do, if you want to filter values of a column based on conditions from another set of columns? For instance, we want a list of all females who are not graduate and got a loan. Boolean indexing can help here. You can use the following code:

data.loc[(data["Gender"]=="Female") &
         (data["Education"]=="Not Graduate") &
         (data["Loan_Status"]=="Y"), ["Gender","Education","Loan_Status"]]

## 2 - Apply Function
#
#Create a new function:
def num_missing(x):
  return sum(x.isnull())

#Applying per column:
print "Missing values per column:"
print data.apply(num_missing, axis=0) #axis=0 defines that function is to be applied on each column

#Applying per row:
print "\nMissing values per row:"
print data.apply(num_missing, axis=1).head() #axis=1 defines that function is to be applied on each row


## 3 -  Imputing missing files
#
# ‘fillna()’ does it in one go. It is used for updating missing values with the overall mean/mode/median of the column. Let’s impute the ‘Gender’, ‘Married’ and ‘Self_Employed’ columns with their respective modes.

#First we import a function to determine the mode
from scipy.stats import mode
mode(data['Gender'])
# Output: ModeResult(mode=array([‘Male’], dtype=object), count=array([489]))

#This returns both mode and count. Remember that mode can be an array as there can be multiple values with high frequency. We will take the first one by default always using:

mode(data['Gender']).mode[0]
#'male'

#Now we can fill the missing values and check using technique #2.

#Impute the values:
data['Gender'].fillna(mode(data['Gender']).mode[0], inplace=True)
data['Married'].fillna(mode(data['Married']).mode[0], inplace=True)
data['Self_Employed'].fillna(mode(data['Self_Employed']).mode[0], inplace=True)

#Now check the #missing values again to confirm:
print data.apply(num_missing, axis=0)


## 4 – Pivot Table
#
#Pandas can be used to create MS Excel style pivot tables. For instance, in this case, a key column is “LoanAmount” which has missing values. We can impute it using mean amount of each ‘Gender’, ‘Married’ and ‘Self_Employed’ group. The mean ‘LoanAmount’ of each group can be determined as:

#Determine pivot table
impute_grps = data.pivot_table(values=["LoanAmount"], index=["Gender","Married","Self_Employed"], aggfunc=np.mean)
print impute_grps

#                                Loan amount
# Gender  Married Self_Employed
# Female    No       No           114.67 
#                    Yes          125.8
#           Yes      No           134.2
#                    Yes          282.2
# Male      No       No           129.4
#                    Yes          180.4
#           Yes      No           153.9
#                    Yes          169.4

##5 - Multi-Indexing
#
#If you notice the output of step #3, it has a strange property. Each index is made up of a combination of 3 values. This is called Multi-Indexing. It helps in performing operations really fast.
#
#Continuing the example from #3, we have the values for each group but they have not been imputed.
#This can be done using the various techniques learned till now.

#iterate only through rows with missing LoanAmount
for i,row in data.loc[data['LoanAmount'].isnull(),:].iterrows():
  ind = tuple([row['Gender'],row['Married'],row['Self_Employed']])
  data.loc[i,'LoanAmount'] = impute_grps.loc[ind].values[0]

#Now check the #missing values again to confirm:
print data.apply(num_missing, axis=0)


## 6 - Crosstab
#
#This function is used to get an initial “feel” (view) of the data. Here, we can validate some basic hypothesis. For instance, in this case, “Credit_History” is expected to affect the loan status significantly. This can be tested using cross-tabulation as shown below:

pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True)
# Loan_Status     N   Y   All
# credit_history
#  0.0            82  7   89
#  1.0            97  378 475
#  All            192 422 614

#These are absolute numbers. But, percentages can be more intuitive in making some quick insights. We can do this using the apply function:

def percConvert(ser):
  return ser/float(ser[-1])
  pd.crosstab(data["Credit_History"],data["Loan_Status"],margins=True).apply(percConvert, axis=1)
7.2 - perc crosstab
# Loan_Status     N     Y     All
# credit_history
#  0.0            0.92  0.07  1
#  1.0            0.20  0.80  1
#  All            0.31  0.69  1

#Now, it is evident that people with a credit history have much higher chances of getting a loan as 80% people with credit history got a loan as compared to only 9% without credit history.
#
#But that’s not it. It tells an interesting story. Since I know that having a credit history is super important, what if I predict loan status to be Y for ones with credit history and N otherwise. Surprisingly, we’ll be right 82+378=460 times out of 614 which is a whopping 75%!













## 8 – Sorting DataFrames
#
#Pandas allow easy sorting based on multiple columns. This can be done as:

data_sorted = data.sort_values(['ApplicantIncome','CoapplicantIncome'], ascending=False)
data_sorted[['ApplicantIncome','CoapplicantIncome']].head(10)

## 9 – Plotting (Boxplot & Histogram)
#
#Many of you might be unaware that boxplots and histograms can be directly plotted in Pandas and calling matplotlib separately is not necessary. It’s just a 1-line command. For instance, if we want to compare the distribution of ApplicantIncome by Loan_Status:

import matplotlib.pyplot as plt
%matplotlib inline
data.boxplot(column="ApplicantIncome",by="Loan_Status")
data.hist(column="ApplicantIncome",by="Loan_Status",bins=30)

## 10 – Cut function for binning
#
#Sometimes numerical values make more sense if clustered together. For example, if we’re trying to model traffic (#cars on road) with time of the day (minutes). The exact minute of an hour might not be that relevant for predicting traffic as compared to actual period of the day like “Morning”, “Afternoon”, “Evening”, “Night”, “Late Night”. Modeling traffic this way will be more intuitive and will avoid overfitting.
#
#Here we define a simple function which can be re-used for binning any variable fairly easily.

#Binning:
def binning(col, cut_points, labels=None):
  #Define min and max values:
  minval = col.min()
  maxval = col.max()

  #create list by adding min and max to cut_points
  break_points = [minval] + cut_points + [maxval]

  #if no labels provided, use default labels 0 ... (n-1)
  if not labels:
    labels = range(len(cut_points)+1)

  #Binning using cut function of pandas
  colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
  return colBin

#Binning age:
cut_points = [90,140,190]
labels = ["low","medium","high","very high"]
data["LoanAmount_Bin"] = binning(data["LoanAmount"], cut_points, labels)
print pd.value_counts(data["LoanAmount_Bin"], sort=False)


## 11 – Coding nominal data
#
#Often, we find a case where we’ve to modify the categories of a nominal variable. This can be due to various reasons:
#
#Some algorithms (like Logistic Regression) require all inputs to be numeric. So nominal variables are mostly coded as 0, 1….(n-1)
#Sometimes a category might be represented in 2 ways. For e.g. temperature might be recorded as “High”, “Medium”, “Low”, “H”, “low”. Here, both “High” and “H” refer to same category. Similarly, in “Low” and “low” there is only a difference of case. But, python would read them as different levels.
#Some categories might have very low frequencies and its generally a good idea to combine them.
#Here I’ve defined a generic function which takes in input as a dictionary and codes the values using ‘replace’ function in Pandas.

#Define a generic function using Pandas replace function
def coding(col, codeDict):
  colCoded = pd.Series(col, copy=True)
  for key, value in codeDict.items():
    colCoded.replace(key, value, inplace=True)
  return colCoded
 
#Coding LoanStatus as Y=1, N=0:
print 'Before Coding:'
print pd.value_counts(data["Loan_Status"])
data["Loan_Status_Coded"] = coding(data["Loan_Status"], {'N':0,'Y':1})
print '\nAfter Coding:'
print pd.value_counts(data["Loan_Status_Coded"])
