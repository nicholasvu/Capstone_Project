import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from string import letters
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def dose_alerts_for_a_drug(drug_name):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  #Recommended refactor:
#Chris Angelico says Current procedure is to load the CSV, filter by drug, then adds a column. Which column depends on which of three similar functions is called.
#Alternative would be to load the CSV, then add three columns to it. Afterward, you can filter by drug and/or calculate other info eg the random forest classifier.
  lisT = [1 if drug_name in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

  # create a separate column for exceeds dose limit
  list1 = []
  for j in range(len(df1)):
      if ('Exceeds maximum daily dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Exceeds maximum daily dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['daily_dose_exceeds'] = list1
 #add the other two columns
  grouped = df1.groupby(['daily_dose_exceeds', 'Warning_Status']).count()
  print drug_name
  print grouped
  return df1

def dose_alerts_for_a_drug_single(drug_name_single):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_single in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

  # create a separate column for exceeds dose limit
  list1 = []
  for j in range(len(df1)):
      if ('Exceeds maximum single dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Exceeds maximum single dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['single_dose_exceeds'] = list1
  grouped = df1.groupby(['single_dose_exceeds', 'Warning_Status']).count()
  print drug_name_single
  print grouped
  return df1
def dose_alerts_for_a_drug_below(drug_name_below):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_below in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

   # create a separate column for exceeds dose limit
  list1 = []
  for j in range(len(df1)):
      if ('Below minimum dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Below minimum dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['below_dose_minimum'] = list1
  grouped = df1.groupby(['below_dose_minimum', 'Warning_Status']).count()
  print drug_name_below
  print grouped
  return df1
#dose_alerts_for_a_drug is a function/variable. drug_name is an object
#dose_alerts_for_a_drug('VANCOMYCIN')
#dose_alerts_for_a_drug('KETOROLAC')
#dose_alerts_for_a_drug_single('VANCOMYCIN')
#following means you don't have to copy and past lines like 26-31
#for drug in ['FAMOTIDINE', 'LORAZEPAM', 'IBUPROFEN']: dose_alerts_for_a_drug(drug)
#for drug in ['FAMOTIDINE', 'LORAZEPAM', 'IBUPROFEN']: dose_alerts_for_a_drug_single(drug)
#FOR BELOW, ONLY CERTAIN DRUGS HAVE IT, SO IF YOU SPECIFY DRUGS THAT DON'T HAVE BELOW MINIMUM DOSE LIMIT, AN ERROR WILL COME UP
#for drug in ['FAMOTIDINE', 'LORAZEPAM', 'IBUPROFEN', 'INSULIN LISPRO', 'VANCOMYCIN']: dose_alerts_for_a_drug_below(drug)

#define y to be target column
#df = pd.read_csv('Dose_Alerts_Edited.csv')
df = dose_alerts_for_a_drug_single('KETOROLAC')
df = dose_alerts_for_a_drug_single('VANCOMYCIN')
df2 = dose_alerts_for_a_drug_below('KETOROLAC')
df2 = dose_alerts_for_a_drug_below('VANCOMYCIN')
df3 = dose_alerts_for_a_drug('KETOROLAC')
df3 = dose_alerts_for_a_drug('VANCOMYCIN')
df = dose_alerts_for_a_drug_single('FAMOTIDINE')
df2 = dose_alerts_for_a_drug_below('FAMOTIDINE')
df3 = dose_alerts_for_a_drug('FAMOTIDINE')
y = df['Warning_Status']
#x = df.drop(['Warning_Status', 'Date_Time', 'Patient', 'Alert_ID', 'Alert_DAT', 'Type', 'Source', 'Medications_Involved_in_warning_checking', 'Severity', 'Providery_Type', 'Override_Reason', 'Override_Comment', 'Dose_Checking_Percent_Allowance_for_Min_Dose', 'Order_Sets', 'Panels', 'Same_OrderSet_Panels'], axis=1)
#redefine x to a new dataframe
x = pd.DataFrame()
#patch in as many columns as we want
x['single_dose_exceeds'] = df['single_dose_exceeds']
x['below_dose_minimum'] = df2['below_dose_minimum']
x['daily_dose_exceeds'] = df3['daily_dose_exceeds']
#can repeat method for other columns x['single_dose_exceeds'] = df['single_dose_exceeds']
clf = RandomForestClassifier(n_estimators=50)
print '=========='
print x['single_dose_exceeds']
print x['below_dose_minimum']
print x['daily_dose_exceeds']
plt.hist(x['single_dose_exceeds'], histtype='bar')
plt.show()
clf.fit(x, y)
print clf

  #you can shorten it even more here!, for/foreach/each (these are functions) works in
#for drug in list_of_drugs: dose_alerts_for_a_drug(drug_name)

#you can never have too many functions! per Denis!
#df2 = df({'x' : ['daily_dose_exceeds', 'single_dose_exceeds'], 'y' : [FAMOTIDINE, LORAZEPAM, IBUPROFEN})
