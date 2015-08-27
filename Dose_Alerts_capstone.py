'''
author: @nicholasvu
08/26/2015 22:50

Random Forest Classifier claims numbers of labels=1 does not match number of samples=6037. Number of samples in x and y match according to my records.
Need to troubleshoot what issue is. Otherwise, program should run.
'''
import re
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from string import letters
import numpy as np
import seaborn as sns
import scipy as sp
import logloss
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
import pdb
import pickle
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

#define df
df = pd.read_csv('Dose_Alerts_Edited.csv')

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Exceeds maximum daily dose limit' only
def dose_alerts_for_a_drug(drug_name):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

  # create a separate column for exceeds daily dose limit called 'daily_dose_exceeds'
  list1 = []
  for j in range(len(df1)):
      if ('Exceeds maximum daily dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Exceeds maximum daily dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['daily_dose_exceeds'] = list1
 #add the other two columns
  grouped = df1.groupby(['daily_dose_exceeds', 'warning_status']).count()
  #print drug_name
  #print grouped
  return df1

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Exceeds maximum single dose limit' only
def dose_alerts_for_a_drug_single(drug_name_single):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_single in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

  # create a separate column called for single dose limit exceeded called 'single_dose_exceeds'
  list1 = []
  for j in range(len(df1)):
      if ('Exceeds maximum single dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Exceeds maximum single dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['single_dose_exceeds'] = list1
  grouped = df1.groupby(['single_dose_exceeds', 'warning_status']).count()
  #print drug_name_single
  #print grouped
  return df1

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Below minimum dose limit' only
def dose_alerts_for_a_drug_below(drug_name_below):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_below in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

   # create a separate column for below minimum dose limit called 'below_dose_minimum'
  list1 = []
  for j in range(len(df1)):
      if ('Below minimum dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list1.append(int(re.findall(r'Below minimum dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list1.append(0)
  df1['below_dose_minimum'] = list1
  #print list1[0]
  grouped = df1.groupby(['below_dose_minimum', 'warning_status']).count()
  #print drug_name_below
  #print grouped
  return df1


#CREATE FEATURES AND ASSIGN SCORES TO VALUES IN EACH FEATURE (lines 89-116)

#assign a strength of recommendation to the Provider_Type (15% of total weight)
Provider_Strength_Score = [8 if Provider_Type=='Pharmacist' else 7 if Provider_Type=='Pharmacy Resident'
else 6 if Provider_Type=='Nurse Practitioner' else 5 if Provider_Type=='Physician Assistant' else 4 if Provider_Type=='Attending Physician' else 3 if Provider_Type=='Resident' else 2 if Provider_Type=='Fellow' else 1 for Provider_Type in df['Provider_Type'].tolist()]

#assign strength of recommendation to Patient_Hospital/Clinic (5% of total weight)
Hospital_Strength_Score = [2 if Patient_Hospital_Clinic =='LA JOLLA HOSPITAL HOD/OP' else 2 for Patient_Hospital_Clinic in df['Patient_Hospital_Clinic'].tolist()]

#assign strength of recommendation to Patient_Department (2% of total weight)
#Department_Strength_Score = [1 if Department_Score=='LJ EMERGENCY DEPT' else 1 for Department_Score in df['Department_Score'].tolist()]

#assign strength to interaction setting (5% of total weight)
Setting_Strength_Score = [7 if Interaction_Setting=='MODEL PHARMACIST INTERACTIONS [1010]' else 7 if Interaction_Setting=='UC PHARMACIST INTERACTIONS - OB [1080]' else 3 for Interaction_Setting in df['Interaction_Setting'].tolist()]

#assign strength to Source (8% of total weight)
Source_Strength_Score = [6 if Source=='Verify Orders' else 4 if Source=='Enter Orders' else 0 for Source in df['Source'].tolist()]

#assign strength to Context (2% of total weight)
Context_Strength_Score = [7 if Context=='Inpatient' else 3 if Context=='Outpatient' else 0 for Context in df['Context'].tolist()]

#assign strength to Same_OrderSet_Panels (2% of total weight)
#Y and N are irrelevant to whether you override/cancel/view the dose alert, so scores are equal for all strings
Same_OrderSet_Panels_Strength_Score = [5 if Same_OrderSet_Panels=='N' else 5 if Same_OrderSet_Panels=='Y' else 5 for Same_OrderSet_Panels in df['Same_OrderSet_Panels'].tolist()]

#assign strength to Warning_Form_Shown_To_User (1% of total weight)
#if you see the Warning (Y), that's really important because the datapoint is useless if the user never saw the warning form, hence the score of 10 for Y
#if the user didn't see the warning form, then there's no point in laying any weight into that clinician's action because it was misinformed
Warning_Seen_Strength_Score = [10 if Warning_Form_Shown_To_User=='Y' else 0 for Warning_Form_Shown_To_User in df['Warning_Form_Shown_To_User'].tolist()]

#assign score to target vector 'warning_status'
warning_strength_score = [3 if warning_status=='Overridden' else 2 if warning_status=='Cancelled' else 1 if warning_status=='viewed' else 0 for warning_status in df['warning_status'].tolist()]

#assign list of medicines to analyze
medicine_names = ['KETOROLAC', 'SODIUM CHLORIDE', 'LOPERAMIDE', 'VANCOMYCIN', 'DOCUSATE SODIUM', 'IPRATROPIUM', 'LORAZEPAM', 'LIDOCAINE-EPINEPHRINE', 'SODIUM BICARBONATE', 'DEXAMETHASONE',
'IPRATROPIUM-ALBUTEROL', 'CALCIUM CHLORIDE', 'IBUPROFEN', 'CEFAZOLIN', 'PENTAMIDINE', 'ACETAZOLAMIDE', 'ENOXAPARIN', 'LACOSAMIDE', 'METHYLPREDNISOLONE', 'INFLIXIMAB', 'INSULIN REGULAR',
'METOCLOPRAMIDE', 'PROGESTERONE', 'FLUPHENAZINE', 'GABAPENTIN', 'TORSEMIDE', 'CEFTAZIDIME', 'ESCITALOPRAM', 'CITALOPRAM', 'IMMUNE GLOBULIN', 'LEVETIRACETAM', 'FLUORESCEIN', 'OXYCODONE',
'LIDOCAINE-EPINEPHRINE', 'INSULIN LISPRO', 'GENTAMICIN', 'TRASTUZUMAB', 'CARBOPLATIN', 'CARFILZOMIB', 'BUMETANIDE', 'FLUOROURACIL', 'OCTREOTIDE', 'ROCURONIUM', 'IFOSFAMIDE', 'MESNA',
'BEVACIZUMAB', 'SULFAMETHOXAZOLE-TRIMETHOPRIM', 'DILTIAZEM', 'CITRIC ACID-SODIUM CITRATE', 'COLISTIMETHATE', 'DIVALPROEX', 'PHOSPHORUS', 'CETIRIZINE', 'MIDAZOLAM',
'DOXORUBICIN', 'FENTANYL', 'TOPIRAMATE', 'BORTEZOMIB', 'LACTULOSE', 'PALONOSETRON', 'AMPICILLIN', 'DIAZEPAM', 'ONDANSETRON', 'MEROPENEM', 'FLUDARABINE', 'MAGNESIUM HYDROXIDE', 'PHENYLEPHRINE',
'PLERIXAFOR']

for i in medicine_names:

    df = dose_alerts_for_a_drug_single(i)
    df2 = dose_alerts_for_a_drug_below(i)
    df3 = dose_alerts_for_a_drug(i)

#assign x to a pandas dataframe
x = pd.DataFrame()
#assign target vector to 'warning_status' column in an array
y = np.array([warning_strength_score])

#assign features to variable x
x['Provider_Strength_Score'] = Provider_Strength_Score
x['Hospital_Strength_Score'] = Hospital_Strength_Score
#x['Department_Strength_Score'] = Department_Strength_Score
x['Setting_Strength_Score'] = Setting_Strength_Score
x['Source_Strength_Score'] = Source_Strength_Score
x['Context_Strength_Score'] = Context_Strength_Score
x['Same_OrderSet_Panels_Strength_Score'] = Same_OrderSet_Panels_Strength_Score
x['Warning_Seen_Strength_Score'] = Warning_Seen_Strength_Score

#test and train data (80% is train and 20% test)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

#assign a classifier to Random Forest Classifier
clf = RandomForestClassifier(n_estimators=50, n_jobs=2)

#x should be (n_samples, n_features) in a matrix
x = x.as_matrix()

#fit the classifier
clf.fit(x, y)

#PLOT CONFUSION MATRIX PLOT
# 1. split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
#2. run classifier
classifier = svm.SVC(kernel='linear')
y_pred = classifier.fit(x_train, y_train).predict(x_test)
print "test (prediction): ", y_pred

# 3. Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
# 4. show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted label')
plt.show()

#show classifier
print clf
