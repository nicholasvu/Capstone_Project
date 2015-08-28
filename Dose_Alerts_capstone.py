'''
author: @nicholasvu
aided by: @wwolf, @roy, @dtherrick
08/26/2015 22:50

This program was designed to answer, "Should a dose alert be overridden, cancelled,
or viewed by a clinician?"

This program simply organizes the Dose_Alerts_Edits.csv data for a list of
drugs according to how much of the daily dose, single dose, and below minimum
dose limits were exceeded. Then it assigns a score to the values under a specific
list of features. Then the Random Forest Classifier will improve predictive accuracy
of whether or not a dose alert will be overridden, cancelled, or viewed.

'''
import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from string import letters
import numpy as np
import seaborn as sns
import scipy as sp
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.datasets import make_multilabel_classification
import pdb
import pickle
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold as KFold
from sklearn.metrics import confusion_matrix
import sklearn.metrics as clf
import pylab as pl
import sqlite3 as lite

#define df
df = pd.read_csv('Dose_Alerts_Edited.csv')

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Exceeds maximum daily dose limit' only
def dose_alerts_for_a_drug(drug_name):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

#create a separate column for exceeds daily dose limit called 'daily_dose_exceeds'
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
  #print list1
  return df1

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Exceeds maximum single dose limit' only
def dose_alerts_for_a_drug_single(drug_name_single):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_single in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

  # create a separate column called for single dose limit exceeded called 'single_dose_exceeds'
  list2 = []
  for j in range(len(df1)):
      if ('Exceeds maximum single dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list2.append(int(re.findall(r'Exceeds maximum single dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list2.append(0)
  df1['single_dose_exceeds'] = list2
  grouped = df1.groupby(['single_dose_exceeds', 'warning_status']).count()
  #print grouped
  #print drug_name_single
  #print list2
  return df1

#create function to pull integer from Percent_Deviation_from_Dose column related to 'Below minimum dose limit' only
def dose_alerts_for_a_drug_below(drug_name_below):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
  lisT = [1 if drug_name_below in x else 0 for x in df['Description'].tolist()]
  df['filter'] = lisT
  df1 = df[df['filter'] == 1]
  df1 = df1.reset_index(drop=True)

   # create a separate column for below minimum dose limit called 'below_dose_minimum'
  list3 = []
  for j in range(len(df1)):
      if ('Below minimum dose limit' in df1['Percent_Deviation_from_Dose'][j]):
          list3.append(int(re.findall(r'Below minimum dose limit \((\d+)%\)', df1['Percent_Deviation_from_Dose'][j])[0]))
      else:
          list3.append(((0), df1['Percent_Deviation_from_Dose'][j][0]))
  df1['below_dose_minimum'] = list3
  grouped = df1.groupby(['below_dose_minimum', 'warning_status']).count()
  #print drug_name_below
  #print list3
  return df1

#CREATE FEATURES AND ASSIGN SCORES TO VALUES IN EACH FEATURE (lines 89-134)

#assign a strength of recommendation to the Provider_Type (15% of total weight)
Provider_Strength_Score = [8 if Provider_Type=='Pharmacist' else 7 if Provider_Type=='Pharmacy Resident'
else 6 if Provider_Type=='Nurse Practitioner' else 5 if Provider_Type=='Physician Assistant' else 4 if Provider_Type=='Attending Physician' else 3 if Provider_Type=='Resident' else 2 if Provider_Type=='Fellow' else 1 for Provider_Type in df['Provider_Type'].tolist()]

#assign strength of recommendation to Patient_Hospital/Clinic (5% of total weight)
Hospital_Strength_Score = [2 if Patient_Hospital_Clinic =='LA JOLLA HOSPITAL HOD/OP' else 2 for Patient_Hospital_Clinic in df['Patient_Hospital_Clinic'].tolist()]

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
warning_strength_score = [3 if warning_status=='Overridden' else 2 if warning_status=='Canceled' else 1 if warning_status=='Viewed' else 0 for warning_status in df['warning_status'].tolist()]

#assign list of medicines to analyze medicine_names = ['KETOROLAC', 'SODIUM CHLORIDE', 'LOPERAMIDE', 'VANCOMYCIN', 'DOCUSATE SODIUM', 'IPRATROPIUM', 'LORAZEPAM', 'LIDOCAINE-EPINEPHRINE', 'SODIUM BICARBONATE', 'DEXAMETHASONE',
medicine_names = ['IPRATROPIUM-ALBUTEROL', 'CALCIUM CHLORIDE', 'IBUPROFEN', 'CEFAZOLIN', 'PENTAMIDINE', 'ACETAZOLAMIDE', 'ENOXAPARIN', 'LACOSAMIDE', 'METHYLPREDNISOLONE', 'INFLIXIMAB', 'INSULIN REGULAR',
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
y = np.ravel([warning_strength_score])
y = y.transpose()


#assign features to variable x
x['Provider_Strength_Score'] = Provider_Strength_Score
x['Hospital_Strength_Score'] = Hospital_Strength_Score
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
clf = RandomForestClassifier(n_estimators=100, n_jobs=4)

#x should be (n_samples, n_features) in a matrix
x = x.as_matrix()

#fit the classifier using a multi-label indicator
clf.fit(x, y)

# 1. split the data into a training set and a test set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
#2. run classifier
classifier = svm.SVC(kernel='linear')

#print respective feature_importance coefficients showing how important each feature is in predicitng target vector
print(clf.feature_importances_)
print "mean accuracy score for the clf model on test data: ", clf.score(x_test, y_test)
#show prediction results
print "test (prediction): ", clf.predict(x_train)
test_pred = clf.predict(x_test)

#compute mean error using multiclass log loss function (this is another means of showing how accurate prediction is)
def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss
    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]
    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

if __name__ == "__main__":
    print "Start training"
    # Get the probability predictions for computing the log-loss function
    kf = KFold(y, n_folds=5)
    # prediction probabilities number of samples, by number of classes
    y_pred = np.zeros((len(y), len(set(y))))
    for train, test in kf:
        x_train, x_test, y_train, y_test = x[train, :], x[test, :], y[train], y[test]
        clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=1)
        clf.fit(x_train, y_train)
        y_pred[test] = clf.predict_proba(x_test)
    print "With CV", multiclass_log_loss(y, y_pred)
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=1)
    clf.fit(x, y)
    y_pred = clf.predict_proba(x)
