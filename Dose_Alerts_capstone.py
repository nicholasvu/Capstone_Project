import re
import pandas as pd

def dose_alerts_for_a_drug(drug_name):
  df = pd.read_csv('Dose_Alerts_Edited.csv')
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
  grouped = df1.groupby(['daily_dose_exceeds', 'Warning_Status']).count()
  print drug_name
  print grouped

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
#dose_alerts_for_a_drug is a function/variable. drug_name is an object
#dose_alerts_for_a_drug('VANCOMYCIN')
#dose_alerts_for_a_drug('KETOROLAC')
#dose_alerts_for_a_drug_single('VANCOMYCIN')
#following means you don't have to copy and past lines like 26-31
for drug in ['FAMOTIDINE', 'LORAZEPAM', 'IBUPROFEN']: dose_alerts_for_a_drug(drug)
for drug in ['FAMOTIDINE', 'LORAZEPAM', 'IBUPROFEN']: dose_alerts_for_a_drug_single(drug)
for drug in ['INSULIN LISPRO']: dose_alerts_for_a_drug_below(drug)


  #you can shorten it even more here!, for/foreach/each (these are functions) works in
#for drug in list_of_drugs: dose_alerts_for_a_drug(drug_name)

#you can never have too many functions! per Denis!
#df2 = df({'x' : ['daily_dose_exceeds', 'single_dose_exceeds'], 'y' : [FAMOTIDINE, LORAZEPAM, IBUPROFEN})
