#you can use these simple lines of code to extract only records for Ketorolac. Then you can do other operations on that.
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

drugs = ['KETOROLAC']

#reg ex expression to grab exceeds max daily dose from Percent_Deviation_from_Dose
def extract_percent_of_daily_dose(daily_dose_string):
    match = re.search(r'Exceeds maximum daily dose limit \((\d+)%\)', daily_dose_string)
#r'^[Overridden|Cancelled|Viewed]'
#match.group(0) would return the whole phrase, match.group(1) returns the first number
    return match and int(match.group(1))

#reg ex expression to grab single max dose from Percent_Deviation_from_Dose column
def extract_percent_of_single_dose(single_dose_string):
    correct = re.search(r'Exceeds maximum single dose limit \((\d+)%\)', single_dose_string)
    return correct and int(correct.group(1))

#to define variable to pull ketorolac only rows for extract_percent_of_daily_dose
def generate_for_drug(drug):
    df = pd.read_csv('Dose_Alerts_Edited.csv')
    df['filter'] = [drug in x for x in df['Description'].tolist()]
    #now you pick the subset of data where you have 'Ketorolac' in the description field
    df = df[df['filter']]
#split the Percent_Deviation_from_Dose column into two columns so you have single and daily max dose limit separately

    #following code applies to all cells in column 'Percent_Deviation_from_Dose'
    #.map method transforms one list of data into another list of data
    df['daily_dose_limit'] = df['Percent_Deviation_from_Dose'].map(extract_percent_of_daily_dose)
    #drops all NaN rows
    df['daily_dose_limit'].dropna(inplace=True)
    return df
#.map method to transform Percent_Deviation_from_Dose into single_dose_limit
#to define variable to pull ketorolac only rows for single daily dose
def generate_for_single(Ketorolac):
    df = pd.read_csv('Dose_Alerts_Edited.csv')
    df['filter'] = [Ketorolac in x for x in df['Description'].tolist()]
    #now you pick the subset of data where you have 'Ketorolac' in the description field
    df = df[df['filter']]
    df['single_dose_limit'] = df['Percent_Deviation_from_Dose'].map(extract_percent_of_single_dose)
    df['single_dose_limit'].dropna(inplace=True)
    return df

#to print daily dose limit stats for ketorolac only
for drug in drugs:
    df = generate_for_drug(drug)
print 'Ketorolac Daily Dose Count'
print df['daily_dose_limit'].value_counts()
print df.groupby('Warning_Status').count()

#to print single dose limit stats for ketorolac only
for Ketorolac in drugs:
    df = generate_for_single(Ketorolac)
print 'Ketorolac Single Dose Count'
print df['single_dose_limit'].value_counts()
#of each value, how many of them were overrides/views/cancels
print df.groupby('Warning_Status').count()
#df.groupby('Warning_Status').my_new_column.hist(alpha=0.4)
