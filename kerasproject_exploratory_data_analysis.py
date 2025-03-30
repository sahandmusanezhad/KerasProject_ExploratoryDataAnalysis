import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_info = pd.read_csv('D:\codes\keras_project\TensorFlow_FILES\DATA\lending_club_info.csv', index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])

def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])

df = pd.read_csv('D:\codes\keras_project\TensorFlow_FILES\DATA\lending_club_loan_two.csv')
df.info()

sns.countplot(x='loan_status', data=df)
plt.show()

plt.figure(figsize=(12, 4))
sns.distplot(df['loan_amnt'], kde=False, bins=40)
plt.show()

# plt.figure(figsize=(12, 7))
# sns.heatmap(df.corr(), annot=True, cmap='viridis')
# plt.ylim(10,0)
# plt.show()

feat_info('installment')

feat_info('loan_amnt')

sns.scatterplot(x='installment', y='loan_amnt', data=df)
plt.show()

sns.boxplot(x='loan_status', y='loan_amnt', data=df)
plt.show()

df.groupby('loan_status')['loan_amnt'].describe()

print('grade: ',df['grade'].unique())

print('sub_grade: ',df['sub_grade'].unique())

feat_info('sub_grade')
sns.countplot(x='grade', data=df, hue='loan_status')
plt.show()

plt.figure(figsize=(12, 4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade', data=df, order=subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()


f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F')]
plt.figure(figsize=(12, 4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade', data=f_and_g, order=subgrade_order, palette='coolwarm', hue='loan_status')
plt.show()

df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

numeric_df = df.select_dtypes(include=['number'])
numeric_df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
plt.show()