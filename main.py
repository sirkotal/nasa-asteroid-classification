import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Load the data
data = pd.read_csv('nasa.csv')
data.fillna("NA", inplace=True)
print(data.head())
stats = data.describe()
print(stats)

print(sb.pairplot(data.dropna(), hue='Hazardous'))
