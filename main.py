import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('nasa.csv')

if data.isna().any().any():
    data.fillna("NA", inplace=True)

print(data.head())

nasa_class = data.pop('Hazardous')

data.insert(0, 'Hazardous', nasa_class)
stats = data.describe()
print(stats)

# print(sb.pairplot(data.dropna(), hue='Hazardous'))

label_encoder = LabelEncoder()

new_data = data.drop('Hazardous', axis=1)
hazard = data['Hazardous']

(training_inputs,
testing_inputs,
training_classes,
testing_classes) = train_test_split(new_data, hazard, test_size=0.25, random_state=1)
