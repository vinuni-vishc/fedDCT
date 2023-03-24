import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('train.csv')
df = df[df.filter(regex='^(?!Unnamed)').columns]
# Save the DataFrame as a pickle file
with open('file.pickle', 'wb') as f:
    pd.to_pickle(df, f)