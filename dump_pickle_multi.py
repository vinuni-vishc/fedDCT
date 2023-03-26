import pandas as pd

# Specify the input directory containing CSV files
input_dir = '/home/quan/code/feddct/fedDCT/dataset/'

# Specify the output directory for pickle files
output_dir = '/home/quan/code/feddct/fedDCT/dataset/'

# Loop through all client numbers from 1 to 20
for client_num in range(3, 4):
    # Construct the input file path
    input_path = input_dir + f'client{client_num}.csv'

    # Construct the output file path
    output_path = output_dir + f'client{client_num}.pickle'

    # Load the CSV file into a DataFrame
    df = pd.read_csv(input_path)
    df = df[df.filter(regex='^(?!Unnamed)').columns]

    # Save the DataFrame as a pickle file
    with open(output_path, 'wb') as f:
        pd.to_pickle(df, f)

    print(f'Successfully converted {input_path} to {output_path}')