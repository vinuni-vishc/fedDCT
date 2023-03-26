import pyexcel as pe

# Specify the input directory containing ODS files
input_dir = '/home/quan/code/feddct/fedDCT/dataset/ham10000/data/'

# Specify the output directory for CSV files
output_dir = '/home/quan/code/feddct/fedDCT/dataset/ham10000/data/'

# Loop through all client numbers from 1 to 20
for client_num in range(1, 21):
    # Construct the input file path
    input_path = input_dir + f'client{client_num}.ods'

    # Construct the output file path
    output_path = output_dir + f'client{client_num}.csv'

    # Load the ODS file into a pyexcel sheet object
    sheet = pe.get_sheet(file_name=input_path)

    # Save the sheet object as a CSV file
    sheet.save_as(output_path, delimiter=',')

    print(f'Successfully converted {input_path} to {output_path}')