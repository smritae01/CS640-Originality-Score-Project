import csv
import os

# csv file to be extracted 
csv_filename = './data/data.csv'
column_index = 1  # assuming data to be extracted is in the 2nd column

output_folder = './data/txtfiles'

if not os.path.exists(output_folder):
    os.makedirs(output_folder, mode=0o777)

with open(csv_filename, 'r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)

    next(reader)

    for row in reader:
        cell_value = row[column_index]

        filename = f"{column_index}_{row[0]}.txt"

        with open(os.path.join(output_folder, filename), 'w', encoding='utf-8') as outfile:
            outfile.write(cell_value)

