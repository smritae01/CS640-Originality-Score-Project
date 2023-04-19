import csv
import os
from gptzero import GPTZeroAPI

api_key = 'a14d481f54b44db4a9f74c5b3e538aea' # replace API key
gptzero_api = GPTZeroAPI(api_key)

input_csv_path = './data/gpt0-rem.csv'
output_csv_path = './data/results1.csv' # keep replacing this if you want to preserve all data
txt_files_dir = './data/txtfiles'

if not os.path.exists(txt_files_dir):
    os.makedirs(txt_files_dir, mode=0o777)

with open(input_csv_path, 'r', encoding='ISO-8859-1') as csvfile:
    reader = csv.reader(csvfile)

    next(reader)

    results = []
    for row in reader:
        row_id = row[0]
        row_content = row[1]

        filename = f"{row_id}.txt"
        file_path = os.path.join(txt_files_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(row_content)

        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            response = gptzero_api.text_predict(content)

            if 'error' in response:
                print('error')
                continue

            sentences = response['documents'][0]['sentences']
            perplexities = [sentence['perplexity'] for sentence in sentences]
            avg_perplexity = sum(perplexities) / len(perplexities)
            
            avg_gen_prob = response['documents'][0]['average_generated_prob']
            comp_gen_prob = response['documents'][0]['completely_generated_prob']
            overall_burstiness = response['documents'][0]['overall_burstiness']
            
            gptzero_prediction = 1 if comp_gen_prob > 0.65 else 0

            results.append([row_id, row_content, avg_gen_prob, comp_gen_prob, overall_burstiness, avg_perplexity, gptzero_prediction])

with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'truncated', 'Average Generated Probability', 'Completely Generated Probability', 'Overall Burstiness', 'Average Perplexity', 'GPTZero Prediction'])
    writer.writerows(results)
