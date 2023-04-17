from gptzero import GPTZeroAPI
import os
import csv

api_key = '6341e691c2b8492baef3035fec80d31a' # gitika's API key
gptzero_api = GPTZeroAPI(api_key)

directory = './data/txtfiles/'
os.chmod(directory, 0o755)

results = []

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            response = gptzero_api.text_predict(content)

            if 'error' in response:
                continue

            sentences = response['documents'][0]['sentences']
            perplexities = [sentence['perplexity'] for sentence in sentences]
            avg_perplexity = sum(perplexities) / len(perplexities)
            
            avg_gen_prob = response['documents'][0]['average_generated_prob']
            comp_gen_prob = response['documents'][0]['completely_generated_prob']
            overall_burstiness = response['documents'][0]['overall_burstiness']
            # paragraphs = response['documents'][0]['paragraphs']
            # sentences = response['documents'][0]['sentences']

            gptzero_prediction = "AI" if comp_gen_prob > 0.65 else "Human"
            
            results.append([content, avg_gen_prob, comp_gen_prob, overall_burstiness, avg_perplexity, gptzero_prediction])

with open('./data/results.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Content', 'Average Generated Probability', 'Completely Generated Probability', 'Overall Burstiness', 'Average Perplexity', 'GPTZero Prediction'])
    writer.writerows(results)
