# CS640-Originality-Score-Project

Drive link with the dataset - https://drive.google.com/drive/folders/18JWM2I-ZCadyzHhHnfoWNixrpiVxgMnR?usp=sharing

Paper Link - 

## GPTZero Implementation

### Interpretting the Results

The schema was DocumentPrediction is:

```
DocumentPredictions{ 
    documents	
    [{ 
        average_generated_prob - number - The average of the probabilties that each sentence was generated by an AI
        completely_generated_prob - number - The probability that the entire document was generated by an AI
        overall_burstiness - number - The amount of variation in the perplexity of the document. A useful indicator to distinguish AI and human written text
        sentences	[
            Information about each sentence is contained in this array, and the sentences in the document are listed in order.
            {
                sentence - string
                perplexity - number - The lower the perplexity, the more likely an AI would have generated this sentence
                generated_prob - number - The probability that this sentence was generated by an AI. Our current model predicts 0/1 labels, but this may change to be a percentage in the future.
            }]

        paragraphs	[
            Paragraphs are newline-delimited bodies of text in the document

            {
            start_sentence_index - number - The index in the sentences array of the first sentence of the paragraph

            num_sentences - number - The number of sentences in this paragraph.

            completely_generated_prob - number - The probability that the entire paragraph was generated by an AI

            }]
    }]
}

```

threshold - 0.65 - completed_geenerated_prob 
token limit 200 - truncated function

GPTZero recommends: 
"using completely_generated_prob to understand whether a document was completely generated by AI. On our validation dataset, here is how the results change when you set all documents with completely_generated_prob under the threshold as human, and above as AI.

At a threshold of 0.65, 85% of AI documents are classified as AI, and 99% of human documents are classified as human
At a threshold of 0.16, 96% of AI documents are classified as AI, and 96% of human documents are classified as human
We recommend using a threshold of 0.65 or higher to minimize the number of false positives, as we think it is currently more harmful to falsely detect human writing as AI than vice versa."

We have followed the threshold of 0.65 for the classification for GPTZero.


## Files and their purpose.

In the gptzero-api, we first convert the csv cells into txt files in the "./data/txtfiles" directory. We do this by running the "./gptzero-api/compute_files.py" file.
We then run "./gptzero-api/generate_csv" to generate the csv file to get the relevant information from GPTZero.

