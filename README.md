# CS640-Originality-Score-Project

Drive link with the dataset - https://drive.google.com/drive/folders/18JWM2I-ZCadyzHhHnfoWNixrpiVxgMnR?usp=sharing

Paper Link - https://drive.google.com/file/d/15FBw3p3SWxiH4_2_5YQvY_m26mNnFWXE/view?usp=share_link 

## DetectGPT Implementation

Code sourced from official DetectGPT repo as well as an open-source prompt implementation from https://github.com/BurhanUlTayyab/DetectGPT

### Features obtained from the code 

* Mean z-score : mean of the measure of how many standard deviations below or above the population mean a raw score is for the z-scores in all sentences of an entry. 

If the z-score obtained is between the range of 0.25 - 0.7 then the text is classified as Human-written. 
If the z-score obtained is higher than 0.7 or lower than 0.25, then the text is classified as AI-generated.

* z-scores are obtained by the formula (real_log_likelihood - mean_generated_log_likelihood)/std_generated_log_likelihood


## GPTZero Implementation

### Interpreting the Results

For GPTZero, we have used their API to get predictions. The possible results from GPTZero through their interface are:

1. Most likely written by a Human

2. May include parts written by an AI

3. Most likely generated by an AI

However, they have not made their criteria for evaluating a document as "May include parts written by an AI". 

This is the schema of the response we get from their API.

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

GPTZero recommends using completely\_generated\_prob to understand whether a document was completely generated by AI. 

"At a threshold of 0.65, 85\% of AI documents are classified as AI, and 99\% of human documents are classified as human At a threshold of 0.16, 96\% of AI documents are classified as AI, and 96\% of human documents are classified as human We recommend using a threshold of 0.65 or higher to minimize the number of false positives, as we think it is currently more harmful to falsely detect human writing as AI than vice versa."

We have followed the threshold of 0.65 for the classification for GPTZero. On our validation dataset, we set all documents with completely\_generated\_prob under the threshold as human, and above as AI.

## Files and their purpose

In the gptzero-api directory, run the main.py file. This will first convert the csv cells into txt files in the "./data/txtfiles" directory and then generates the csv file to get the relevant information from GPTZero.

Note: Only 250 requests are possible per hour for the GPTZero API
