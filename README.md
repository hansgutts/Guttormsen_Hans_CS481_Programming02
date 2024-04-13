<h1>Binary Bag of Words Naive Bayes Classifier With Add One Smoothing</h1>
This is my Naive Bayes Classifier for sentiment analysis made from scratch for CS 481 Artificial Intelligence Language Understanding taught by professor Jacek Dzikowski. 

<h2>Dataset</h2>
The dataset I used is the UCI_Drug (https://www.kaggle.com/datasets/arpikr/uci-drug/data) dataset found on Kaggle. In short, it has reviews of drugs that treat varying medicial 
conditions. The dataset includes uniqueID, drugName, condition, review, rating, date, and usefulCount but the only dimensions I used for sentiment analysis were review and rating. 
Rating was converted to a negative sentiment (0) if the rating was < 7 or a positive sentiment (1) if the rating was >= 7. This cut off point was initially arbitrarially decided.
The dataset initially came in two tsv files (UCIdrug_test.csv and UCIdrug_train.csv) though these had to be combined for different training/testing splits. There are 215063 samples
after preprocessing with 142306 rated as positive and 72757 rated as negative, which does create an unbalanced dataset in favor of positive ratings.
  
<h3>Preprocessing</h3>
As discussed, the initial datasets needed to be condensed into a single csv file. The actual review entries contained html tags in the form "&----;" that needed to be transformed to 
real characters in order to have correct words. Punctuation was then removed in its entirety as we are only focused on the words in the review. I then also removed the name of the 
drug being reviewed from the actual user review as I was worried the classifier would be overtuned toward negative or positive classifications based on whether the drug itself was 
generally rated positive or negative (basically, I thought the weight on drug names would be larger than other words. This did not end up being the case, but I left the drug names 
out). Preprocessing was completed in the file named "CleanCSV.py". We did not do any additional preprocessing such as lemmatization, stemming, or removal of stop words.

<h2>Training The Classifier</h2>
We are classifying sentiment using a binary bag of words naive bayes classifer. The process of training the model takes place in "cs481_P02_A20462410.py". The function is meant to 
take one command line argument which indicates the training size and test size split. The default split is 80% training, 20% testing. All of the weights are calculated in the main
function.
The general process is:
<ol>
  <li>Convert csv file to training and testing list (need lengths and is just easier to work with)</li>
  <li>Loop through each entry and calculate P(doc = pos) and P(doc = neg) while simultaneously building vocabulary, positive vocabulary, and negative vocabulary.</li>
  <li>Go through our built vocabulary and use it to build dictionary of weights (key is the word, weight is the value). P(word | label) = (count(word | label) + smoothing)
    /(count(tokens | label) + (count(types | label) * smoothing). We do this for both the negative and positive label.</li>
  <li>We then export the dictionaries to a json file to be used in AnalyzeInputs.py</li>
</ol>
The python file will then prompt the user (actual function in "AnalyzeInputs.py") to see if they would like to manually enter sentences (the input = documents in our probability 
calculations) and outputs the sentiment classification as well as the probability of each label.

<h2>Predictions</h2>
The classifier makes its prediction in the function "Analyze" in "AnalyzeInputs.py". The classifier converts the json file we exported before and is able to classify sentiment
using P(label | document) = P(doc = label) * π P(word | label). We do this for both labels, negative and positive and whichever has the higher probability is the decided sentiment.
So, if P(negative | document) > P(positive | document) the document is classified as negative (and vice versa). It should be noted the probability calculations take place in log 
space and then are converted back to regular space (avoid underflow on longer sentences, P(word | label) is extremely small when vocab is as large as it is).

<h2>Performance</h2>
Given the lack of anything other than counts and simple math, the model performs surprisingly well. Below is the confusion matrix for our model, and it performs well in every
metric.

|                    | Actual Positive  | Actual Negative |                    |
|--------------------|------------------|-----------------|--------------------|
| Predicted Positive | TP = 23294       | FP = 5135       |Sensitivity = .8194 |
| Predicted Negative | FN = 4065        | TN = 10519      |Specificity = .7213 |
|                    | Precision = .8514| NPV = .672      |Accuracy = .7861    |

The F-Score comes out to .8351.

<h2>Analysis</h2>
There are a few issues with our model. Not much testing was done to determine if additional preprocessing would further improve the model. Stemming, lemmatization, and the 
removal of stop words are the first ideas to come to mind. Additionally, this approach is very limited. We attribute a large amount of this models success to the straightforward
nature of medical treatment reviews. In general, reviews aren't trying to be creative or funny as they would for, say, movie reviews. Sarcasm simply would not be able to be 
identified by a model of this simplicity. Finally, nothing was done about the label imbalance. As it stands, the model has a clear bias towards positive reviews and struggles at 
identifying negative reviews more than positive reviews, though this problem is not catastrophic. Despite the problems listed above, we still believe this is a strong model.
Improvements could be made, but the best improvement would be using a more sophisticated approach rather than trying to improve the naive bayes approach.
