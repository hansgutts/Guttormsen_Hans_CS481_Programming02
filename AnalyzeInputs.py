import json
import math
import string

#paths to json files that have weights
posweightfilepath = "PosWeightsUCIDrug.json"
negweightfilepath = "NegWeightsUCIDrug.json"

#analyze function
def analyze(input) :

    input = input.replace('/', ' ').replace('.', ' ').replace('-', ' ')
    input = input.translate(str.maketrans('', '', string.punctuation)).lower()

    #dicts to hold weights
    posweightsdict = {}
    negweightsdict = {}

    #open the json files and import them to dictionaries
    with open(posweightfilepath) as posweights :
        posweightsdict = json.load(posweights)

    with open(negweightfilepath) as negweights :
        negweightsdict = json.load(negweights)

    #initialize the probabilities to the probability any document is pos or neg
    #use log space to avoid underflow
    probDocPositive = math.log(posweightsdict["ProbPositive"])
    probDocNegative = math.log(negweightsdict["ProbNegative"])
    
    #for each word in document
    for word in input.split() :

        #make sure the word actually has a probability
        if word in posweightsdict :

            #multiply (in this case add because log space) the probability
            probDocPositive += math.log(posweightsdict[word])

        #make sure the word actually has a probability
        if word in negweightsdict :

            #multiply (in this case add because log space) the probability
            probDocNegative += math.log(negweightsdict[word])

    #come back from log space e^ln(a+b) = a*b
    probDocPositive = math.exp(probDocPositive)
    probDocNegative = math.exp(probDocNegative)

    if probDocPositive > probDocNegative :
        label = 1
    else :
        label = 0

    return [label, probDocPositive, probDocNegative]

def prompt() :
    #make it so we enter the while loop
    response = "Y"

    #ask for user sentences until they say no
    while response.upper() == "Y" :

        #prompt for sentence
        sentence = input("Enter your sentence: ")

        #analyze the inputs
        output = analyze(sentence)

        #output of analyzeinputs = [label, probpos, probneg]
        if output[0] == 1 :
            label = "positive"
        else :
            label = "negative"

        #tell the user the classification and probability
        print(f"\nSentence S: \"{sentence}\" was classified as {label}.\nP(positive | S) = {output[1]} \nP(negative | S) = {output[2]}\n")

        #prompt to continue
        response = input("Do you want to enter another sentence [Y/N]?")