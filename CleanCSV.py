import csv
import pandas
import html
import string

csv1path = "UCIdrug_test.csv"
csv2path = "UCIdrug_train.csv"
newcsvpath = "UCIdrugClean.csv"

#a list of our new csv file
newcsv = []

#open both csv files
with open(csv1path, errors='ignore') as csv1 :
    with open(csv2path, errors='ignore') as csv2 :
        
        #skip header
        csv1reader = csv.reader(csv1)
        next(csv1reader, None)

        #skip header
        csv2reader = csv.reader(csv2)
        next(csv2reader, None)

        #variables to keep track of data
        i = 0
        currline = ''
        csv2len = 0
        csv1len = 0

        #counts of positive and negative documents
        pos = 0
        neg = 0

        try :#certain lines might error
            
            #go through the first csv
            for line in csv1reader :
                #newline is the new csv value
                newline = []

                #get the name of the drug (want to remove it to make sure it isn't just associating certain drugs with pos/neg)
                drug = line[1].lower()

                #extract rating from old one
                rating = line[4]

                #convert 1-10 rating to pos or neg review
                if int(rating) >= 7 :
                    pos += 1
                    rating = 1
                else :
                    neg += 1
                    rating = 0

                #convert html tags to matching punctuation
                currline = html.unescape(line[3]).lower()

                #clean up certain punctuation and remove the name of the drug
                currline = currline.replace('/', ' ').replace('.', ' ').replace('-', ' ').replace(drug, '')

                #remove the rest of the punctuation
                currline = currline.translate(str.maketrans('', '', string.punctuation))

                #create the csv value with rating and review
                newline = [rating, currline]

                #add it to out file
                newcsv.append(newline)

                #count len of file
                csv1len += 1

            #same as above but for second csv file
            for line in csv2reader :

                newline = []

                drug = line[1].lower()

                rating = line[4]
                if int(rating) >= 7:
                    pos += 1
                    rating = 1
                else :
                    neg += 1
                    rating = 0
                currline = html.unescape(line[3]).lower()
                currline = currline.replace('/', ' ').replace('.', ' ').replace('-', ' ').replace(drug, '')

                currline = currline.translate(str.maketrans('', '', string.punctuation)).lower()

                newline = [rating, currline]

                newcsv.append(newline)
                csv2len += 1

        except Exception as e:
            print("---------------- ERROR ---------------------")
            print(e)
            print(currline)

        print(f"Length of csv1 {csv1len}, csv2 {csv2len}, newcsv {len(newcsv)}, positive {pos}, negative {neg}")

        newcsv = pandas.DataFrame(newcsv)
        newcsv.to_csv(newcsvpath, index=False, header=False)

        