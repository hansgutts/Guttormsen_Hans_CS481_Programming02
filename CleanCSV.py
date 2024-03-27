import csv
import pandas
import html
import string

csv1path = "UCIdrug_test.csv"
csv2path = "UCIdrug_train.csv"
newcsvpath = "UCIdrugClean.csv"

newcsv = []

with open(csv1path, errors='ignore') as csv1 :
    with open(csv2path, errors='ignore') as csv2 :
        csv1reader = csv.reader(csv1)
        next(csv1reader, None)

        csv2reader = csv.reader(csv2)
        next(csv2reader, None)

        i = 0
        currline = ''
        csv2len = 0
        csv1len = 0
        try :
            
            for line in csv1reader :
                newline = []
                rating = line[4]
                if int(rating) >= 7 :
                    rating = 1
                else :
                    rating = 0
                currline = html.unescape(line[3])
                currline = currline.replace("\"", '').replace(".", ' ')

                currline = currline.translate(str.maketrans('', '', string.punctuation)).lower()

                newline = [rating, currline]

                newcsv.append(newline)
                csv1len += 1

            
            for line in csv2reader :

                newline = []
                rating = line[4]
                if int(rating) >= 7:
                    rating = 1
                else :
                    rating = 0
                currline = html.unescape(line[3])
                currline = currline.replace('/', ' ').replace('.', ' ').replace('-', ' ')

                currline = currline.translate(str.maketrans('', '', string.punctuation)).lower()

                newline = [rating, currline]

                newcsv.append(newline)
                csv2len += 1

        except Exception as e:
            print("---------------- ERROR ---------------------")
            print(e)
            print(currline)

        print(f"Length of csv1 {csv1len}, csv2 {csv2len}, newcsv {len(newcsv)}")

        newcsv = pandas.DataFrame(newcsv)
        newcsv.to_csv(newcsvpath, index=False, header=False)

        