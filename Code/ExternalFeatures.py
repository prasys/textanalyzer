import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import gensim
from string import ascii_lowercase
#import Use_NN as nn
import re
from sklearn.svm import LinearSVC , SVC
from sklearn import preprocessing
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier , RadiusNeighborsClassifier
from collections import Counter
import string
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#test22


removeUnWanted = re.compile('[\W_]+') #strip off the damn characters
tagQues = ["isn't she","don't they","aren't we","wasn't it","didn't he","weren't we","haven't they","hasn't she","hadn't he","hadn't we","won't she","won't they","won't she","can't he","mustn't he","are we","does she","is it","was she","did they","were you","has she","has he","had we","had you","will they","will he","will she","will he","can she","must they"]
metaphor = []

def loadMetaphors(filename):
  with open(filename) as f:
    content = f.readlines()
  content = [x.strip() for x in content]
  return content

# Re-WRITE THIS FUNCTION 
def checkMetaphors(text,list_=metaphor):
  if any(word in text for word in list_):
    return 1 #if we have found any of the words for it
  else: # if we cannot find any 
    return 0 # if we have not found any of the words


# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

def countOfWords(text):
    count = len(re.findall(r'\w+', text))
    return count

def commentCleaner(df,ColumnName):
    df[commentParent] = df[commentParent].str.lower()
   # df[commentParent] = df[commentParent].str.replace("[^abcdefghijklmnopqrstuvwxyz1234567890' ]", "")

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

# Converts to POS Tags that can be used for other stuff
# def tag(sent):
#     words=nltk.word_tokenize(sent)
#     tagged=nltk.pos_tag(words)
#     return tagged

def tagQuestions(text,list_=tagQues):
  if any(word in text for word in list_):
    return 1
  else: # if we cannot find any 
    return 0 

#Checks for Nouns , To Implement the method found in Cindy Chung's Physc Paper (Search for Cindy Chung and James Pennebaker and cite here)

# def checkForNouns(text,method='None'):
#     counter = 0
#     counter2 = 0
#     if "aa" in text: #Dummy variable to inform that it is outside , so we dont' track them 
#         return counter
#     else:
#         wrb = tag(text)
#         index = 0
#         for row  in wrb:
#             POSTag = wrb[index][1]
#           #  print(POSTag)
#             if (POSTag in "IN") or (POSTag in "PRP") or (POSTag in "DT") or (POSTag in "CC") or (POSTag in "VB") or (POSTag in "VB") or (POSTag in "PRP$") or (POSTag is "RB"):
#                 counter = counter+1
#             else:
#                 counter2 = counter2+1
                
#             index = index + 1
#         if "function" in method:
#             return counter
#         elif "ratio" in method:
#             return abs(counter2/counter)
#         else:
#             return counter2


#Calculates the amount of interjections 
def getInterjections(blah):
   # blah = blah.lower()
    doc = nlp(blah)
    result = 0
    for word in doc:
        if word.pos_ is 'INTJ': #INTJ is interjection with spacy
            result += 1
    return result

#Count the Number of Hyperboles for us to do calculations
def getHyperboles(blah,dataFrameObject):
   # blah = blah.lower()
    doc = nlp(blah)
    flag = False # a flag to check if there is the word is being found or not 
    result = 0 # number of strong subjective/sentiments 
    for word in doc:
        if word.is_punct is False: #This is not a punctuation anyway , so we can take a look at what to do next
          # print(word)
          checkIndex = dataFrameObject.loc[dataFrameObject['Word']==word.text] # check the index
        if checkIndex.empty:
          result += 0 #ignore this as we did not find any positive hyperbole
          flag = False
        else:
          flag = True
        if flag is True:
          if len(checkIndex) == 1: # if there is only one item for it
            t = dataFrameObject[dataFrameObject.Word==word.text].Subjectivity.item()
          else: 
            t = dataFrameObject[dataFrameObject.Word==word.text].iloc[0] #
            t = t['Subjectivity']
          if t == 'strongsubj':
            result += 2 #strong sentiment
          else:
            result += 1 #weak sentiment
    return result

# Our method to count the Punctuation for it
def getPunctuation(text): 
  punctuation = []
  for char in text:
    if char in string.punctuation:
      punctuation.append(char)

  counter = Counter(punctuation)

  if len(punctuation) == 0:
    return 0 # if we only have none in it

  if len(punctuation) == 1 or len(counter) == 1:
    return 1 # if we only have 1 
  else :
    return 2 #we have multiple elements inside , we just get the total number of them

# Made more Pythonic by implementing the suggestion from https://stackoverflow.com/questions/49078267/counting-upper-case-words-in-a-variable-in-python
def countTotalCaps(text):
  return (sum(map(str.isupper,text.split())))

# checks for the quotation marks if they are present in the system and would return the amount that is present 

def detectQuotationMarks(text):
  startIndex = text.find('\"')
  if startIndex == -1:
    return 0 #if we did not get the quotation mark for it
  else:
    return 1 # if we have found the quitation mark


def checkForExclamation(text):
  #return 1 if there is 1 , and 2 if there are multiple uses of markers , and 0 if there is none
  result = 0
  for char in text:
        if char == '!':
            result +=1
    
  if result == 0:
    return 0 # if there is nothing at all 
  elif result == 1:
        return 1 # if there is 1
  else:
    return 2 # if there is more than 1
  return result


def cleanSpecialCharacters(text):
    return (re.sub( '[^a-z0-9\']', ' ', text))

def scrub_words(text):
    """Basic cleaning of texts."""
    """Taken from https://github.com/kavgan/nlp-in-practice/blob/master/text-pre-processing/Text%20Preprocessing%20Examples.ipynb """
    
    # remove html markup
    text=re.sub("(<.*?>)","",text)
    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)"," ",text)
    
    # remove the extra spaces that we have so that it is easier for our split :) Taken from https://stackoverflow.com/questions/2077897/substitute-multiple-whitespace-with-single-whitespace-in-python
    text=re.sub(' +', ' ', text).strip()
    return text

def LemenSpacy(text,useNLPObj=False,isFirstTime=False):
    # if isFirstTime and useNLPObj:       
    #     nlp = spacy.load("en_core_web_sm")
    #     print("Load Spacy")
    #     nlp.tokenizer = Tokenizer(nlp.vocab) #lod our customized tokenizer overwritten method
    #     isFirstTime  = False
    text = text.lower()
    doc = nlp(text)
    tokens = []
    for token in doc:
        if token.is_punct is False:
            if token.orth_ == 've': #special handling case
                tokens.append("'ve")
            elif token.orth_ == "  ":
                tokens.append(" ")
            else:
                if token.lemma_ == '-PRON-':
                    tokens.append(token.orth_)
                else:
                    tokens.append(token.lemma_)

    return (' '.join(tokens)) 
    #return tokens


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print(score["compound"])
    return score['compound']
    #print("{:-<40} {}".format(sentence, str(score)))

#Main Method
if __name__ == '__main__':
  nlp = spacy.load("en_core_web_sm") # load the NLP toolkit software
  commentChild = 'Cleaned' # name of the field for child
  commentParent = 'lemen_parent'
  analyser = SentimentIntensityAnalyzer()
  df = pd.read_csv('/data/pradeesh/data/test_alta_dataset.csv') #  Read the Classifier Software
  df2 = pd.read_csv('/data/pradeesh/data/MPQAHyperbole.csv') # add the path to the files , so it can be read properly
  #df2.drop(df.filter(regex="Unname"),axis=1, inplace=True) #do some clean ups


  # LEMENTIZE the parents comment
  df['lemen_parent'] = df['Parent'].apply(LemenSpacy)

  ## FOR THE COMMENTS 
  df['exclamation_comment'] = df[commentChild].apply(checkForExclamation) #detect exclamation
  # df['tagQuestions_comment'] = df[commentParent].apply(tagQuestions)  # detect tag questions
  df['interjections_comment'] = df[commentChild].apply(getInterjections) # get any interjections if there are present
  df['punch_comment'] = df[commentChild].apply(getPunctuation) # get the no of punctuations to be used as features
  df['hyperbole_comment'] = df[commentChild].apply(getHyperboles,dataFrameObject=df2) # get the no of punctuations to be used as features
  df['quotation_comment'] = df[commentChild].apply(detectQuotationMarks) # adding to detect qutation marks
  df['totalCaps_comment'] = df[commentChild].apply(countTotalCaps) # adding support to count total number of CAPS
  df['noOfWords_comment'] = df[commentChild].apply(countOfWords) #count no of words

  ## FOR THE PARENT COMMENTS

  df['exclamation_parent'] = df[commentParent].apply(checkForExclamation) #detect exclamation
  df['tagQuestions_parent'] = df[commentParent].apply(tagQuestions)  # detect tag questions
  df['interjections_parent'] = df[commentParent].apply(getInterjections) # get any interjections if there are present
  df['punch_parent'] = df[commentParent].apply(getPunctuation) # get the no of punctuations to be used as features
  df['hyperbole_parent'] = df[commentParent].apply(getHyperboles,dataFrameObject=df2) # get the no of punctuations to be used as features
  df['quotation_parent'] = df[commentParent].apply(detectQuotationMarks) # adding to detect qutation marks
  df['totalCaps_parent'] = df[commentParent].apply(countTotalCaps) # adding support to count total number of CAPS
  df['noOfWords_parent'] = df[commentParent].apply(countOfWords) # adding support for the nof of parent comments


  df.to_csv('/data/pradeesh/data/test_processed.csv')


  print(df)

    
