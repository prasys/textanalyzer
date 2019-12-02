import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import gensim
from pprint import pprint
from string import ascii_lowercase
#import Use_NN as nn
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split , KFold , LeaveOneOut , LeavePOut , ShuffleSplit , StratifiedKFold , GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor , VotingClassifier , RandomTreesEmbedding, ExtraTreesClassifier , RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.svm import LinearSVC , SVC
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import TheilSenRegressor , SGDClassifier
from sklearn.naive_bayes import GaussianNB , BernoulliNB, MultinomialNB , ComplementNB
from sklearn.linear_model import LogisticRegressionCV , PassiveAggressiveClassifier, HuberRegressor
from sklearn.metrics import f1_score , recall_score , accuracy_score , precision_score , jaccard_score , balanced_accuracy_score, confusion_matrix
from mlxtend.plotting import plot_decision_regions, plot_confusion_matrix
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier , RadiusNeighborsClassifier
import nltk
from nltk.tokenize import RegexpTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from imblearn.pipeline import make_pipeline
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
import string
import xgboost as xgb
from pushover import Client




removeUnWanted = re.compile('[\W_]+') #strip off the damn characters
isClassify = False #to run classification on test data
isCreationMode = False
isWord2Vec = False
isEmbeddings = True
isBOW = False
doc2VecFileName ="doc2vec"
useSMOTE = True
searchParams = False
STATE = 21
#logistic , nb , svm , xgboost, rf
DETERMINER = 'xgboost'
embedType = 'bert' #or bert


# Take any text - and converts it into a vector. Requires the trained set (original vector) and text we pan to infer (shall be known as test)
def vectorize(train,test):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    vectorizer = CountVectorizer(ngram_range=(2,3),min_df=0, lowercase=True, analyzer='char_wb',tokenizer = token.tokenize, stop_words='english') #this is working
    #vectorizer = CountVectorizer(min_df=0, lowercase=True)
   # vectorizer = TfidfTransformer(use_idf=True,smooth_idf=True)
    x = vectorizer.fit(train)
    x = vectorizer.transform(test)
    return x

def loadEmbeddings(filename):
	embeddings = numpy.load(filename,allow_pickle=True)
	print(embeddings.shape)
	return embeddings

# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    colName = ['ID','Comment','Prediction']
    column_dtypes = {
                 'ID': 'uint8',
                 'Comment' : 'str',
                 'Prediction' : 'uint8'
                 }
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0, dtype=column_dtypes,usecols=colName,encoding = "ISO-8859-1")
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

def read_csv2(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    colName = ['ID','Comment']
    column_dtypes = {
                 'ID': 'uint8',
                 'Comment' : 'str'
                 }
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0, dtype=column_dtypes,usecols=colName)
    #df_chuck = df_chuck.fillna(0)
    return df_chunk

#Classify Sarcasm Based on the Neural Network That Was Trained for it - TO DO 
def detectSarcasm(text):
    #text = re.sub('[^A-Za-z0-9]+', '', text)
   # print(text)
  # return ("3")
    return nn.use_neural_network(text)

def calcSyllableCount(text):
    return textstat.syllable_count(text, lang='en_US')


def calcLexCount(text):
    return textstat.lexicon_count(text)

def commentCleaner(df):
    df['Comment'] = df['Comment'].str.lower()
   # df['Comment'] = df['Comment'].str.replace("[^abcdefghijklmnopqrstuvwxyz1234567890' ]", "")

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

# Converts to POS Tags that can be used
def tag(sent):
    words=nltk.word_tokenize(sent)
    tagged=nltk.pos_tag(words)
    return tagged

#Checks for Nouns , To Implement the method found in Cindy Chung's Physc Paper (Search for Cindy Chung and James Pennebaker and cite here)
def checkForNouns(text,method='None'):
    counter = 0
    counter2 = 0
    if "aa" in text: #Dummy variable to inform that it is outside , so we dont' track them 
        return counter
    else:
        wrb = tag(text)
        index = 0
        for row  in wrb:
            POSTag = wrb[index][1]
          #  print(POSTag)
            if (POSTag in "IN") or (POSTag in "PRP") or (POSTag in "DT") or (POSTag in "CC") or (POSTag in "VB") or (POSTag in "VB") or (POSTag in "PRP$") or (POSTag is "RB"):
                counter = counter+1
            else:
                counter2 = counter2+1
                
            index = index + 1
        if "function" in method:
            return counter
        elif "ratio" in method:
            return abs(counter2/counter)
        else:
            return counter2

#Given an un-seen dataframe and [TO DO - the column] , it will convert it into Matrix 
def convertToVectorFromDataframe(df):
    matrix = []
    targets = list(df['tokenized_sents'])
    for i in range(len(targets)):
        matrix.append(model.infer_vector(targets[i])) # A lot of tutorials use the model directly , we will do some improvement over it
    targets_out = numpy.asarray(matrix)
    return (matrix)

#A simple method which basically takes in the tokenized_sents and the tag and starts do it. 
def make_tagged_document(df,train):
    #  taggeddocs = []
    for doc, tanda in zip(df['tokenized_sents'], train):
        yield(TaggedDocument(doc,[tanda]))



def calculateScoresVariousAlphaValues(predicted_data,truth_data,threshold_list=[0.00,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99,1.00]):
  for i in threshold_list:
    squarer = (lambda x: 1 if x>=i else 0)
    fucd = numpy.vectorize(squarer)
    vfunc = fucd(predicted_data)
    f1score = f1_score(y_true=truth_data,y_pred=vfunc)
    print(str(i)+","+str(perf_measure(truth_data,vfunc)))
    #print(confusion_matrix(vfunc, truth_data))
    #print(str(i)+","+ str(f1score))


# Creates a Doc2Vec Model by giving an input of documents [String]. It's much of an easier way. It then saves to disk , so it can be used later :) 
def createDoc2VecModel(documents,tag):
  docObj = list(make_tagged_document(documents,tag)) # document that we will use to train our model for
  model = Doc2Vec(documents=docObj,vector_size=500,
            # window=2, 
            alpha=.025,
            epochs=100, 
            min_alpha=0.00025,
            sample=0.335,
            ns_exponent=0.59,
            dm_concat=0,
            dm_mean=1,
            # negative=2,
            seed=10000, 
            min_count=2, 
            dm=0, 
            workers=4)
  model.save(doc2VecFileName) #our file name
  return model

# Loads Doc2Vec model based on the filename given
def loadDoc2VecModel(filepath=doc2VecFileName):
  model = Doc2Vec.load(filepath)
  return model

# Implements Class Weight to ensure that fair distribution of the classes
def get_class_weights(y):
    counter = Counter(y)
    majority = max(counter.values())
    return  {cls: round(float(majority)/float(count), 2) for cls, count in counter.items()}


# Selects a Classifier to perform the task
def selectClassifier(weights='balanced',classifymethod='logistic'):
  #classifier = RandomForestRegressor(n_estimators=100)
  #clf = svm.NuSVC(kernel='rbf',decision_function_shape='ovo',probability=True)
  #classifier = LinearSVC(random_state=21, tol=1e-4,C=1000,fit_intercept=False)
  if 'logistic' in classifymethod:
    #cy = LogisticRegression(fit_intercept=True, max_iter=8000,solver='newton-cg',random_state=STATE,class_weight=weights)
    cy = LogisticRegression(C=5.0, fit_intercept=False, max_iter=100, penalty= 'l2', solver='sag', tol=0.0001)
    return cy
  elif 'nb' in classifymethod:
  	cy = GaussianNB()
  	return cy
  elif 'xgboost' in classifymethod:
    cy = xgb.XGBClassifier(colsample_bytree= 0.6, gamma= 2, max_depth= 5, min_child_weight= 5, n_estimators= 100, subsample= 0.8)
    return cy
  elif 'svm' in classifymethod:
    cy = SVC(random_state=STATE,probability=True,C= 10, gamma= 0.001, max_iter= 500, tol= 0.001)
    return cy
  elif 'rf' in classifymethod:
    cy = RandomForestClassifier(bootstrap = True, class_weight= 'balanced_subsample', criterion= 'entropy', max_depth= 8, max_features = 'log2', min_samples_split= 30, min_weight_fraction_leaf= 0.0, n_estimators = 300,random_state=STATE)
    return cy
  elif 'kn' in classifymethod:
    cy = MLPClassifier(hidden_layer_sizes=50,learning_rate='adaptive',random_state=STATE,solver='lbfgs')
    return cy
  else:
    return null


def gridParameters(classifyMethod):
  if 'rf' in classifyMethod:
    grid_param = {
    'n_estimators': [100, 300, 500, 800, 900,1000],
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto','sqrt','log2'],
    'min_samples_split' : [2,4,8,10,15,30],
    'class_weight': ['balanced','balanced_subsample'],
    'min_weight_fraction_leaf' : [0.0,0.1,0.3,0.5],
    'max_depth': [1,3,5,8,10,15,20],
    'bootstrap': [True, False]
  }
  elif 'logistic' in classifyMethod:
    grid_param = {
    'penalty' : ['l2'],
    'tol': [1e-4, 1e-5],
    'C': [0.5,1.0,5.0],
    'fit_intercept': [True,False],
    'solver': ['newton-cg','lbfgs','sag'],
    'max_iter': [100,200]
    }

  elif 'xgboost' in classifyMethod:
    grid_param = {
    'n_estimators': [100,200,600,800],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5]
  }

  elif 'svm' in classifyMethod:
    grid_param = {
    # 0.01, 0.1, 1, 10,100,1000,3000 , 1e-4, 1e-5, 5e-4 , 5e-5,5e-10 , 500,1000,8000
    'C' : [0.001,0.1,1,10,100,1000,3000],
    'tol': [1e-3,1e-4,1e-5,5e-4,5e-5,5e-10],
    'max_iter': [100,500,1000],
    'gamma': [0.001,0.005,0.010],
  }

  return grid_param


def getChars(s):
  count = lambda l1,l2: sum([1 for x in l1 if x in l2])
  return (count(s,set(string.punctuation)))

def mergeMatrix(matrixa,matrixb):
  print(matrixa.shape)
  print(matrixb.shape)
  print(matrixb)
  return(numpy.concatenate((matrixa, matrixb[:,None]), axis=1))

def w2v_preprocessing(df):
  df['Comment'] = df['Comment'].str.lower()
  df['nouns'] = df['Comment'].apply(checkForNouns,'function')
  df['tokenized_sents'] = df.apply(lambda row: nltk.word_tokenize(row['Comment']), axis=1)
  df['uppercase'] = df['Comment'].str.findall(r'[A-Z]').str.len() # get upper case
  df['punct'] = df['Comment'].apply(getChars)


def FoldValidate(original,truth,classifier,iter=3):
  Val = StratifiedKFold(n_splits=iter, random_state=STATE, shuffle=True) # DO OUR FOLD here , maybe add the iteration
  scores = []
  tns = [] # true neg
  fps = [] # false positive
  fns = [] # false negative
  tps = [] # true positive
  for train_index,test_index in Val.split(original,truth):
    model2 = classifier
    model2.fit(original[train_index], truth[train_index])
    x_output = model2.predict(original[test_index])
   # print(x_output.shape)
   # print(truth[test_index].shape)
    #scores.append(classifier.score(x_output, truth[test_index]))
    tn, fp , fn , tp = confusion_matrix(x_output, truth[test_index]).ravel()
    score = accuracy_score(x_output,truth[test_index])
    tns.append(tn)
    fps.append(fp)
    fns.append(fn)
    tps.append(tp)
    scores.append(score)

  print("TP is,",numpy.mean(tps))
  print("FP is,",numpy.mean(fps))
  print("FN is,",numpy.mean(fns))
  print("TN is,",numpy.mean(tns))
  print("Avg Accuracy is,",numpy.mean(scores))


    #score = classifier.score(original[train_index], truth[train_index])
    #print("Linear Regression Accuracy (using Weighted Avg):", score)
  #  tester = classifier.predict_proba(original[test_index])
  #  tester = tester[:,1]
  #  calculateScoresVariousAlphaValues(tester,truth[test_index])
 # scores = numpy.asarray(scores)
  #print("Accuracy Score Is:", numpy.mean(scores))


   # print("Valuesdfs for train are ", train_index)
   # print("Values for test index are ",test_index)
   # print("Testing with the values",original[train_index])
   # print("Testing it with the values",truth[train_index])
  	#weights = get_class_weights(truth_data[test_index]) # implement the weights
  	#model2.fit(classifer_data, truth_data, class_weight=weights)
  	#unseendata = convertToVectorFromDataframe(test)
  	#tester = classifier.predict_proba(unseendata)
  	#tester = tester[:,1]
  	#calculateScoresVariousAlphaValues(tester,truth_data)

def showGraph(model):
  xgb.plot_importance(classifier, importance_type='gain',max_num_features=10)
  plt.show()


# Performs a Grid Search
def gridSearch(model,params,x_train,y_train):
  params = gridParameters(params)
  gd_sr = GridSearchCV(estimator=model,
                     param_grid=params,
                     scoring='accuracy',
                     cv=3,
                     n_jobs=5)
  gd_sr.fit(x_train, y_train)
  best_parameters = gd_sr.best_params_
  score2 = ("Best Performing Parameters",best_parameters)
  best_score = gd_sr.best_score_
  score1 = ("Best Score",best_score)
  #output = best_parameters + " " + best_score
  return (score2,score1)





def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1
    return(TP,FP)

#return(TP, FP, TN, FN)

#Calculates the interjection 
def getInterjections(blah):
   # blah = blah.lower()
    doc = nlp(blah)
    result = 0
    for word in doc:
        if word.pos_ is 'INTJ': #INTJ is interjection with spacy
            result += 1
    return result


def checkForExclamation(text):
  #return 1 if there is 1 , and 2 if there are multiple uses of markers , and 0 if there is none
  result = 0
  for char in text:
        if char == '!':
            result +=1
    
  if result == 0:
    return 0
  elif result == 1:
        return 1
  else:
    return 2
  return result


#Main Method
if __name__ == '__main__':
  #train_classifier_3
    df = read_csv("train_classifier.csv") #Read CSV which contains everything
    # To implement method to do the pre-processing here 
