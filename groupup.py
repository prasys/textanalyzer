#!/usr/bin/python3

from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd
import sys
import textstat
import numpy as numpy
import math
import pickle
import gensim
from gensim import corpora
from pprint import pprint
from nltk.corpus import stopwords
from string import ascii_lowercase
import gensim, os, re, pymongo, itertools, nltk, snowballstemmer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandarallel import pandarallel
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.metrics.pairwise import cosine_similarity , linear_kernel
from sklearn.cluster import KMeans
import matplotlib.pylab as plt
from sklearn.metrics.pairwise import *
import seaborn as sns
import numexpr as ne
import scipy.sparse as sparse
from pathlib import Path
from wordcloud import WordCloud
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import glob


#from odo import odo
#import dask.dataframe as pd

#star_rating
#vine
#cat_text = 'verified_purchase'
#cat_text = sys.argv[1] #category we want to extract
db_name = sys.argv[1] #get the db name for it
flag_calc_scores = False
isMongo = False
isCalc = True
#size = 500000
chunk_list = []  # append each chunk df here 
analyser = SentimentIntensityAnalyzer()
pandarallel.initialize(progress_bar=True)
#cat_text = 'vine'
#db_name = "amazon_reviews_us_Books_v1_02.tsv"


def stemit():
	stemmer = snowballstemmer.EnglishStemmer()
	stop = stopwords.words('english')
	stop.extend(['may','also','zero','one','two','three','four','five','six','seven','eight','nine','ten','across','among','beside','however','yet','within']+list(ascii_lowercase))
	stoplist = stemmer.stemWords(stop)
	stoplist = set(stoplist)
	stop = set(sorted(stop + list(stoplist))) 
	return stop

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print(score["compound"])
    return score
    #print("{:-<40} {}".format(sentence, str(score)))


def read_csv(filepath):
	parseDate = ['review_date']
	#dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
	#colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
	colName = ['customer_id','product_category','star_rating','helpful_votes','total_votes', 'verified_purchase', 'vine','readscore', 'compound' ,'review_date']
	column_dtypes = {'marketplace': 'category',
                 'customer_id': 'uint32',
                 #'review_id': 'str',
                 #'product_id': 'str',
                 #'product_parent': 'uint32',
                # 'product_title' : 'str',
                 'product_category' : 'str',
                 'star_rating' : 'Int64',
                 'helpful_votes' : 'Int64',
                 'total_votes' : 'Int64',
                 'vine' : 'str',
                 #'review_date' : 'str',
                 'verified_purchase' : 'str',
                 #'review_headline' : 'str',
                 #'review_body' : 'str',
                 'readscore' : 'float32',
                 'compound' : 'float32'
                 }
	#df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
	df_chunk = pd.read_csv(filepath, sep=',', header=0, error_bad_lines=False, dtype=column_dtypes, usecols=colName, parse_dates=parseDate)
	#df_chuck = df_chuck.fillna(0)
	return df_chunk


def establishMongoDB():
	client = MongoClient('mongodb://localhost:27017/')
	return clientb


def getAllFiles(fileFormat):
	all_txt_files =[]
	cwd = os.getcwd()
	fileFormatEmbed = "*." + fileFormat
	all_txt_files = glob.glob(os.path.join(cwd,fileFormatEmbed))
	return all_txt_files


def fetch_db(text):
	fetch_db = db[text].find()
	return fetch_db



def get_top_n_words(corpus, n=None):
    """
    List the top n words in a vocabulary according to occurrence in a text corpus.
    
    get_top_n_words(["I love Python", "Python is a language programming", "Hello world", "I love the world"]) -> 
    [('python', 2),
     ('world', 2),
     ('love', 2),
     ('hello', 1),
     ('is', 1),
     ('programming', 1),
     ('the', 1),
     ('language', 1)]
    """
    print("Processing Matrix")
    vec = TfidfVectorizer(stop_words='english',min_df=0.02, max_df=0.90, max_features=500,use_idf=True,smooth_idf=True)
    matrix = vec.fit_transform(corpus)
    output = cosine_distances(matrix)

    #km = KMeans(n_clusters=5)
    #km.fit(matrix)
    print(output)

   # bag_of_words = vec.transform(corpus)
   # sum_words = bag_of_words.sum(axis=0) 
   # words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
   # words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
   # return words_freq[:n]

def calculateDocumentSim(document):
	print("Count Vector")
	cv = TfidfVectorizer(stop_words='english',min_df=0.02, max_df=0.85, max_features=100000,use_idf=False,smooth_idf=True)
	#cv = CountVectorizer(stop_words='english',min_df=0.02, max_df=0.85, max_features=100000)
	dt_mat = cv.fit_transform(document)
	print("TFIDF Vector")
	#tfidf = TfidfTransformer(smooth_idf=True,use_idf=False)
	#dt_mat = tfidf.fit_transform(dt_mat)
	#pairwise_similarity = dt_mat * dt_mat.T
	dt_mat = dt_mat.astype("float32") #make things way faster
	#features = cv.get_feature_names()
	#df_t_reduced = SelectKBest(k=50).fit_transform(dt_mat,document)
	#features = dt_mat.get_feature_names()
	print(features)
	#km = KMeans(n_clusters=5)
	#km.fit(dt_mat)
	#clusters = km.labels_.tolist()
	#print(clusters)
	#wolo = get_top_tf_idf_words(dt_mat,features,2)
	#rint(wolo)
	#print(dt_mat.dtype)
	#print(dt_mat.shape)
#	print("Calculate the Dot Product")
	#ne.evaluate(dt_mat.T)
	#dt_mat = linear_kernel(dt_mat,dense_output=False)
#	tsvd = TruncatedSVD(n_components=10)
	#X_sparse_tsvd = tsvd.fit(dt_mat).transform(dt_mat)
	#dt_mat = X_sparse_tsvd.dot(X_sparse_tsvd.T)
	#pairwise_similarity = dt_mat
	#pairwise_similarity = dt_mat * dt_mat.T # Multiple the matrix by it's transformation to get the identify matrix to find the similarity 
	print("Return")
	return dt_mat

def func(row):
    xml = ['<doc>']
    for field in row.index:
        xml.append('  <{0}>{1}</{0}>'.format(field, row[field]))
    xml.append('</doc>')
    return '\n'.join(xml)

#sample = denoise_text(sample)
#print(sample)

#Debug code for args 	
#print ("This is the name of the script: " + sys.argv[0])
#print ("Number of arguments: " +  str(sys.argv))
#print ("The arguments are: " + str(sys.argv))



if isMongo == True:
	print("lol")

else:
	#print(df.head())
	isRun = False
	#chunk_list
	#chunk_list =[]
	df = read_csv(db_name)
	print(df.shape)
	#vc = df['customer_id'].value_counts()
	#df[df['customer_id'].isin(vc.index[vc.values > 1])].uid.value_counts() # drop single occurance of customer purchase as we do not need them to be processed at all 
	cat_features =['vine','verified_purchase','product_category','star_rating']
	cont_features =['compound','readscore','total_votes','helpful_votes']
	print("Normalizing features")
	for index, col in enumerate(eremucat_features):
		print("current index " + str(index))
		dummies = pd.get_dummies(df[col], prefix=col)
		df = pd.concat([df, dummies], axis=1)
		df.drop(col, axis=1, inplace=True)
	print("Processed them")
	std_data = df[cont_features]
	print("Make Scaler to fix the data")
	scaler = StandardScaler().fit(std_data)
	new_values = scaler.transform(std_data)
	df[cont_features] = new_values
	print("Save to the dataframe")
	df.to_pickle('./output.pkl')








	# Remove outliers or 1 post reviews so to be ignored for it to improve processing speed times 
	#group_df = df.groupby('customer_id') # group them by their ids


	#group_df = df.groupby('customer_id')
	#output = group_df['review_date'].apply(lambda x: x - x.iloc[0]) #calculate diff for the date 

	#df.assign(output) #output the new column








	#https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f








	#https://stackoverflow.com/questions/38915186/pandas-use-groupby-to-count-difference-between-dates




 #41 Miliion customers 




	#Process them 

	#41,914,525 rows ....which i don't think is correcttt

	#22,150,186

	#64,064,711, are the people who have posted at least 2 or more comments 









	# Standardize and assume all features are important , maybe some are not - but lets take a look..... we have a sample of how it works and lets see if it does classification 





