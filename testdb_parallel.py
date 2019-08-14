#!/usr/bin/python3

from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd
import sys
import textstat
import numpy as numpy
import math

#from odo import odo
#import dask.dataframe as pd

#star_rating
#vine
#cat_text = 'verified_purchase'
cat_text = sys.argv[1] #category we want to extract
db_name = sys.argv[2] #get the db name for it
flag_calc_scores = False
isMongo = False
size = 500000
chunk_list = []  # append each chunk df here 
#cat_text = 'vine'
#db_name = "amazon_reviews_us_Books_v1_02.tsv"


def read_csv(filepath):
	parseDate = ['review_date']
	column_dtypes = {'marketplace': 'category',
                 'customer_id': 'uint32',
                 'review_id': 'str',
                 'product_id': 'str',
                 'product_parent': 'uint32',
                 'product_title' : 'str',
                 'product_category' : 'category',
                 'star_rating' : 'uint8',
                 'helpful_votes' : 'uint8',
                 'total_votes' : 'uint8',
                 'vine' : 'category',
                 'verified_purchase' : 'category',
                 'review_headline' : 'str',
                 'review_body' : 'str',}
	df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes)
	df_chuck = df_chuck.fillna(0)
	return df_chunk



def establishMongoDB():
	client = MongoClient('mongodb://localhost:27017/')
	return clientb


def sentiment_calc_polarity(text):
	#score = TextBlob(text).sentiment.polarity
	try:
		#if score > 0.4:
			#print (text)
		#if "delivery" in text:
		#	print ("[DELIVERY DETECTED] " + text)
		#print (TextBlob(text).sentiment.polarity)
		return TextBlob(text).sentiment.polarity
	except:
		return None

def sentiment_calc_subjectivity(text):
	#score = TextBlob(text).sentiment.subjectivity
	try:
		#print (TextBlob(text).sentiment.subjectivity)
		return TextBlob(text).sentiment.subjectivity
	except:
		0
		return None

def fetch_db(text):
	fetch_db = db[text].find()
	return fetch_db

def test(text):
	#print (text)
	score = textstat.automated_readability_index((str (text)))
	if math.isnan(score) == True:
		return 0.0
	else:
		return score

def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def remove_between_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text

def memory_usage(df):
	return(round(df.memory_usage(deep=True).sum() / 1024 ** 2, 2))

#sample = denoise_text(sample)
#print(sample)

#Debug code for args 	
#print ("This is the name of the script: " + sys.argv[0])
#print ("Number of arguments: " +  str(sys.argv))
#print ("The arguments are: " + str(sys.argv))



if isMongo == True:
	client = establishMongoDB()
	db = client.amazon
	col = client['amazon_o'][db_name]
	#collection = fetch_db('sample') #STG
	print("Loading data from MongoDB")
	collection = fetch_db(db_name)
	print("Loading data into Pandas")
	df = pd.DataFrame(list(collection))
	#print(df.head())
else:
	df_chunk = read_csv(db_name) # read our CSV file location - needs to be absolute file path. To be tested it out
	for chunk in df_chunk:
		print("CONVERTING THEM")
		#chunk['star_rating'] = 
		#hunk['review_date'] = pd.to_datetime(chunk['review_date'])
		chunk['year'] = chunk['review_date'].dt.year
		print("APPLYING READABILITY SCORE")
		chunk['readscore'] = chunk['review_body'].apply(test)
		chunk_list.append(chunk)
	df = pd.concat(chunk_list)
	print(df.head())

	#print(df.dtypes())





# concat the list into dataframe 





#vine
#rev



#print("Calculating Readability Score")
#df['readscore'] = df['review_body'].apply(test)

#df['review_date'] = pd.to_datetime(df['review_date']) #convert time
#df['year'] = df['review_date'].year
#grouped = df.groupby(cat_text) #group by star rating for amazon here
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean()) std
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean())
#print(group1)
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().std())
#print(group1)
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().sum()) # Calculate the Total Number of Words
#print(group1)
#print ("Calculating Polarity")
#df['polarity'] = df['review_body'].apply(sentiment_calc_polarity)

#print ("Calculating Subjectivity")
#df['subjectivity'] = df['review_body'].apply(sentiment_calc_subjectivity)


data = df.to_dict(orient='records')

col.insert_many(data)

#odo(df,db[colle])


if flag_calc_scores == True:
	for group_name, df_group in grouped:
		print(cat_text + " " + str(group_name))
		#print("Total No of Reviews" + str(df_group['review_body'].count()))
		#df_group['test'] = df_group['review_body'].apply(sentiment_calc_polarity)
		#peek = df_group['review_body'].str.extract()
		#print(peek)
		avg_pol = df_group['polarity'].mean()
		stv_pol = df_group['polarity'].std()
		avg_sub = df_group['subjectivity'].mean()
		stv_sub = df_group['subjectivity'].std()
		print(avg_pol)
		print(stv_pol)
		print(avg_sub)
		print(stv_sub)
		#for single in temp:
	#	a = single['review_body']
	#	print(a)

#for entity in collection:
#	  sentiment_calc_polarity(entity['review_body'])
		#print('{0} {1}'.format(car['review_headline'], 
		#	car['review_body'])) 
