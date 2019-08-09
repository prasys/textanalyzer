#!/usr/bin/python3

from pymongo import MongoClient
from textblob import TextBlob
import pandas as pd
import sys
#from odo import odo
#import dask.dataframe as pd

#star_rating
#vine
#cat_text = 'verified_purchase'
cat_text = sys.argv[1] #category we want to extract
db_name = sys.argv[2] #get the db name for it
#cat_text = 'vine'
#db_name = "amazon_reviews_us_Books_v1_02.tsv" 


def establishMongoDB():
	client = MongoClient('mongodb://localhost:27017/')
	return client


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


#Debug code for args 	
#print ("This is the name of the script: " + sys.argv[0])
#print ("Number of arguments: " +  str(sys.argv))
#print ("The arguments are: " + str(sys.argv))



client = establishMongoDB()
db = client.amazon
col = client['amazon_o'][db_name]
#collection = fetch_db('sample') #STG
collection = fetch_db(db_name)
df = pd.DataFrame(list(collection))
#print(df.head())


#vine
#rev

if cat_text is 'review_date':
	print("HELLO !")
	df['review_date'] = pd.to_datetime(df['review_date'])
	df['year'] = df['review_date'].dt.year
	cat_text = 'year' #set it to year

#df['review_date'] = pd.to_datetime(df['review_date']) #convert time
#df['year'] = df['review_date'].year
grouped = df.groupby(cat_text) #group by star rating for amazon here
#group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean()) std
group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().mean())
print(group1)
group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().std())
print(group1)
group1 = df.groupby(cat_text)['review_body'].apply(lambda x: x.str.split().str.len().sum()) # Calculate the Total Number of Words
print(group1)
print ("Calculating Polarity")
df['polarity'] = df['review_body'].apply(sentiment_calc_polarity)

print ("Calculating Subjectivity")
df['subjectivity'] = df['review_body'].apply(sentiment_calc_subjectivity)
data = df.to_dict(orient='records')

col.insert_many(data)

#odo(df,db[colle])




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
