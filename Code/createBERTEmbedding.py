import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning) # to surpress future warnings
import pandas as pd
import sys
import numpy as numpy
import math
import gensim
from pprint import pprint
from string import ascii_lowercase
#import Use_NN as nn
import re
import string
from bert_embedding import BertEmbedding
import mxnet as mx



# Pandas Method to read our CSV to make it easier
def read_csv(filepath):
    #parseDate = ['review_date']
    #dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    #colName = ['customer_id','product_category', 'review_id', 'star_rating','helpful_votes','total_votes','vine','verified_purchase','review_body','review_date']
    #df_chunk = pd.read_csv(filepath, sep='\t', header=0, chunksize=500000, error_bad_lines=False,parse_dates=parseDate, dtype=column_dtypes, usecols=colName, date_parser=dateparse)
    df_chunk = pd.read_csv(filepath, sep=',', header=0,encoding = "ISO-8859-1")
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



#Main Method
if __name__ == '__main__':
	df = read_csv('train_22.csv')
	CommentList = df['Comment'].tolist()
	print(mx.context.num_gpus())
	print("Load GPU")
	ctx = mx.gpu(0)
	print("Start Embeddings")
	bert_embedding = BertEmbedding(ctx=ctx)
	results = bert_embedding(CommentList)
	print(len(results))
	toNumpy = []

	#for result in results:
		#print(result[1])
		#toNumpy.append(result[1])

	#output = numpy.asarray(toNumpy) # convert it to array
	#print(output.shape)


 