import pandas as pd
import fasttext
from io import StringIO
import csv
from sklearn.metrics import f1_score , recall_score , accuracy_score , precision_score , jaccard_score , balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split , KFold , LeaveOneOut , LeavePOut , ShuffleSplit , StratifiedKFold , GridSearchCV

colNames = ['Comment', 'Prediction']
LABEL = 'Prediction'
STATE = 21 # random state for reproducability 
TEXT = 'Comment'
TRAINTXT = 'train.txt'
TESTTXT = 'test.txt'
noSplits = 2
PREDICTED = 'Predicted'


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


def getLabelValue(wolo):

	string = wolo[0]

	strings = string.split()
	for s in strings:
		if s.isdigit():
			return s

# This method gives 
def preProcessToFastTXTFormat(dataFrame,label,text,outputName,col=['Comment','Prediction']):
	"""
    This method takes the dataframe and outputs it to a format which can be
    understood by fast-text to do the classification. It maes it easier for people who want to use fast-text to do
    classification
    Input : Dataframe , Label (Truth Data) , Text , Col Names
    Output : a Text File that is friendly
    """
	newDF = dataFrame[col]
	newDF[LABEL] = newDF[LABEL].astype(str) #force as str
	newDF[label]=['__label__'+ s for s in newDF[label]] #add the label for it to work
	newDF[TEXT]= newDF[TEXT].replace('\n',' ', regex=True).replace('\t',' ', regex=True)
	print("OUTPUT TO FILE")
	newDF.to_csv(outputName, index=False, sep=' ', header=False, quoting=csv.QUOTE_NONE, quotechar="", escapechar=" ")




def splitTextFileAndValidate(df,actual,labels):
	Val = StratifiedKFold(n_splits=noSplits, random_state=STATE, shuffle=True) # DO OUR FOLD here , maybe add the iteration
	for train_index,test_index in Val.split(actual,labels):
		train_df = df.iloc[train_index]
		test_df = df.iloc[test_index]
		preProcessToFastTXTFormat(train_df,LABEL,TEXT,TRAINTXT) #it will do processing here for us
		model = fasttext.train_supervised(TRAINTXT)
		test_df['Predicted'] = test_df[TEXT].apply(model.predict) #run our prediction model
		test_df[PREDICTED] = test_df[PREDICTED].apply(getLabelValue) #run our prediction model
		print(test_df.head())


if __name__ == '__main__':
	df = read_csv('train_classifier.csv')
	print(df.head())
	actual = df[TEXT].tolist()
	labels = df[LABEL].tolist()
	splitTextFileAndValidate(df,actual,labels)
