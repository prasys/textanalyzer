import autosklearn.classification
import sklearn.metrics
import pandas as pd
import scipy
import numpy as np

def loadTheFiles(trainX,trainY,testX,testY):
	trainX = np.load(trainX,allow_pickle=True) # load train file
	trainY = np.load(trainY,allow_pickle=True)
	testX = np.load(testX,allow_pickle=True)
	testY = np.load(testY,allow_pickle=True)
	return (trainX,trainY,testX,testY)


def doThePrediction(testX,trainX,trainY):
	automl = autosklearn.classification.autoSklearnClassifier()
	automl.fit(trainX,trainY)
	predictedValue = automl.predict(testX)
	return predictedValue

def matrixScoreCalculations(predictedX,actualX):
	tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true=actualX, y_pred=predictedX).ravel()
	print(tp,fp,fn,tn)
	print("Accuracy score", sklearn.metrics.accuracy_score(predictedX, actualX))
	print("Recall score", sklearn.metrics.recall_score(predictedX, actualX))
	print("Percision score", sklearn.metrics.precision_score(predictedX, actualX))
	print("Positive Score",(tp/(tp+fn)))
	print("Negative Score",(tn/(tn+fp)))

	# TO-DO ADD A METHOD TO SAVE IT TO FILE

if __name__ == '__main__':
	print ('Number of arguments:', len(sys.argv), 'arguments.')
	print ('Argument List:', str(sys.argv))
	if len(sys.argv[1]) > 1:
		trainX = sys.argv[1]
		trainY = sys.argv[2]
		testX = sys.argv[3]
		testY = sys.argv[4]
		trainX , trainY , testX , testY = loadTheFiles(trainX,trainY,testX,testY)
		predictedX = doThePrediction(testX,trainX,trainY)
		matrixScoreCalculations(predictedX,testY)
	else:
		print("UNDEFINED FILE NAME , PLEASE DEFINE FILE NAME TO BE PROCESSED")
		exit() #force exit
