from __future__ import print_function
from scraping import ROYData
from scraping import currentRookieData

#Import Libraries
import matplotlib
matplotlib.use('TkAgg') #Sets environment for Matplotlib
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from random import randint
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


"""
	This function returns a matrix of features which will be used in matrix multiplication

	arg(s):
		features -> DataFrame which contains features to be used in model

	return: numpy matrix
"""
def input_function(features):

	arr = []

	# Create numpy array of python dictionaries
	for key, value in dict(features).items():
		arr.append(value)
		
	features = np.array(arr).T

	return features





"""
	This function is the main training method for the model. It utilizes a random forest classifier to make predictions
	on the probability of an individual winning rookie of the year

	arg(s):
		training_features -> DataFrame that contains the data of the features we will use to train the model
		training_targets -> DataFrame that contains the data of the targets we will use to train the model
		validation_features -> DataFrame that contains the data of the features we will use to validate the model
		validation_targets -> DataFrame that contains the data of the targets we will use to validate the model
		current_data -> DataFrame which contains statistics of rookies
		players -> numpy array containing names of players


	return: None
"""
def train_model(training_features, training_targets, validation_features, validation_targets, current_data, players):

	#Number of features that are used
	n_features = len(training_features.columns)

	training_features = input_function(training_features)
	training_targets = np.array(training_targets['ROY'])

	validation_features = input_function(validation_features)
	validation_targets = np.array(validation_targets['ROY'])



	#to be used later for plotting
	ppg = current_data['PTS']
	eff = current_data['EFF']



	current_data = input_function(current_data)


	# Defines classifier
	rfc = RandomForestClassifier()

	# trains model
	rfc.fit(training_features, training_targets)



	print('Classification report on training data')
	predictions = rfc.predict(training_features)
	print(classification_report(training_targets,predictions, target_names=['Class 0', 'Class 1']))
	training_accuracy = accuracy_score(training_targets, predictions)



	print('Classification report on validation data')
	predictions = rfc.predict(validation_features)
	print(classification_report(validation_targets,predictions, target_names=['Class 0', 'Class 1']))
	validation_accuracy = accuracy_score(validation_targets, predictions)




	print('Training accuracy: ', (training_accuracy*100).round(2) )
	print('Validation accuracy: ', (validation_accuracy*100).round(2) )


	# predict on new data
	results = rfc.predict(current_data)
		
	print('Rookie of the Year:')

	potential_winners = []
	for player, result in zip(players, results):
		#Decision threshold
		if result==1:
			potential_winners.append(player)



	length = len(potential_winners)
	winner = None

	if(length == 0):
		### If it can't predict a definitive winner, it will choose the person with the highest probability
		results = rfc.predict_proba(current_data)
		max_prob = 0
		for player, result in zip(players, results):
			if result[1] > max_prob:
				max_prob = result[1]
				winner = player
		print(winner)
	### If it predicts that multiple people are likely to win, it will choose at random
	elif length > 1:
		winner = potential_winners[randint(0, length - 1)]
		print(winner)
	else:
		winner = potential_winners[0]
		print(winner)


	for index in range(len(current_data)):
		if players[index] == winner:
			winner_ppg = ppg.iat[index]
			winner_eff = eff.iat[index]
			break



	### Figure showing PPG vs. EFF to make some sense as to why the model predicted the particular person ###
	plt.ylabel('PPG')
	plt.xlabel('EFF')
	plt.title('Points per game vs. Efficiency')
	plt.tight_layout()
	plt.scatter(eff, ppg, label='Other players', c='b')
	plt.scatter(winner_eff, winner_ppg, label='Rookie of the Year', c='r')
	plt.legend()






"""
	This method selects the features that will be used in our model. It also does a simple
	recalculation of player efficiency rating so that we can have a unifrom metricsthat we can
	use when scraping new data.

	arg(s): 
		nba_historcial_data -> DataFrame that contains data of NBA rookies

	return: DataFrame that only consists of the features that will be used in the ML model
"""
def feature_processing(nba_historical_data):

	selected_features = nba_historical_data.loc[:,
	['GP',
	'MIN',
	'PTS',
	'FGPercent',
	'FTPercent',
	'REB',
	'AST',
	'STL',
	'BLK',
	'EFF',
	'TOV'
	]]

	
	selected_features.loc[:,'EFF'] = nba_historical_data.loc[:,'PTS'] + \
									  nba_historical_data.loc[:,'REB'] + \
									  nba_historical_data.loc[:,'AST'] + \
									  nba_historical_data.loc[:,'STL'] + \
									  nba_historical_data.loc[:,'BLK'] - \
									  ( nba_historical_data.loc[:,'FTA'] - nba_historical_data.loc[:,'FTM'] ) - \
									  ( nba_historical_data.loc[:,'FGA'] - nba_historical_data.loc[:,'FGM'] ) - \
									  nba_historical_data.loc[:,'TOV']

	return selected_features




"""
	This method gets the ROY column in the nba_historical_data DataFrame

	arg(s): 
		nba_historical_data -> DataFrame that contains historical data of all NBA rookies from 1980-2016

	return: Returns ROY column as a series
"""
def target_processing(nba_historical_data):
	return nba_historical_data.loc[:,['ROY']]





"""
	This method adds a boolean column that signfies whether a particular player has won 
	the Rookie of the Year award

	arg(s): 
		nba_historical -> DataFrame that contains historical data of all NBA rookies from 1980-2016

	return: Returns updated DataFrame that has ROY column
"""

def addROYBooleanColumn(nba_historical_data):

	# These lines get the names of previous winners
	previous_winners = ROYData()
	previous_winners_set = set()

	for prev_winner in previous_winners:
		previous_winners_set.add(prev_winner)


	# Contains the data of each player
	data = []


	for name in nba_historical_data['Name']:
		if name.encode('ascii') in previous_winners_set:
			data.append(1)
		else:
			data.append(0)

	nba_historical_data = nba_historical_data.assign(ROY=data)

	return nba_historical_data





"""
	This method does data preprocessing to prepare the data to be input to the
	model.

	arg(s): 
		nba_historical_data -> DataFrame which contains statistics of rookies from 1980-2015

	return: Returns updated DataFrame
"""
def preprocess_training_data(nba_historical_data):

	

	print('Pre-processing data')



	# Gets rid of 2016 season data since the data in this set was collected when season wasn't complete and 
	# statistics collected could provide false insight into ROY prediction
	nba_historical_data = nba_historical_data.iloc[32:]

	nba_historical_data = nba_historical_data[nba_historical_data.GP >= 41]

	nba_historical_data = nba_historical_data.reset_index()


	# Drop 3P Statistics because this may overfit data due to the fact that the 3PT line
	# was introduced in 1979 and not many players in early 80s shot 3 Pointers
	nba_historical_data = nba_historical_data.drop(columns=['3P Made', '3PA', '3P%'])


	nba_historical_data = nba_historical_data.rename(index=str, columns={'FG%': 'FGPercent', 'FT%': 'FTPercent'})
	nba_historical_data = nba_historical_data.fillna(0)
	nba_historical_data = addROYBooleanColumn(nba_historical_data)

	# Randomize data
	nba_historical_data = nba_historical_data.reindex(np.random.permutation(nba_historical_data.index))



	print('Done pre-processing data')



	return nba_historical_data




"""
	This method does data preprocessing to prepare the data to be input to the
	model.

	arg(s): 
		current_data -> DataFrame which contains statistics of rookies

	return: Returns updated DataFrame
"""

def preprocess_current(current_data):

	current_data = current_data[current_data.G >= 41]
	current_data = current_data.reset_index()


	columns=['3P', '3P.1', '3PA', 'Yrs', 'Age', 'Unnamed: 0', 'FG%', 'FT%', 'MP.1', 'PS/G','TRB.1', 'AST.1', 'PF', 'ORB']
	current_data = current_data.drop(columns=columns).rename(columns={'FG': 'FGM', 'FT': 'FTM', 'TRB': 'REB', 'MP': 'MIN', 'G': 'GP'})

	FGPercent = (current_data.loc[:,'FGM'] / current_data.loc[:,'FGA']).round(3) * 100
	FTPercent = (current_data.loc[:,'FTM'] / current_data.loc[:,'FTA']).round(3) * 100
	

	current_data = current_data.assign(FGPercent=FGPercent, FTPercent=FTPercent)

	current_data[['MIN','FGM','FGA','FTM','FTA','REB','AST','STL','BLK','TOV','PTS']] = \
	current_data[['MIN','FGM','FGA','FTM','FTA','REB','AST','STL','BLK','TOV','PTS']].div(current_data.GP, axis=0).round(1)

	return feature_processing(current_data), np.array(current_data['Player'])





"""
	This method splits data into a training set and valdiation set and calls
	the main driver function to train the model

	arg(s): 
		nba_historical_data -> DataFrame which contains statistics of rookies from 1980-2015
		current_data -> DataFrame containing data on current rookies
		players -> numpy array containing names of players


	return: None
"""
def train_and_predict(nba_historical_data, current_data, players):


	training_set = nba_historical_data.iloc[:1131]
	validation_set = nba_historical_data.iloc[1131:]

	# Splitting training data, validation data, testing data approximately 75/25
	training_features = feature_processing(training_set)
	training_targets = target_processing(training_set)
	
	validation_features = feature_processing(validation_set)
	validation_targets = target_processing(validation_set)

	train_model( 
		training_features=training_features, 
		training_targets=training_targets, 
		validation_features=validation_features, 
		validation_targets=validation_targets,
		current_data=current_data,
		players=players
	)

"""
	This method drives the entire process of pulling necessary data, processing the data, and calling the training methods

	arg(s): None

	return: None
"""

def main():

	print('Pulling data from https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x ....')
	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x', encoding='utf-8')
	print('Done pulling data.')

	currentRookieData()

	current_data = pd.read_csv('currentRookieData.csv')


	nba_historical_data = preprocess_training_data(nba_historical_data)
	current_data, players = preprocess_current(current_data)

	train_and_predict(nba_historical_data, current_data, players)

	print('Generating plot')
	plt.show()




if __name__=='__main__':
	main()