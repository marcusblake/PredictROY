from __future__ import print_function
from scraping import ROYData
from scraping import currentRookieData

#Import Libraries
import matplotlib
#Sets environment for Matplotlib
matplotlib.use('TkAgg')
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn import metrics
# from tensorflow.python.data import Dataset






"""
	arg(s):

	return:
"""
def get_feature_columns(training_features):

	temp = set()
	for column in training_features:
		temp.add(tf.feature_column.numeric_column(column))

	return temp





"""
	arg(s):

	return:
"""
def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):

	#Create numpy array of python dictionaries
	features = {key:np.array(value) for key,value in dict(features).items()}

	dataset = tf.data.Dataset.from_tensor_slices((features, targets)) ##failing
	
	dataset = dataset.batch(batch_size).repeat(num_epochs)
	
	if shuffle:
		dataset.shuffle(500)

	features, labels = dataset.make_one_shot_iterator().get_next()

	return features, labels





"""
	arg(s):

	return:
"""
def train_model(learning_rate, steps, batch_size, training_features, training_targets, validation_features, validation_targets):

	periods = 10

	steps_per_period = steps / periods

	# Defining Gradient Descent Optimizer to increase runtime efficiency of algorithm.
	gd_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	# Applying gradient clipping to prevent divergence during gradient descent
	gd_optimizer = tf.contrib.estimator.clip_gradients_by_norm(gd_optimizer, 5.0)


	# Create linear classifier
	linear_classifier = tf.estimator.LinearClassifier(feature_columns=get_feature_columns(training_features), n_classes=2, optimizer=gd_optimizer)


	###### DEFINE INPUT  FUNCTIONS FOR LINEAR CLASSIFIER ######
	#Want to shuffle training data and use mini-batching for training
	training_input_fn = lambda: input_function(training_features, training_targets['ROY'], batch_size=batch_size)
	predict_training_input_fn = lambda: input_function(training_features, training_targets['ROY'], shuffle=False, num_epochs=1)
	predict_validation_fn = lambda: input_function(validation_features, validation_targets['ROY'], shuffle=False, num_epochs=1)



	training_log_loss = []
	validation_log_loss = []



	print("Starting training:\n")
	for period in range(10):
		print("Training...")
		linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)


		training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
		training_predictions = np.array([item['probabilities'] for item in training_predictions])

		validation_predictions = linear_classifier.predict(input_fn=predict_validation_fn)
		validation_predictions = np.array([item['probabilities'] for item in validation_predictions])

		training_loss = metrics.log_loss(training_targets, training_predictions)
		validation_loss = metrics.log_loss(validation_targets, validation_predictions)



		#Add loss to array to later graph with Matplotlib
		training_log_loss.append(training_loss)
		validation_log_loss.append(validation_loss)

		print("Training loss for period %d: %f" % (period, training_loss))
		print("Validation loss for period %d: %f" % (period, validation_loss))
		
	print("Training done")


	plt.ylabel('Log Loss')
	plt.xlabel('Period')
	plt.title("Log Loss vs. Periods")
	plt.tight_layout()
	plt.plot(training_log_loss, label='Training', c='g')
	plt.plot(validation_log_loss, label='Validation', c='b')
	plt.legend()
	plt.show()

	return linear_classifier



"""
	arg(s):

	return:
"""
def feature_processing(nba_historical_data):

	selected_features = nba_historical_data[
	['MIN',
	'PTS',
	'FGM',
	'FGA',
	'FGPercent',
	'FTM',
	'FTA',
	'FTPrecent',
	'OREB',
	'DREB',
	'REB',
	'AST',
	'STL',
	'BLK',
	'EFF',
	'TOV'
	]]

	
	selected_features.loc[:, 'EFF'] = ( selected_features['FGM'] * 89.910 ) + \
									( selected_features['STL'] * 53.897 ) + \
									( selected_features['FTM'] * 46.845 ) + \
									( selected_features['BLK'] * 39.190 ) + \
									( selected_features['OREB'] * 39.190 ) + \
									( selected_features['AST'] * 34.677 ) + \
									( selected_features['DREB'] * 14.707 ) - \
									( (selected_features['FTA'] - selected_features['FTM']) * 20.091 ) - \
									( (selected_features['FGA'] - selected_features['FGM']) * 39.190 ) - \
									( selected_features['TOV'] * 38.973 )
	

	selected_features['EFF'] = selected_features['EFF'] / selected_features['MIN']

	return selected_features





def target_processing(nba_historical_data):

	return nba_historical_data[['ROY']]





"""
	This method adds a boolean column that signfies whether a particular player has won 
	the Rookie of the Year award

	arg(s): DataFrame that contains historical data of all NBA rookies from 1980-2016

	return: Returns updated DataFrame that has ROY column
"""

def addROYBooleanColumn(nba_historical_data):

	#These lines get the names of previous winners
	previous_winners = ROYData()
	previous_winners_set = set()

	for i in range( len(previous_winners) ):
		previous_winners_set.add(previous_winners[i])


	#contains the data of each player
	data = []


	for i in range(len(nba_historical_data['Name'])):
		if nba_historical_data['Name'][i].encode('ascii') in previous_winners_set:
			data.append(1)
		else:
			data.append(0)

	nba_historical_data = nba_historical_data.assign(ROY=data)

	return nba_historical_data





"""
	This is the main driver of this model

	args: None

	return: None
"""

def main():

	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x', encoding='utf-8')

	print('Pre-processing data')
	# Gets rid of 2016 season data since the data in this set was collected when season wasn't complete and 
	# statistics collected could provide false insight into ROY prediction
	nba_historical_data = nba_historical_data.iloc[32:].reset_index()
	nba_historical_data = nba_historical_data.drop(columns=['3P Made', '3PA', '3P%'])
	nba_historical_data = nba_historical_data.rename(index=str, columns={'FG%': 'FGPercent', 'FT%': 'FTPrecent'})
	nba_historical_data = nba_historical_data.fillna(0)

	nba_historical_data = addROYBooleanColumn(nba_historical_data)

	#Randomize data
	nba_historical_data = nba_historical_data.reindex(np.random.permutation(nba_historical_data.index))


	#Splitting training data, validation data, testing data approximately 50/25/25
	training_features = feature_processing(nba_historical_data.iloc[:754])
	training_targets = target_processing(nba_historical_data.iloc[:754])

	validation_features = feature_processing(nba_historical_data.iloc[754:1131])
	validation_targets = target_processing(nba_historical_data.iloc[754:1131])

	testing_features = feature_processing(nba_historical_data.iloc[1131:])
	testing_targets = target_processing(nba_historical_data.iloc[1131:])
	print('Done pre-processing data')



	logistic_regressor = train_model(
		learning_rate=.0001,  
		steps=100, 
		batch_size=10, 
		training_features=training_features, 
		training_targets=training_targets, 
		validation_features=validation_features, 
		validation_targets=validation_targets
		)




	# sample = nba_historical_data.sample(n=500)

	# ROYS = nba_historical_data.iloc[previous_winners]
	
	# plt.ylabel('Points Per Game')
	# plt.xlabel('Efficiency')

	# plt.scatter(sample['EFF'], sample['PTS'], color='g')
	# plt.scatter(ROYS['EFF'], ROYS['PTS'], color='r')
	# plt.show()





if __name__ == '__main__':
	main() 
