from __future__ import print_function
from scraping import ROYData
from scraping import currentRookieData
from sklearn import metrics

#Import Libraries
import matplotlib
matplotlib.use('TkAgg') #Sets environment for Matplotlib
import math
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np





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

	# Create numpy array of python dictionaries
	features = {key:np.array(value) for key,value in dict(features).items()}

	dataset = tf.data.Dataset.from_tensor_slices((features, targets))
	
	dataset = dataset.batch(batch_size).repeat(num_epochs)
	
	if shuffle:
		dataset.shuffle(500)

	features, labels = dataset.make_one_shot_iterator().get_next()

	return features, labels





"""
	arg(s):
		learning -> float that we use for gradient descent
		steps -> integer that is the number of steps that we take as we train
		batch_size -> integer that is the size of mini-btaches that we will use
		training_features -> DataFrame that contains the data of the features we will use to train the model
		training_targets -> DataFrame that contains the data of the targets we will use to train the model
		validation_features -> DataFrame that contains the data of the features we will use to validate the model
		validation_targets -> DataFrame that contains the data of the targets we will use to validate the model


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


	
	# Want to shuffle training data and use mini-batching for training
	training_input_fn = lambda: input_function(training_features, training_targets['ROY'], batch_size=batch_size)
	predict_training_input_fn = lambda: input_function(training_features, training_targets['ROY'], shuffle=False, num_epochs=1)
	predict_validation_fn = lambda: input_function(validation_features, validation_targets['ROY'], shuffle=False, num_epochs=1)



	training_log_loss = []
	validation_log_loss = []



	print('Starting training:')
	for period in range(10):
		print('Training...')
		linear_classifier.train(input_fn=training_input_fn, steps=steps_per_period)


		training_predictions = linear_classifier.predict(input_fn=predict_training_input_fn)
		training_predictions = np.array([item['probabilities'] for item in training_predictions])

		validation_predictions = linear_classifier.predict(input_fn=predict_validation_fn)
		validation_predictions = np.array([item['probabilities'] for item in validation_predictions])

		training_loss = metrics.log_loss(training_targets, training_predictions)
		validation_loss = metrics.log_loss(validation_targets, validation_predictions)



		# Add loss to array to later graph with Matplotlib
		training_log_loss.append(training_loss)
		validation_log_loss.append(validation_loss)

		print('Training loss for period %d: %f' % (period+1, training_loss))
		print('Validation loss for period %d: %f' % (period+1, validation_loss))
		
	print('Training done')


	plt.ylabel('Log Loss')
	plt.xlabel('Period')
	plt.title('Log Loss vs. Periods')
	plt.tight_layout()
	plt.plot(training_log_loss, label='Training', c='g')
	plt.plot(validation_log_loss, label='Validation', c='b')
	plt.legend()

	return linear_classifier



"""
	This method selects the features that will be used in our model. It also does a simple
	recalculation of player efficiency rating so that we can have a unifrom metricsthat we can
	use when scraping new data.

	arg(s): DataFrame that contains historical data of all NBA rookies from 1980-2016

	return: DataFrame that only consists of the features that will be used in the ML model
"""
def feature_processing(nba_historical_data):

	selected_features = nba_historical_data.loc[:,
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

	
	selected_features.loc[:,'EFF'] = selected_features.loc[:,'PTS'] + \
									  selected_features.loc[:,'REB'] + \
									  selected_features.loc[:,'AST'] + \
									  selected_features.loc[:,'STL'] + \
									  selected_features.loc[:,'BLK'] - \
									  ( selected_features.loc[:,'FTA'] - selected_features.loc[:,'FTM'] ) - \
									  ( selected_features.loc[:,'FGA'] - selected_features.loc[:,'FGM'] ) - \
									  selected_features.loc[:,'TOV']
	
	# Efficiency per minute
	selected_features.loc[:,'EFF'] = selected_features.loc[:,'EFF'] / selected_features.loc[:,'MIN']

	return selected_features





def target_processing(nba_historical_data):
	return nba_historical_data.loc[:,['ROY']]





"""
	This method adds a boolean column that signfies whether a particular player has won 
	the Rookie of the Year award

	arg(s): DataFrame that contains historical data of all NBA rookies from 1980-2016

	return: Returns updated DataFrame that has ROY column
"""

def addROYBooleanColumn(nba_historical_data):

	# These lines get the names of previous winners
	previous_winners = ROYData()
	previous_winners_set = set()

	for i in range( len(previous_winners) ):
		previous_winners_set.add(previous_winners[i])


	# Contains the data of each player
	data = []


	for i in range(len(nba_historical_data['Name'])):
		if nba_historical_data['Name'][i].encode('ascii') in previous_winners_set:
			data.append(1)
		else:
			data.append(0)

	nba_historical_data = nba_historical_data.assign(ROY=data)

	return nba_historical_data





"""
	This method does data preprocessing to prepare the data to be input to the
	model.

	arg(s): DataFrame which contains statistics of rookies from 1980-2015

	return: Returns updated DataFrame
"""
def preprocess_data(nba_historical_data):

	

	print('Pre-processing data')


	# Gets rid of 2016 season data since the data in this set was collected when season wasn't complete and 
	# statistics collected could provide false insight into ROY prediction
	nba_historical_data = nba_historical_data.iloc[32:].reset_index()


	# Drop 3P Statistics because this may overfit data due to the fact that the 3PT line
	# was introduced in 1979 and not many players in early 80s shot 3 Pointers
	nba_historical_data = nba_historical_data.drop(columns=['3P Made', '3PA', '3P%'])


	nba_historical_data = nba_historical_data.rename(index=str, columns={'FG%': 'FGPercent', 'FT%': 'FTPrecent'})
	nba_historical_data = nba_historical_data.fillna(0)
	nba_historical_data = addROYBooleanColumn(nba_historical_data)

	# Randomize data
	nba_historical_data = nba_historical_data.reindex(np.random.permutation(nba_historical_data.index))



	print('Done pre-processing data')



	return nba_historical_data





"""
	This method splits data into a training set, valdiation set, and testing set before
	calling the train_model() function. It then uses the regressor returned from train_model()
	to predict on the testing set in order to gauge the accuracy of the model. A plot should also
	appear showing Log Loss vs Periods

	arg(s): DataFrame which contains statistics of rookies from 1980-2015

	return: Done
"""
def train(nba_historical_data):


	training_set = nba_historical_data.iloc[:754]
	validation_set = nba_historical_data.iloc[754:1131]
	testing_set = nba_historical_data.iloc[1131:]

	# Splitting training data, validation data, testing data approximately 50/25/25
	training_features = feature_processing(training_set)
	training_targets = target_processing(training_set)
	
	validation_features = feature_processing(validation_set)
	validation_targets = target_processing(validation_set)

	testing_features = feature_processing(testing_set)
	testing_targets = target_processing(testing_set)


	logistic_regressor = train_model(
		learning_rate=.0001,  
		steps=100, 
		batch_size=10, 
		training_features=training_features, 
		training_targets=training_targets, 
		validation_features=validation_features, 
		validation_targets=validation_targets
	)

	predict_testing_fn = lambda: input_function(testing_features, testing_targets['ROY'], shuffle=False, num_epochs=1)
	testing_predictions = logistic_regressor.predict(input_fn=predict_testing_fn)
	testing_predictions = np.array([item['probabilities'] for item in testing_predictions])
	testing_log_loss = metrics.log_loss(testing_targets, testing_predictions)

	print('Loss on testing is: %f' % testing_log_loss)


	print('Generating plot')
	plt.show()