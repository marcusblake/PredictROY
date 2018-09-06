from __future__ import print_function
from scraping import ROYData

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
def input_function(features):

	arr = []

	
	# Create numpy array of python dictionaries
	for key, value in dict(features).items():
		arr.append(value)
		

	features = np.array(arr).T

	return features





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
def train_model(learning_rate, iteration_num, batch_size, training_features, training_targets, validation_features, validation_targets, current_data, players):

	training_features = input_function(training_features)
	training_targets = np.array(training_targets['ROY'])

	validation_features = input_function(validation_features)
	validation_targets = np.array(validation_targets['ROY'])


	current_data = input_function(current_data)

	ratio = 38.0 / (38.0 + 1468.0)
	ratio = 1 - ratio

	
	#Number of features that are used
	n_features = 16

	weights_shape = (n_features, 1)
	bias_shape = (1, 1)
	
	W = tf.Variable(initial_value=tf.random_normal(weights_shape), dtype=tf.float32, name='weights')

	X = tf.placeholder(dtype=tf.float32, name='features', shape=[None, n_features])

	b = tf.Variable(initial_value=tf.random_normal(bias_shape), dtype=tf.float32, name='bias')

	y_pred = tf.matmul(X, W) + b

	y_true = tf.placeholder(dtype=tf.float32, name='labels', shape=[None, 1])

	loss_function = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=y_true, logits=y_pred, pos_weight=ratio))

	# Defining Gradient Descent Optimizer to increase runtime efficiency of algorithm.
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_function)


	prediction = tf.round(tf.sigmoid(y_pred))
	correct = tf.cast(tf.equal(prediction, y_true), dtype=tf.float32)
	accuracy = tf.reduce_mean(correct)


	training_loss = []
	training_accuracy = []
	validation_accuracy = []

	results = None


	with tf.Session() as sess:

		sess.run(tf.global_variables_initializer())

		print('Starting training:')
		for iteration in range(iteration_num):

			batch_index = np.random.choice(len(training_features), size=batch_size)
			batch_train_X = training_features[batch_index]
			batch_train_y = np.matrix(training_targets[batch_index]).T



			sess.run(optimizer, feed_dict={X: batch_train_X, y_true: batch_train_y})
		

			train_acc = sess.run(accuracy, feed_dict={X: training_features, y_true: np.matrix(training_targets).T})
			training_accuracy.append(train_acc)

			test_acc = sess.run(accuracy, feed_dict={X: validation_features, y_true: np.matrix(validation_targets).T})
			validation_accuracy.append(test_acc)


			if (iteration + 1) % 200 == 0:
				train_loss = sess.run(loss_function, feed_dict={X: batch_train_X, y_true: batch_train_y})
				training_loss.append(train_loss)

				print('Training Loss at iteration %d: %f' % (iteration + 1, train_loss))
				print('Training accuracy at iteration %d: %f' % (iteration + 1, train_acc))
				print('Validation accuracy at iteration %d: %f' % (iteration + 1, test_acc))

			
		print('Training done')

		results = sess.run(prediction, feed_dict={X: current_data})

		print(zip(players, results))

		exit()

		sess.close()

	### Figure showing accuracy vs iterations ###
	plt.figure(1)
	plt.ylabel('Accuracy')
	plt.xlabel('Iterations')
	plt.title('Accuracy vs. Iterations')
	plt.tight_layout()
	plt.plot(training_accuracy, label='Training', c='g')
	plt.plot(validation_accuracy, label='Validation', c='b')
	plt.legend()


	plt.figure(2)






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
	'FTPercent',
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
	# selected_features.loc[:,'EFF'] = selected_features.loc[:,'EFF'] / selected_features.loc[:,'MIN']

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
def preprocess_training_data(nba_historical_data):

	

	print('Pre-processing data')


	# Gets rid of 2016 season data since the data in this set was collected when season wasn't complete and 
	# statistics collected could provide false insight into ROY prediction
	nba_historical_data = nba_historical_data.iloc[32:].reset_index()


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





def preprocess_current(current_data):

	current_data = current_data[current_data.G >= 30]
	current_data = current_data.reset_index()


	columns=['3P', '3P.1', '3PA', 'Yrs', 'Age', 'Unnamed: 0', 'FG%', 'FT%', 'MP.1', 'PS/G','TRB.1', 'AST.1', 'PF']
	current_data = current_data.drop(columns=columns).rename(columns={'FG': 'FGM', 'FT': 'FTM', 'ORB': 'OREB', 'TRB': 'REB', 'MP': 'MIN'})

	DREB = current_data.loc[:,'REB'] - current_data.loc[:,'OREB']
	FGPercent = (current_data.loc[:,'FGM'] / current_data.loc[:,'FGA']).round(3) * 100
	FTPercent = (current_data.loc[:,'FTM'] / current_data.loc[:,'FTA']).round(3) * 100
	

	current_data = current_data.assign(DREB=DREB, FGPercent=FGPercent, FTPercent=FTPercent)

	current_data[['MIN','FGM','FGA','FTM','FTA','OREB', 'DREB', 'REB','AST','STL','BLK','TOV','PTS']] = \
	current_data[['MIN','FGM','FGA','FTM','FTA','OREB', 'DREB', 'REB','AST','STL','BLK','TOV','PTS']].div(current_data.G, axis=0).round(2)

	return feature_processing(current_data), np.array(current_data['Player'])





"""
	This method splits data into a training set, valdiation set, and testing set before
	calling the train_model() function. It then uses the regressor returned from train_model()
	to predict on the testing set in order to gauge the accuracy of the model. A plot should also
	appear showing Log Loss vs Periods

	arg(s): DataFrame which contains statistics of rookies from 1980-2015

	return: Done
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
		learning_rate=.001,  
		iteration_num=1500,
		batch_size=32, 
		training_features=training_features, 
		training_targets=training_targets, 
		validation_features=validation_features, 
		validation_targets=validation_targets,
		current_data=current_data,
		players=players
	)

	print('Generating plot')
	plt.show()