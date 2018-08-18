#Import Libraries
import matplotlib
matplotlib.use('TkAgg')

import math
from scraping import ROYData
from scraping import currentRookieData
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np



"""
	args:

	return:
"""
def feature_processing(nba_historical_data):

	selected_features = nba_historical_data[
	["MIN",
	"PTS",
	"FGM",
	"FGA",
	"FG%",
	"3P Made",
	"3PA",
	"3P%",
	"FTM",
	"FTA",
	"FT%",
	"OREB",
	"DREB",
	"REB",
	"AST",
	"STL",
	"BLK",
	"EFF",
	"TOV"
	]]

	
	selected_features['EFF'] = ( selected_features['FGM'] * 89.910 ) + \
								 ( selected_features['STL'] * 53.897 ) + \
								 ( selected_features['3P Made'] * 51.757 ) + \
								 ( selected_features['FTM'] * 46.845 ) + \
								 ( selected_features['BLK'] * 39.190 ) + \
								 ( selected_features['OREB'] * 39.190 ) + \
								 ( selected_features['AST'] * 34.677 ) + \
								 ( selected_features['DREB'] * 14.707 ) - \
								 ( (selected_features['FTA'] - selected_features['FTM']) * 20.091 ) - \
								 ( (selected_features['FGA'] - selected_features['FGM']) * 39.190 ) - \
								 ( selected_features['TOV'] * 38.973 )


	selected_features['EFF']/=selected_features['MIN']

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


	for i in range( len(nba_historical_data['Name']) ):
		if nba_historical_data['Name'][i].encode('ascii') in previous_winners_set:
			data.append(1)
		else:
			data.append(0)

	nba_historical_data['ROY'] = pd.Series( data )

	return nba_historical_data





def main():


	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x')


	# Gets rid of 2016 season data since the data in this set was collected when season wasn't complete and 
	# statistics collected could provide false insight into ROY prediction
	nba_historical_data = nba_historical_data.iloc[32:].reset_index()
	nba_historical_data = addROYBooleanColumn(nba_historical_data)

	#Randomize data
	nba_historical_data = nba_historical_data.reindex(np.random.permutation(nba_historical_data.index))



	#Splitting training data, validation data, testing data approximately 50/25/25
	training_features = feature_processing(nba_historical_data.iloc[:754])
	validation_features = feature_processing(nba_historical_data.iloc[754:1131])
	testing_features = feature_processing(nba_historical_data.iloc[1131:])



	training_targets = target_processing(nba_historical_data.iloc[:754])
	validation_targets = target_processing(nba_historical_data.iloc[754:1131])
	testing_targets = target_processing(nba_historical_data.iloc[1131:])


	print( training_targets )




	# sample = nba_historical_data.sample(n=500)

	# ROYS = nba_historical_data.iloc[previous_winners]
	
	# plt.ylabel('Points Per Game')
	# plt.xlabel('Efficiency')

	# plt.scatter(sample['EFF'], sample['PTS'], color='g')
	# plt.scatter(ROYS['EFF'], ROYS['PTS'], color='r')
	# plt.show()


if __name__ == "__main__":
	main() 
