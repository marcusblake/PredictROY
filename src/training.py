import math
import matplotlib
matplotlib.use('TkAgg')
from scraping import ROYData
from scraping import currentRookieData
import matplotlib.pyplot as plt
#Import Libraries
import pandas as pd


previous_winners = ROYData()
previous_winners_set = set()

for i in range( len(previous_winners) ):
	previous_winners_set.add(previous_winners[i])


def contains( x ):
	if x in previous_winners_set:
		return True

	return False


def main():
	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x')
	
	array = []
	winners = []
	for i in range( len(nba_historical_data['Name']) ):
		if contains(nba_historical_data['Name'][i].encode('ascii')):
			array.append(True)
			winners.append(i)
		else:
			array.append(False)

	nba_historical_data['ROY'] = pd.Series( array )
	sample = nba_historical_data.sample(n=500)

	ROYS = nba_historical_data.iloc[winners]
	
	plt.ylabel('Points Per Game')
	plt.xlabel('Efficiency')

	plt.scatter(sample['EFF'], sample['PTS'], color='g')
	plt.scatter(ROYS['EFF'], ROYS['PTS'], color='r')
	plt.show()

if __name__ == "__main__":
	main() 
