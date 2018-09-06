import training as tr
import pandas as pd
from scraping import currentRookieData



def main():

	print('Pulling data from https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x ....')
	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x', encoding='utf-8')
	print('Done pulling data.')

	currentRookieData()

	current_data = pd.read_csv('currentRookieData.csv')


	nba_historical_data = tr.preprocess_training_data(nba_historical_data)
	current_data, players = tr.preprocess_current(current_data)

	tr.train_and_predict(nba_historical_data, current_data, players)


if __name__ == '__main__':
	main()