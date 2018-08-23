import training as tr
import pandas as pd



def main():

	print('Pulling data from https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x ....')
	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x', encoding='utf-8')
	print('Done pulling data.')


	nba_historical_data = tr.preprocess_data(nba_historical_data)
	tr.train(nba_historical_data)


if __name__ == '__main__':
	main()