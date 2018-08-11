import math


from scraping import ROYData
from scraping import currentRookieData


#Import Libraries
import pandas as pd

def main():
	nba_historical_data = pd.read_excel('https://query.data.world/s/ntr4fv2oniqbrs4b55epcyyia5x66x')
	print nba_historical_data

if __name__ == "__main__":
	main() 
