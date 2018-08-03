from bs4 import BeautifulSoup
import requests


#
# Gets statistics for Rookies from the 2017-18 NBA Season
#
def currentRookieData():
	
	#Website that contains statistics for current rookies
	req = requests.get('https://www.basketball-reference.com/leagues/NBA_2018_rookies.html')
	soup = BeautifulSoup(req.text, 'html.parser')


	
	dataset = []
	data = soup.findAll('tr', class_='thead')
	
	#get tables
	tables = soup.findAll('tr', class_='full_table')
	
	for row in tables:
		statistics = []
		playerStats = row.findAll('td')
		for stat in playerStats:
			if stat.text != '':
				#ASCII encoding of string
				statistics.append(stat.text.encode('ascii'))
		dataset.append(statistics)
	
	return dataset

#
# Collects the winners of the rookie of the year award from 1980-current
#
def ROYData():

	req = requests.get('http://www.espn.com/nba/history/awards/_/id/35')
	soup = BeautifulSoup(req.text, 'html.parser')

	dataset = []

	table = soup.find('table', class_='tablehead').findAll('tr', class_='oddrow')	
	
	for row in table:
		rookie_of_year = row.findAll('td')
		#Want Rookie of the Year until 1980 since training dataset is from years 1980-2016
		if rookie_of_year[0].text.encode('ascii') == '1979':
			break

		dataset.append(rookie_of_year[1].text.encode('ascii'))
		
	return dataset	
