from bs4 import BeautifulSoup
import requests

req = requests.get('https://www.basketball-reference.com/leagues/NBA_2018_rookies.html')
soup = BeautifulSoup(req.text, 'html.parser')



def run():

	#Getting necessary labels for the data
	headers = soup.findAll('tr', class_='thead')
	part1 = headers[0]
	part2 = headers[1]

	#get tables
	tables = soup.findAll('tr', class_='full_table')
	
	print tables
    


if __name__=="__main__":
	run()
