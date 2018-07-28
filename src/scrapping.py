from bs4 import BeautifulSoup
import requests

req = requests.get('https://www.basketball-reference.com/leagues/NBA_2018_rookies.html')
soup = BeautifulSoup(req.text, 'html.parser')



def run():
    print soup.prettify()


if __name__=="__main__":
    run()
