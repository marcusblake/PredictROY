import math


from scrapping import ROYData
from scrapping import currentRookieData


#Import Libraries
import pandas as pd

def main():
	rookies_of_year = pd.Series( ROYData() )
	print rookies_of_year


if __name__ == "__main__":
	main() 
