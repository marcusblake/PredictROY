# Predict NBA Rookie of the Year using Machine Learning
	This is a personal project which involves utilizing a simple machine learning algorithm in order to predict the NBA Rookie of The Year based statistics of all rookies in any given year.

#### Setting Up Environment
	It is highly recommended to use a virtual environment to avoid conflict with existing Python installations. You can manually setup a virtual environment or run the following commands in your terminal:

	1. `chmod +x setup.sh`
	2. `./setup.sh`
	3. `source env/bin/activate`

#### Installation
	The file script.sh installs all of the necessary libraries and dependencies in order to scrap the necessary data to use in the algorithm as well as the libraries needed to perform the machine learning algorithm. The required libraries are listed in requirements.txt. You can install the necessary libraries with by running the following command in your terminal (it is recommended that you setup a virtual environment before installing these libraries):

	1. `chmod +x install.sh`
	2. `./install.sh`

#### Deactivate Environment
	If you activate a virtual environment, you must deactivate it once you are done. Run the following command in your terminal to deactivate the virtual environment:

	`deactivate`

#### Description of files
	* **scraping.py**
		This file does web scraping to get statistical data from the 2017-2018 NBA Season which is the year we wish to predict. This file also does web scraping to get the Rookie of the Year winners.

	* **training.py**
		This file contains the body for the machine learning algorithm which uses logistic regression in order to classify a player as the rookie of the year.