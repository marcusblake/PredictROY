# Predict NBA Rookie of the Year using Machine Learning
This is a personal project which involves utilizing a simple machine learning algorithm in order to predict the NBA Rookie of The Year based on statistics of all rookies in any given year.

#### Setting Up Environment
It is highly recommended to use a virtual environment to avoid conflict with existing Python installations. You can manually setup a virtual environment or run the following commands in your terminal:
```
chmod +x setup.sh
./setup.sh
source env/bin/activate
```

#### Installation
The file script.sh installs all of the necessary libraries and dependencies in order to scrap the necessary data to use in the algorithm as well as the libraries needed to perform the machine learning algorithm. The required libraries are listed in requirements.txt. You can install the necessary libraries with by running the following command in your terminal (it is recommended that you setup a virtual environment before installing these libraries):
```
chmod +x install.sh
./install.sh
```
#### Deactivate Environment
If you activate a virtual environment, you must deactivate it once you are done. Run the following command in your terminal to deactivate the virtual environment:

`deactivate`

#### Description of files
* **scraping.py**
This file does web scraping to get statistical data from the 2017-2018 NBA Season which is the year we wish to predict. This file also does web scraping to get the Rookie of the Year winners.

* **training.py**
This file contains the body for the machine learning algorithm which uses logistic regression in order to classify a player as the rookie of the year. You can run the file by doing the following.
```
python training.py
```

#### Project Dependencies
The following list contains the external libraries that I use in this project. These libraries can all be installed using the installation instructions above.
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
* [Matplotlib](https://matplotlib.org/api/index.html)
* [Tensorflow](https://www.tensorflow.org/)
* [Scikit-learn](http://scikit-learn.org/stable/index.html)
* [NumPy](http://www.numpy.org/)
