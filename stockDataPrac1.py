import csv
import numpy
import matplotlib.pyplot as plt 
from sklearn.svm import SVR

dates = []
prices = []

def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)
		for row in csvFileReader:
			dates.append(int(row[0].split('/')[0]))
			prices.append(float(row[1]))
	return
def predict_prices(dates, prices, x):
	dates = np.reshape(dates, (len(dates), 1))

	SVR_lin = SVR(kernel = 'linear', c=1e3)
	SVR_poly = SVR(kernel = 'poly', c=1e3, degree = 2)
	SVR_rbf = SVR(kernel = 'rbf', c=1e3, gamma = 0.1)
	SVR_lin.fit(dates, prices)
	SVR_poly.fit(dates, prices)
	SVR_rbf.fit(dates, prices)

	plt.scatter(dates, prices, color = 'black', label = 'Data')
	plt.plot(dates, SVR_rbf.predict(dates), color = 'red', label = 'RBF model')
	plt.plot(dates, SVR_poly.predict(dates), color = 'green', label = 'Polynomial Model')
	plt.plot(dates, SVR_lin.predict(dates), color = 'blue', label = 'Linear Model')
	plt.xlable('Dates')
	plt.xlable('Prices')
	plt.title('Support Vector Regression')
	plt.legend()
	plt.show()


	return SVR_lin.predict(x)[0], SVR_poly.predict(x)[0], SVR_rbf.predict(x)[0]

get_data('/Users/sridhardumpala/Downloads/HistoricalQuotes.csv')

predictedPrice = predict_price(dates, prices, 29)

print(predict_price)
