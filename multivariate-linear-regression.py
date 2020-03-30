""" multivariate-linear-regression.py
    Algoritmo que implementa el Gradiente Descendiente 
    para un algoritmo de regresion lineal multivariado

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Thu 30 March, 2020
"""

def main():
	import numpy as np
	import utilityfunctions as uf

	x_train, y_train, mean, std, features = uf.load_data('training-data-multivariate.csv')

	learning_rate = 0.5
	stopping_criteria = 0.01

	w = (np.array([[0.0]*features]).T)

	w = uf.gradient_descent(x_train, y_train, w, stopping_criteria, learning_rate)

	uf.show_w(w)

	x_testing = uf.load_predict('predict_last_mile.csv',mean,std)

	price = uf.predict_last_mile(x_testing,w)

	uf.show_last(price)
	
main()