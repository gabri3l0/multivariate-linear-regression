#!/usr/bin/python3.7
""" multivariate-linear-regression.py
    Algoritmo que implementa el Gradiente Descendiente 
    para un algoritmo de regresion lineal multivariado

    Author: Gabriel Aldahir Lopez Soto
    Email: gabriel.lopez@gmail.com
    Institution: Universidad de Monterrey
    First created: Thu 30 March, 2020
"""

def main():
	"""
	Aqui se manda llamar el archivo para leer los datos del CSV, asi mismo
	se obtienen las x, y de entrenamiento, el promedio de las x aplicando 
	escalamiento de caracteristicas y la desviacion estandar para despues
	obtener los parametros w y con base eso usar los datos de prueba y
	predecir las y o el costo de la ultima milla

	Datos de entrada:
	Nada

	Datos de salida:
	Nada
	"""
	# Importa las librerias estandard y la libreria utilityfunctions
	import numpy as np
	import utilityfunctions as uf

	# Metodo para obtener el x,y de entrenamiento, promedio, desviacion estandar, y caracteristicas
	x_train, y_train, mean, std, features = uf.load_data('training-data-multivariate.csv')

	# Se inicializa los hiperparametros
	learning_rate = 0.0005
	stopping_criteria = 0.01

	# Inicializa w
	w = (np.array([[0.0]*features]).T)

	# Metodo para obtener el gradiente descendiente
	w = uf.gradient_descent(x_train, y_train, w, stopping_criteria, learning_rate)

	# Metodo para imprimir las w optimas
	uf.show_w(w)

	# Metodo para cargar los datos de pruebas
	x_testing = uf.load_predict('predict_last_mile.csv',mean,std)

	# Metodo para obtener el costo de la ultima milla de los datos de prueba
	price = uf.predict_last_mile(x_testing,w)

	# Metodo para imprimir el costo de la ultima milla de los datos de prueba
	uf.show_last(price)

main()