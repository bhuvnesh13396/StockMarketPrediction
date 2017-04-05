

import csv
import numpy as np
from sklearn.model_selection import train_test_split
import bhuvi-neural

close_price = []
output_vector = []


def calculate_output_vector(li):

""" If close_price is greater than yesterday's close_price
than append 1 in output_vector else append -1 ."""

#	with open(filename,'r') as file:
	#	reader = csv.DictReader(file)
	#	reader.next()

	#	for row in reader:
	#		close_price.append(row['Close'])



	for i in range(len-1):
		
		if li[i]-li[i+1]>=0:
			output_vector.append(1)

		else:
			output_vector.append(-1)

	
	return output_vector



def measure_accuracy(result_test_output,result_predicted_output):
	""" Predict the accuray of model in percentage. 
	Based on how much entries in test_output
	and predicted_output are equal. """

	counter=0
	for i in range(len(test_output)):
		if result_test_output[i] == result_predicted_output[i]:
			counter=counter+1

    # Print percentage value
	return ((counter/len(test_output))*100)







# Imported from bhuvi-neural
test_output=testY
predicted_output=stock_prediction(filename)



		
result_test_output = calculate_output_vector(test_output)
result_predicted_output = calculate_output_vector(predicted_output)			



accuray=measure_accuracy(result_test_output , result_predicted_output)

print('Accuray of the mode ::: {}'.format(accuray))




