import time
import datetime
import torch

def test_model(model, data, criterion, optimizer):

	model.eval() # Set model to test mode

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward
	outputs = model(data['x'])

	# loss
	loss = criterion(outputs, data['y'])

	return outputs, loss.data

def train_model(model, data, criterion, optimizer):

	model.train()  # Set model to training mode

	# zero the parameter gradients
	optimizer.zero_grad()

	# forward
	outputs = model(data['x'])

	# loss
	loss = criterion(outputs, data['y'])
# 	print('loss ', loss)

	# backward + optimize
	loss.backward()
	optimizer.step()

	return outputs, loss.data