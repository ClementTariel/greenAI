#!/usr/bin/python

from abc import ABC, abstractmethod

import os
import sys
import warnings
from threading import Thread , Timer
import subprocess as sp
import time

import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import onnx
from onnx import numpy_helper
import onnxruntime as ort

import tensorflow as tf

from tqdm import tqdm
from sklearn.utils import shuffle 

import re  # regex


class PartiallySupportedFileWarning(UserWarning):
	pass

class GPUNotFoundWarning(UserWarning):
	pass

# To prevent a crash when using GPU with tensorflow
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
	warnings.warn("tf.config.list_physical_devices('GPU') is empty, there is no GPU available for tensorflow. ",GPUNotFoundWarning)


def volume(shape):
	"""
    Return the volume of a n-dimensional hyperrectangle.

    Args:
    	shape (list[int]): Contains the length of the hyperrectangle on each of its dimension

    Returns:
    	int: The volume of the hyperrectangle

    """
	v = 1
	for l in shape:
		v *= l
	return v


def default_preprocess(image,input_shape):
	"""
	Apply a default preprocess to an image.

	Check whether the format of the input shape matches NCHW or NHWC
	and call the approriate function accordingly

	Args:
		image (PIL image): an image to which the preprocess will be applied
		input_shape (list[int]): the expected shape of the image after the preprocess

	Returns:
		numpy.array[numpy.float32]: An array with the data of the image
	
	"""
	# It is assumed number of channel < size of the image
	if input_shape[-3] > input_shape[-1]:
		return default_preprocess_NHWC(image,input_shape)
	return default_preprocess_NCHW(image,input_shape)

def default_preprocess_NCHW(image,input_shape):
	"""
	Apply a default preprocess to an image.

	Resize the image to make it match a given input shape and convert it to a numpy array.
	Then change the order of the dimension of the array to match NCHW format
	(batch size is on the first dimension, then Color, then Heigth, then Width).

	Args:
		image (PIL image): an image to which the preprocess will be applied
		input_shape (list[int]): the expected shape of the image after the preprocess

	Returns:
		numpy.array[numpy.float32]: An array with the data of the image in NCHW format
	
	"""
	# Resize the image to match input layer shape.
	input_array = np.asarray(image.resize((input_shape[-1],input_shape[-2])),dtype=np.float32)
	# Change the order of the dimension of the array to match NCHW format
	input_array = np.transpose(input_array, (2, 1, 0))
	if len(input_shape) == 4:
		# Artificially add a dimension to match input layer shape.
		# This dimension correspond to the batch size, mostly used during training.
		# For a single inference test, the batch size is 1.
		input_array = np.asarray([input_array])
	# #normalize
	return input_array

def default_preprocess_NHWC(image,input_shape):
	"""
	Apply a default preprocess to an image.

	Resize the image to make it match a given input shape and convert it to a numpy array.
	Then change the order of the dimension of the array to match NCHW format
	(the batch size is on the first dimension, then Height, then Width, then Color).

	Args:
		image (PIL image): an image to which the preprocess will be applied
		input_shape (list[int]): the expected shape of the image after the preprocess

	Returns:
		numpy.array[numpy.float32]: An array with the data of the image in NHWC format
	
	"""
	# Resize the image to match input layer shape.
	input_array = np.asarray(image.resize((input_shape[-2],input_shape[-3])),dtype=np.float32)
	# Change the order of the dimension of the array to match NCHW format
	input_array = np.transpose(input_array, (1, 0, 2))
	if len(input_shape) == 4:
		# Artificially add a dimension to match input layer shape.
		# This dimension correspond to the batch size, mostly used during training.
		# For a single inference test, the batch size is 1.
		input_array = np.asarray([input_array])
	# #normalize
	return input_array



class DLLibrary(ABC):

	"""
	A class to store a generic Deep Learning Model.

	Attributes:
		self._data_path : str
		    A string to reprensent the path to the file where the model is stored
		self._enable_GPU : bool
			If set to False, prevent the model to try to use a GPU
		self._data_loaded : 
		    The model once it is loaded
		self._number_of_parameters : int
		    The number of parameters

	Methods:
		__init__()
		_load_data(data_path)
		_compute_number_of_parameters()
		_inference_energy_consumption(test_duration,input_data)
		get_number_of_parameters()
		free_model_data()
		run(input_data)
		inference_energy_consumption(profiler,test_duration,input_data,safe_delay,**kwargs)
		train(output_file,train_data,test_data)
		training_energy_consumption(profiler,test_duration,train_data_path,test_data_path,safe_delay,**kwargs)
		training_energy_consumption(profiler,test_duration,train_data_path,test_data_path,safe_delay,**kwargs)

	"""
	
	def __init__(self,data_path):
		"""
		Initialisation.

		First store the data path.
		Then load the data from the file at data_path,
		compute the number of parameters of the model, store it,
		and finally free the memory where the model was loaded.

		Args:
			data_path (str): the path leading to the file to load
		"""
		self._data_path = data_path
		self._load_data(self._data_path)
		self._number_of_parameters = self._compute_number_of_parameters()
		self.free_model_data()

	def get_number_of_parameters(self):
		"""Return the number of parameters that was computed during __init__()."""
		return self._number_of_parameters

	def free_model_data(self):
		"""Free the memory where the model was loaded."""
		self._data_loaded = None

	@abstractmethod
	def _load_data(self,data_path):
		"""overridden by subclass"""
		raise NotImplementedError

	@abstractmethod
	def _compute_number_of_parameters(self):
		"""overridden by subclass"""
		raise NotImplementedError

	@abstractmethod
	def run(self,input_data):
		"""overridden by subclass"""
		raise NotImplementedError

	def inference_energy_consumption(self,profiler,test_duration,input_data,safe_delay=5,**kwargs):
		"""
		Measure the average energy consumption on an inference run

		Wait for safe_delay seconds to avoid interference.
		Then launches inference runs during test_duration seconds.

		Args:
			profiler (EnergyProfiler): The profiler used to measure energy consumption
			test_duration (float): The minimum duration of the tests
			input_data (str): Contains the path to the data to feed to the model
			safe_delay (float): The time (in seconds) of inactivity before the beginning of the tests
			**kwargs: The optional arguments of the inference run function of the model
			
		Returns:
			float: The value of the average energy consumption per run (the unit depends on the profiler used)
		
		"""
		time.sleep(safe_delay)
		filtered_kwargs = {}
		for key, value in kwargs.items():
			if key == "preprocess" and hasattr(value, '__call__'): # It is callable so it is a function
				filtered_kwargs[key] = value
		delta_t, energy, nb_iter = profiler.evaluate(self._inference_energy_consumption,test_duration,input_data,**filtered_kwargs)
		print("nb iter : ",nb_iter)
		print("energy : ",energy)
		return energy/nb_iter

	@abstractmethod
	def _inference_energy_consumption(self,test_duration,input_data):
		"""overridden by subclass"""
		raise NotImplementedError

	def train(self,output_file,train_data,test_data):
		"""overridden by subclass"""
		raise NotImplementedError

	def training_energy_consumption(self,profiler,test_duration,train_data_path,test_data_path,safe_delay=5,**kwargs):
		"""
		Measure the average energy consumption during a training session

		Wait for safe_delay seconds to avoid interference.
		Then launches a training session during test_duration seconds.

		Args:
			profiler (<class 'energyprofiler.EnergyProfiler'>): The profiler to use for profiling
			test_duration (float): The minimum duration of the tests
			train_data_path (str): The path to the data to feed to the training loop of the model
	    	test_data_path (str): The path to the data to feed to the testing phase
	    	safe_delay (float): The time (in seconds) of inactivity before the beginning of the tests
			**kwargs: The optional arguments of the training function of the model
			
		Returns:
			float: The value of the average energy consumption per run (the unit depends on the profiler used)
		
		"""
		time.sleep(safe_delay)
		filtered_kwargs = {}
		for key, value in kwargs.items():
			if key == "batch_size":
				filtered_kwargs[key] = value
			elif key == "number_of_epochs":
				filtered_kwargs[key] = value
			elif key == "print_every":
				filtered_kwargs[key] = value
			elif key == "optimizer":
				filtered_kwargs[key] = value
			elif key == "scheduler":
				filtered_kwargs[key] = value
		print("start training measurements",filtered_kwargs)
		delta_t, energy, nb_iter = profiler.evaluate(self._training_energy_consumption,test_duration,train_data_path,test_data_path,**filtered_kwargs)
		print("nb iter : ",nb_iter)
		print("energy : ",energy)
		return energy/nb_iter

	def _training_energy_consumption(self,test_duration,train_data_path,test_data_path):
		"""overridden by subclass"""
		raise NotImplementedError


class PyTorchLibrary(DLLibrary):

	def __init__(self,data_path,enable_GPU=True,model_constructor=None,input_shape=None):
		"""
		Initialisation.

		Call the initialization of the parent class.
		The devices used can be specified with enable_GPU to enable/disable GPU usage.
		
		Args:
			data_path (str): The path leading to the file to load
			enable_GPU (bool): States whether GPU can be used or not
			model_constructor (class): The class constructor of the model to use 
									if the file at data_path only contains weights
			input_shape (list[int]): The list of the size of the input on each of its dimension

		"""
		self._model_constructor = model_constructor
		self._enable_GPU = enable_GPU
		device_str = "cpu"
		if (self._enable_GPU):
			if (torch.cuda.is_available()):
				device_str = "cuda"
			else:
				warnings.warn("torch.cuda.is_available() returns False, there is no GPU available. ",GPUNotFoundWarning)
		self._device = torch.device(device_str)
		super().__init__(data_path)
		if not self._is_jit_model and model_constructor is None:
			warnings.warn("The data given in argument does not contain a class to instanciate the model, "
						+"therefore the run function will return None.\nTo avoid this issue, "
						+"provide the class constructor of the model using the optional argument model_constructor "
						+"or consider using a model saved using scripted modules provided by jit.",
						PartiallySupportedFileWarning)
		self._input_shape = input_shape

		

	def _load_data(self,data_path):
		"""Load the data in self._data_loaded with pytorch API."""
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="'torch.load' received a zip file that looks like a TorchScript")
			self._data_loaded = torch.load(data_path, map_location=self._device)
		self._is_jit_model = not isinstance(self._data_loaded, dict)
		if self._is_jit_model:
			self._data_loaded = torch.jit.load(data_path, map_location=self._device)

	def _compute_number_of_parameters(self):
		"""
		Compute the number of parameters

		Go through each layer of the model, get their size, deduce the number of parameters and sum them

		Returns:
    		int: The number of parameters of the model
		
		"""
		if self._is_jit_model:
			total=0
			previous_size = 0  # To keep track of the shape of the layers and avoid counting multiple times layer of dim=1
			for layer in list(self._data_loaded.parameters()):
				if (len(list(layer.size())) > 1 or (len(list(layer.size())) > 0 and previous_size != 1)):
					previous_size = len(list(layer.size()))
					total += volume(list(layer.size()))
					
		else:
			params = self._data_loaded
			total = 0
			previous_size = 0  # To keep track of the shape of the layers and avoid counting multiple times layer of dim=1
			for key in params.keys():
				if (len(list(params[key].size())) > 1 or (len(list(params[key].size())) > 0 and previous_size != 1)):
					previous_size = len(list(params[key].size()))
					total += volume(list(params[key].size()))
		return total

	def _define_input_shape(self, input_shape=[1,1,1,1]):
		"""
		Find the input shape of accepted by the PyTorch model

		Goes through trial and error, test different input shapes and find one that does not raise errors.

		Args:
			input_shape (list[int]): The list of the size of the input on each of its dimension

		Returns:
    		list[int]: The input shape accepted by the PyTorch model
		
		"""
		if not self._input_shape is None :
			return
		if self._is_jit_model:
			model = self._data_loaded
		else:
			model = self._model_constructor()
			model.load_state_dict(self._data_loaded)
		model.to(self._device)
		model.eval()  # To set dropout and batch normalization layers to evaluation mode before running inference
		dummy_input = np.zeros(input_shape,dtype=np.float32)

		# # tests to see if it is possible to just read the input size
		# print(model.parameters())
		# print(model.state_dict().keys())
		# # get the layers labeled 0 and 1 
		# regexp = re.compile(r'.*([0]*1(?![0-9]).*|.*(?![1-9])*[0](?![1-9]).*)')
		# print("res:")
		# for name in model.state_dict().keys():
		# 	result = regexp.search(name)
		# 	if not result is None:
		# 		print(name)
		# 		print(result)
		# 		print(list(model.state_dict()[name].shape))

		# Get nb of channel
		try:
			output = model(torch.from_numpy(dummy_input).to(self._device))
		except RuntimeError as err:
			regexp = re.compile(r'expected input.*to have (?P<expected_number_of_channels>[0-9]*) channel')
			for line in str(err).splitlines():
				result = regexp.search(line)	
				if result:
					input_shape[1] = int(result.group('expected_number_of_channels'))
					dummy_input = np.zeros(input_shape,dtype=np.float32)

		output_is_too_small = True
		shapes_cannot_be_multiplied = True
		# Find the point where the input is big enough for the convolution to work fine.
		# and the dimension of the matrix are identical in linear phase
		while output_is_too_small or shapes_cannot_be_multiplied:
			output_is_too_small = False
			shapes_cannot_be_multiplied = False
			try:
				output = model(torch.from_numpy(dummy_input).to(self._device))
			except RuntimeError as err:
				for line in str(err).splitlines():
					if "Output size is too small" in line:
						input_shape[2] = 2*input_shape[2] 
						input_shape[3] = 2*input_shape[3] 
						dummy_input = np.zeros(input_shape,dtype=np.float32)
						output_is_too_small = True
				if not output_is_too_small:
					regexp = re.compile(r"shapes cannot be multiplied .*x(?P<input_dimension>[0-9]*) and (?P<output_dimension>[0-9]*)x")
					for line in str(err).splitlines():
						result = regexp.search(line)	
						if result:
							input_dimension = int(result.group('input_dimension'))
							output_dimension = int(result.group('output_dimension'))
							coeff = output_dimension//input_dimension
							inv_coeff = input_dimension//output_dimension
							if coeff>=inv_coeff:
								# The shape is assumed to be a square so each side is multiplied by sqrt(coeff)
								sqrt_coeff = int(np.sqrt(coeff))
								if sqrt_coeff > 1:
									input_shape[2] = sqrt_coeff*input_shape[2] 
									input_shape[3] = sqrt_coeff*input_shape[3] 
								else:  # the shape cant be squared
									input_shape[2] = coeff*input_shape[2] 
							else :
								# The shape is assumed to be a square so each side is divided by sqrt(1/coeff)
								sqrt_coeff = int(np.sqrt(inv_coeff))
								if sqrt_coeff > 1:
									input_shape[2] = input_shape[2]//sqrt_coeff
									input_shape[3] = input_shape[3]//sqrt_coeff 
								else:  # the shape cant be squared
									input_shape[2] = input_shape[2]//sqrt_coeff
							dummy_input = np.zeros(input_shape,dtype=np.float32)
							shapes_cannot_be_multiplied = True
		self._input_shape = input_shape

	def _inference_energy_consumption(self,test_duration,input_data,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
			input_data : Contains the path to the data to feed to the model
	    	preprocess (<class 'function'>): the preprocess to apply to the input data
	    Returns:
	    	int: the number of time that the inference was run
		"""
		self._load_data(self._data_path)
		model = self._data_loaded
		if not self._is_jit_model:
			if self._model_constructor is None:
				warnings.warn("The data given in argument does not contain a class to instanciate the model, "
						+"therefore the run function will return None.\nTo avoid this issue, "
						+"provide the class constructor of the model using the optional argument model_constructor "
						+"or consider using a model saved using scripted modules provided by jit.",
						PartiallySupportedFileWarning)
				return None
			else:
				model = self._model_constructor()
				model.load_state_dict(self._data_loaded)
		
		model.to(self._device)

		model.eval()  # To set dropout and batch normalization layers to evaluation mode before running inference
		
		self._define_input_shape()
		input_shape = self._input_shape		
		
		image = Image.open(input_data)

		t_start = time.time()
		nb_iter = 0
		
		while(time.time()-t_start < test_duration or nb_iter==0):
			nb_iter += 1
			input_array = preprocess(image,input_shape)
			output = model(torch.from_numpy(input_array).to(self._device))

		self.free_model_data()
		return nb_iter

	def run(self,input_data,input_size=None,preprocess=default_preprocess):
		"""
		Run the inference model.

		Args:
	    	input_data : Contains the path to the data to feed to the model
	    	input_size (list[int]): the expected shape of the input (None by default if unkown)
	    	preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)

	    Returns:
	    	The raw output given by the model
		"""
		self._load_data(self._data_path)
		model = self._data_loaded
		if not self._is_jit_model:
			if self._model_constructor is None:
				warnings.warn("The data given in argument does not contain a class to instanciate the model, "
						+"therefore the run function will return None.\nTo avoid this issue, "
						+"provide the class constructor of the model using the optional argument model_constructor "
						+"or consider using a model saved using scripted modules provided by jit.",
						PartiallySupportedFileWarning)
				return None
			else:
				model = self._model_constructor()
				model.load_state_dict(self._data_loaded)
		
		model.to(self._device)

		model.eval()  # To set dropout and batch normalization layers to evaluation mode before running inference
		
		self._define_input_shape()
		input_shape = self._input_shape		
		
		image = Image.open(input_data)
		
		input_array = preprocess(image,input_shape)
		output = model(torch.from_numpy(input_array).to(self._device))
		
		self.free_model_data()
		return output

	def train(self,output_file,train_data_path,test_data_path,save_with_jit=None,batch_size=1,number_of_epochs=20,print_every=5,preprocess=None,optimizer=None,scheduler=None):
		"""
		Train the model.

		Trains the model and saves it at the location given in argument

		Args:
			output_file (str): The path where to save the model after training (None or "" not to save the model)
	    	train_data_path (str): The path to the data to feed to the training loop of the model
	    	test_data_path (str): The path to the data to feed to the testing phase
	    	save_with_jit (bool): Indicate whether to save the trained model with torch.jit or not.
	    						If not precised, the save format is the same as the one of the original model
	    	batch_size (int): The size of the batches during training
	    	number_of_epochs (int): The number of epochs (loop)
	    	print_every (int): The number of step between each evaluation of the model to print accuracy, loss, etc
	    	preprocess (<class 'function'>): The preprocess to apply to each image
			optimiser (<class 'torch.optim.sgd.SGD'>): The PyTorch optimiser to use for training
	    	scheduler (<class 'torch.optim.lr_scheduler.StepLR'>): The PyTorch scheduler to use for training
	    	
	    Returns:
	    	int: The number of steps (= the number of batches) run during the training

	    """
		self._load_data(self._data_path)
		model = self._data_loaded
		if not self._is_jit_model:
			if self._model_constructor is None:
				warnings.warn("The data given in argument does not contain a class to instanciate the model, "
						+"therefore the run function will return None.\nTo avoid this issue, "
						+"provide the class constructor of the model using the optional argument model_constructor "
						+"or consider using a model saved using scripted modules provided by jit.",
						PartiallySupportedFileWarning)
				return
			else:
				model = self._model_constructor()
				model.load_state_dict(self._data_loaded)
		
		device = self._device
		model.to(device)

		if preprocess is None:
			self._define_input_shape()
			input_shape = self._input_shape	

			preprocess = transforms.Compose([
			    transforms.Resize(input_shape[-2]), # in -2 it's never the batchsize neither the channel number
			    transforms.CenterCrop(input_shape[-2]),
			    transforms.ToTensor(),
			    # arbitrarily chosen values
			    transforms.Normalize(mean=[0.3418, 0.3126, 0.3224], std=[0.1627, 0.1632, 0.1731])
			])

		trainset = torchvision.datasets.ImageFolder(train_data_path, transform=preprocess)
		train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=False)

		testset = torchvision.datasets.ImageFolder(test_data_path, transform=preprocess)
		test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

		criterion = nn.CrossEntropyLoss()
		if optimizer is None:
			# arbitrarily chosen values
			optimizer = optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)
		if scheduler is None:
			# not sure about the values
			scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)
		epochs = number_of_epochs
		steps = 0
		test_loss = 0
		running_loss = 0
		train_losses = []
		test_losses = []
		
		start_time = time.time()
		for epoch in range(epochs):
		    steps = 0
		    for inputs, labels in train_loader:
		        steps += 1
		        inputs, labels = inputs.to(device), labels.to(device)
		       
		        optimizer.zero_grad()
		        
		        logps = model.forward(inputs)
		        loss = criterion(logps, labels)
		        loss.backward()
		        optimizer.step()
		        
		        running_loss += loss.item()
		        
		        if steps % print_every == 0:
		            print("steps : ",steps,", delta time : ",(time.time()-start_time)*10//1/10)
		            test_loss = 0
		            accuracy_t = 0
		            model.eval()
		            with torch.no_grad():
		                for inputs_val, labels_val in test_loader:
		                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
		                    
		                    logps_t = model.forward(inputs_val)
		                    batch_loss = criterion(logps_t, labels_val)
		                    test_loss += batch_loss.item()
		                    
		                    top_p, top_class = logps_t.topk(1, dim=1)
		                    equals = top_class == labels_val.view(*top_class.shape)
		                    accuracy_t += torch.mean(equals.type(torch.FloatTensor)).item()
		            
		            print(f"Epoch {epoch+1}/{epochs}.. "
		                  f"Train loss: {running_loss/print_every:.3f}.. "
		                  f"Test loss: {test_loss/len(test_loader):.3f}.. "
		                  f"Test accuracy: {accuracy_t/len(test_loader):.3f}")
		            running_loss = 0
		            model.train()
		    scheduler.step()
		    test_losses.append(test_loss/len(test_loader))
		    train_losses.append(running_loss/len(train_loader))

		if (not output_file is None) and (len(output_file) > 0):
			if save_with_jit is None:
				save_with_jit = self._is_jit_model
			if save_with_jit:
				model.eval()
				jitmodel = torch.jit.script(model)
				torch.jit.save(jitmodel, output_file)
			else:
				torch.save(model.state_dict(), output_file)

		self.free_model_data()
		return steps

	def _training_energy_consumption(self,test_duration,train_data_path,test_data_path,batch_size=1,number_of_epochs=1,preprocess=None,optimizer=None,scheduler=None,print_every=100):
		"""
		Run training sessions several times to measure the equivalence of CO2 emissions of a training session.

		Args:
			test_duration (float): the duration of the tests in seconds
			train_data_path (str): The path to the data to feed to the training loop of the model
	    	test_data_path (str): The path to the data to feed to the testing phase
	    	batch_size (int): The size of the batches during training
	    	number_of_epochs (int): The number of epochs (loop)
	    	preprocess (<class 'function'>): the preprocess to apply to each image
	    									(a default preprocess is applied if none is specified)
	    	optimiser (<class 'torch.optim.sgd.SGD'>): The PyTorch optimiser to use for training
	    	scheduler (<class 'torch.optim.lr_scheduler.StepLR'>): The PyTorch scheduler to use for training
	    	print_every (int): The number of step between each evaluation of the model to print accuracy, loss, etc
	    
	    Returns:
	    	int: the number of steps that the training took

		"""
		t_start = time.time()
		nb_steps = 0
		
		while(time.time()-t_start < test_duration or nb_steps == 0):
			nb_steps += self.train("",train_data_path,test_data_path,print_every=print_every,number_of_epochs=number_of_epochs,batch_size=batch_size,optimizer=optimizer,scheduler=scheduler,preprocess=preprocess)

		self.free_model_data()
		return nb_steps
		

class ONNXLibrary(DLLibrary):

	def __init__(self,data_path,enable_GPU=True):
		"""
		Initialisation.

		Call the initialization of the parent class.
		The devices used can be specified with enable_GPU to enable/disable GPU usage.
		
		Args:
			data_path (str): the path leading to the file to load
			enable_GPU (bool): states whether GPU can be used or not

		"""
		self._enable_GPU = enable_GPU
		super().__init__(data_path)
		
	def _load_data(self,data_path):
		"""
		Load the data in self._data_loaded with ONNX API.

		Args:
			data_path (str): the path leading to the file to load
		"""
		self._data_loaded = onnx.load(data_path)

	def _compute_number_of_parameters(self):
		"""
		Compute the number of parameters.

		Go through each layer of the model, get their size, deduce the number of parameters and sum them.

		Returns:
    		int: The number of parameters of the model
		
		"""
		model = self._data_loaded
		onnx_weights = model.graph.initializer
		total = 0
		for w in onnx_weights:
			shape = numpy_helper.to_array(w).shape
			if (len(list(shape)) > 0):
				total += volume(list(shape))
		return total

	def _inference_energy_consumption(self,test_duration,input_data,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
			input_data : Contains the path to the data to feed to the model
			preprocess (<class 'function'>): the preprocess to apply to the input data
	    Returns:
	    	int: the number of time that the inference was run

		"""
		providers = ort.get_available_providers()
		if (not self._enable_GPU):
			providers = ["CPUExecutionProvider"]
		elif (self._enable_GPU and not "CUDAExecutionProvider" in providers):
			warnings.warn("CUDAExecutionProvider not found, there is no GPU available. ",GPUNotFoundWarning)
		
		print("providers : ",providers)
		ort_sess = ort.InferenceSession(self._data_path, None, providers=providers)
		
		input_shape = ort_sess.get_inputs()[0].shape
		input_name = ort_sess.get_inputs()[0].name

		image = Image.open(input_data)
		
		t_start = time.time()
		nb_iter = 0
		
		while(time.time()-t_start < test_duration or nb_iter==0):
			nb_iter += 1
			input_array = preprocess(image,input_shape)
			_ = ort_sess.run(None, {input_name: input_array})

		self.free_model_data()
		return nb_iter

	def run(self,input_data,preprocess=default_preprocess):
		"""
		Run the inference model.

		Args:
	    	input_data : Contains the data to feed to the model
			preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)
	    Returns:
	    	The raw output given by the model
		"""
		providers = ort.get_available_providers()
		if (not self._enable_GPU):
			providers = ["CPUExecutionProvider"]
		elif (self._enable_GPU and not "CUDAExecutionProvider" in providers):
			warnings.warn("CUDAExecutionProvider not found, there is no GPU available. ",GPUNotFoundWarning)
		
		ort_sess = ort.InferenceSession(self._data_path, None, providers=providers)
		
		input_shape = ort_sess.get_inputs()[0].shape
		input_name = ort_sess.get_inputs()[0].name

		image = Image.open(input_data)
		
		input_array = preprocess(image,input_shape)
		output = ort_sess.run(None, {input_name: input_array})

		self.free_model_data()
		return output


class TensorFlowLibrary(DLLibrary):

	def __init__(self,data_path,enable_GPU=True):
		"""
		Initialisation.

		Call the initialization of the parent class.
		The devices used can be specified with enable_GPU to enable/disable GPU usage.
		
		Args:
			data_path (str): the path leading to the file to load
			enable_GPU (bool): states whether GPU can be used or not

		"""
		self._enable_GPU = enable_GPU
		self._device_str = "cpu"
		if enable_GPU:
			if len(tf.config.list_physical_devices('GPU')) > 0:
				self._device_str = "gpu"
			else:
				warnings.warn("tf.config.list_physical_devices('GPU') is empty, there is no GPU available. ",GPUNotFoundWarning)
		super().__init__(data_path)
		self._input_shape = None
		self._load_data(data_path)
		self._define_input_shape()
		self.free_model_data()

	def _load_data(self,data_path):
		"""
		Load the data in self._data_loaded with tensorflow.keras API.
		
		Args:
			data_path (str): the path leading to the file to load
		"""
		
		with tf.device(self._device_str):
			# compile=False means the model can be used only in inference (avoid warnings about training)
			# self._data_loaded = tf.keras.models.load_model(data_path, compile=False)
			try:
				self._data_loaded = tf.saved_model.load(data_path)
				self._is_keras_model = False
				#print("not keras")
			except:
				self._is_keras_model = True			
				self._data_loaded = tf.keras.models.load_model(data_path)
				#print("keras")		

	def _compute_number_of_parameters(self):
		"""
		Get the number of parameters using a function from the tensorflow.keras API.

		Returns:
    		int: The number of parameters of the model
		
		"""
		if self._is_keras_model:
			total = self._data_loaded.count_params()
		else:
			# Count param for tensorflow saved_model
			model = self._data_loaded
			trainable_count = 0
			for w in list(model.variables.trainable_weights):
				e = list(w)
				v = 1
				l = len(e)
				while (l > 1):
					v *= len(e)
					try :
						e = list(e[0])
						l = len(e)
					except TypeError:
						l = 1
				trainable_count += v
			non_trainable_count = 0
			for w in list(model.variables.non_trainable_weights):
				e = list(w)
				v = 1
				l = len(e)
				while (l > 1):
					v *= len(e)
					try :
						e = list(e[0])
						l = len(e)
					except TypeError:
						l = 1
				non_trainable_count += v
			total = trainable_count + non_trainable_count
		return total

	def _define_input_shape(self):
		"""
		Find the input shape of accepted by the TensorFlow model

		Returns:
    		list[int]: The shape accepted by the TensorFlow model
		
		"""
		if not self._input_shape is None :
			return
		model = self._data_loaded
		if self._is_keras_model:
			model_config = model.get_config()
			# We want inputshape = (batch size, number of color channel, width, height)
			# batch size can be None since the model has some flexibility
			input_shape = list(model_config["layers"][0]["config"]["batch_input_shape"])
		else:
			signatures = list(model.signatures.keys())
			if len(signatures) > 0:
				input_shape = list(model.signatures[signatures[0]].inputs[0].shape)
			else:
				input_shape = list(model.variables[0].shape)
		self._input_shape = input_shape

	def _inference_energy_consumption(self,test_duration,input_data,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
	    	input_data : Contains the path to the data to feed to the model
	    	preprocess (<class 'function'>): the preprocess to apply to the input data 

	    Returns:
	    	int: the number of time that the inference was run

		"""
		self._load_data(self._data_path)
		model = self._data_loaded

		input_shape = self._input_shape

		if len(input_shape) == 4:
			input_shape[0] = 1
		
		image = Image.open(input_data)
		
		t_start = time.time()
		nb_iter = 0
		
		with tf.device(self._device_str):
			while(time.time()-t_start < test_duration or nb_iter==0):
				nb_iter += 1
				input_array = preprocess(image,input_shape)
				_ = model(input_array, training=False)
		
		self.free_model_data()
		return nb_iter


	def run(self,input_data,preprocess=default_preprocess):
		"""
		Run the inference model.

		Args:
	    	input_data : Contains the path to the data to feed to the model
	    	preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)

	    Returns:
	    	The raw output given by the model
		"""
		self._load_data(self._data_path)
		model = self._data_loaded
		
		input_shape = self._input_shape

		if len(input_shape) == 4:
			input_shape[0] = 1
		print("input_shape : ",input_shape)
		
		image = Image.open(input_data)

		if len(input_shape) == 4:
			input_shape[0] = 1
		
		with tf.device(self._device_str):
			input_array = preprocess(image,input_shape)
			output = model(input_array, training=False)
			
		self.free_model_data()
		return output

	def train(self,output_file,train_data_path,test_data_path,batch_size=1,number_of_epochs=20,validation_split=0.2,print_every=5,preprocess=default_preprocess,optimizer=None):
		"""
		Train the model.

		Trains the model and saves it at the location given in argument

		Args:
			output_file (str): The path where to save the model after training (None or "" not to save the model)
	    	train_data_path (str): The path to the data to feed to the training loop of the model
	    	test_data_path (str): The path to the data to feed to the testing phase
	    	batch_size (int): The size of the batches during training
	    	number_of_epochs (int): The number of epochs (loop)
	    	validation_split (float): the proportion of the training set used to evaluate the progress
	    	print_every (int): The number of step between each evaluation of the model to print accuracy, loss, etc
	    	preprocess (<class 'function'>): The preprocess to apply to each image
	    	optimiser : The TensorFlow optimiser to use for training

	    Returns
	    	int: The number of steps (= the number of batches) run during the training

	    """
		self._load_data(self._data_path)
		model = self._data_loaded
		
		input_shape = self._input_shape

		if len(input_shape) > 3:
			input_shape = input_shape[-3:]
		print("input_shape : ",input_shape)		

		class_names = os.listdir(train_data_path)
		class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

		nb_classes = len(class_names)

		datasets = [train_data_path, test_data_path]
		output = []

		# Iterate through training and test sets
		for dataset in datasets:

			images = []
			labels = []

			print("Loading {}".format(dataset))

			# Iterate through each folder corresponding to a category
			for folder in os.listdir(dataset):
				label = class_names_label[folder]

				# Iterate through each image in our folder
				for file in tqdm(os.listdir(os.path.join(dataset, folder))):
				    
					# Get the path name of the image
					input_data = os.path.join(os.path.join(dataset, folder), file)
					image = Image.open(input_data)
					input_array = preprocess(image,input_shape)

					# Append the image and its corresponding label to the output
					images.append(input_array)
					labels.append(label)
			        
			images = np.array(images, dtype = 'float32')
			labels = np.array(labels, dtype = 'int32')   

			output.append((images, labels))

		(train_images, train_labels), (test_images, test_labels) = output

		if self._is_keras_model:
			train_images, train_labels = shuffle(train_images, train_labels, random_state=42)
		else:
			# Prepare the training dataset.
			train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
			train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
			# Prepare the validation dataset.
			val_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
			val_dataset = val_dataset.batch(batch_size)

		if optimizer is None:
			# arbitrarily chosen values
			optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

		if self._is_keras_model:
			model.compile(
			    optimizer=optimizer,  # Optimizer
			    # Loss function to minimize
			    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
			    # List of metrics to monitor
			    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
			)
			model.fit(train_images, train_labels, batch_size=batch_size, epochs=number_of_epochs, validation_split=validation_split)
		else:
			# Instantiate a loss function.
			loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
			# Prepare the metrics.
			train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
			val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
			for epoch in range(number_of_epochs):
			    print("\nStart of epoch %d" % (epoch,))
			    start_time = time.time()
			    # Iterate over the batches of the dataset.
			    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
			        # Open a GradientTape to record the operations run
			        # during the forward pass, which enables auto-differentiation.
			        with tf.GradientTape() as tape:
			            # Run the forward pass of the layer.
			            # The operations that the layer applies
			            # to its inputs are going to be recorded
			            # on the GradientTape.
			            logits = model(x_batch_train, training=True)  # Logits for this minibatch
			            # Compute the loss value for this minibatch.
			            loss_value = loss_fn(y_batch_train, logits)
					# Use the gradient tape to automatically retrieve
			        # the gradients of the trainable variables with respect to the loss.
			        grads = tape.gradient(loss_value, model.variables.trainable_weights)
			        # Run one step of gradient descent by updating
			        # the value of the variables to minimize the loss.
			        optimizer.apply_gradients(zip(grads, model.variables.trainable_weights))
			        # Update training metric.
			        train_acc_metric.update_state(y_batch_train, logits)

			        if step % print_every == 0:
			            print("Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
			            print("Seen so far: %s samples" % ((step + 1) * batch_size))
			    # Display metrics at the end of each epoch.
			    train_acc = train_acc_metric.result()
			    print("Training acc over epoch: %.4f" % (float(train_acc),))
			    # Reset training metrics at the end of each epoch
			    train_acc_metric.reset_states()
			    # Run a validation loop at the end of each epoch.
			    for x_batch_val, y_batch_val in val_dataset:
			        val_logits = model(x_batch_val, training=False)
			        # Update val metrics
			        val_acc_metric.update_state(y_batch_val, val_logits)
			    val_acc = val_acc_metric.result()
			    val_acc_metric.reset_states()
			    print("Validation acc: %.4f" % (float(val_acc),))
			    print("Time taken: %.2fs" % (time.time() - start_time))

		if (not output_file is None) and (len(output_file) > 0):
			saved_with_keras = self._data_path.endswith(".h5")
			if saved_with_keras:
				# automatically choose between keras and TensorFlow SavedModel
				# depending on whether output file ends with ".h5" or not
				model.save(output_file, overwrite=True, include_optimizer=True)
			else:
				tf.saved_model.save(model, output_file)
		
		self.free_model_data()
		return number_of_epochs*len(train_images)*(1-validation_split)

	def _training_energy_consumption(self,test_duration,train_data_path,test_data_path,batch_size=1,number_of_epochs=20,validation_split=0.2,print_every=5,preprocess=default_preprocess,optimizer=None):
		"""
		Run training sessions several times to measure the equivalence of CO2 emissions of a training session.

		Args:
			test_duration (float): the duration of the tests in seconds
			train_data_path (str): The path to the data to feed to the training loop of the model
	    	test_data_path (str): The path to the data to feed to the testing phase
	    	batch_size (int): The size of the batches during training
	    	number_of_epochs (int): The number of epochs (loop)
	    	validation_split (float): the proportion of the training set used to evaluate the progress
	    	print_every (int): The number of step between each evaluation of the model to print accuracy, loss, etc
	    	preprocess (<class 'function'>): the preprocess to apply to each image
	    	optimiser : The TensorFlow optimiser to use for training
	    	
	    Returns:
	    	int: the number of steps that the training took

		"""
		t_start = time.time()
		nb_steps = 0
		
		while(time.time()-t_start < test_duration or nb_steps == 0):
			print("batch_size",batch_size)
			nb_steps += self.train("",train_data_path,test_data_path,batch_size=batch_size,number_of_epochs=number_of_epochs,validation_split=validation_split,print_every=print_every,preprocess=preprocess,optimizer=optimizer)
			
		return nb_steps


class DLModelFactory:

	"""
	A class to create instances of the appropriate sub-classes of DLLibrary for given files.

	Attributes:
		self._creators : dict of str: class constructor
			A dictionnary with :
				input: The extension of a DL model format (example : ".pt")
		    	output: The DLLibrary sub-class constructor that correpond to this format

	Methods:
		__init__()
		register_format(format, creator)
		register_predefined_formats()
		get_model(data_path)

	"""

	def __init__(self):
		"""
		Initialization.

		Create an empty dictionnary to store later the class constructor for the supported formats.

		"""
		self._creators = {}

	def register_format(self, format, creator):
		"""
		Map a class constructor to a file extension.

		Args:
			format (str): Contains the extension corresponding to a file format (for example ".pt")
			creator (class constructor): Contains the constructor of the class correponding to format

		"""
		self._creators[format] = creator

	def register_predefined_formats(self):
		"""
		Hard-coded registration of the formats already coded.

		the formats are:
			- PyTorch models with ".pt" extension
			- ONNX models with ".onnx" extension
			- TensorFlow models saved as a folder or using keras API with ".h5" extension

		"""
		self.register_format(".pt",PyTorchLibrary)
		self.register_format(".onnx",ONNXLibrary)
		self.register_format(".h5",TensorFlowLibrary)
		self.register_format("/",TensorFlowLibrary)

	def get_model(self, data_path):
		"""
	    Return the class constructor associated with the format of the file at data_path.

	    Args:
	    	data_path (str): Contains the path to a file where a DL model is stored

	    Returns:
	    	class constructor: The constructor in self._creators mapped to the extension of the file at data_path.

	    Raises:
        	ValueError: If the extension of the file at data_path is not mapped

	    """
		creator = None
		for format in self._creators:
			if data_path.endswith(format):
				creator = self._creators[format]
		if not creator:
			raise ValueError(format)
		return creator