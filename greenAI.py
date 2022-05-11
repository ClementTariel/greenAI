#!/usr/bin/python

from codecarbon import EmissionsTracker
from codecarbon import track_emissions

from abc import ABC, abstractmethod

import os
import sys
import warnings

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn 
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import tensorflow as tf

import re  # regex

import time


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


import torch.nn as nn
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv_bn_1 = nn.BatchNorm2d(16)
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv_bn_2 = nn.BatchNorm2d(32)
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        
        self.conv3 = nn.Conv2d(32, 64 , 3, 1, padding=1)
        self.conv_bn_3 = nn.BatchNorm2d(64)
        torch.nn.init.normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        
        self.conv4 = nn.Conv2d(64, 128 , 3, 1, padding=1)
        self.conv_bn_4 = nn.BatchNorm2d(128)
        torch.nn.init.normal_(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        
        #self.conv5 = nn.Conv2d(64, 128 , 3, 1, padding=1)
        #self.conv_bn_5 = nn.BatchNorm2d(128)
        #torch.nn.init.normal_(self.conv5.weight)
        #torch.nn.init.zeros_(self.conv5.bias)
        
        #self.conv4 = nn.Conv2d(256, 512 , 3, 1, padding=1)
        #self.conv_bn_4 = nn.BatchNorm2d(512)
        #torch.nn.init.normal_(self.conv4.weight)
        #torch.nn.init.zeros_(self.conv4.bias)
        
        self.pool  = nn.MaxPool2d(2,2)

        self.act   = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.2)
        
        self.mlp = nn.Sequential(
            nn.Linear(4*4*128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 2)  
            #nn.Softmax(dim=-1)    
        )
    

    def forward(self, x):
        x = self.conv_bn_1(self.conv1(x))
        x = self.pool(self.act(x))
        x = self.drop(x)

        x = self.conv_bn_2(self.conv2(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        x = self.conv_bn_3(self.conv3(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        x = self.conv_bn_4(self.conv4(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        bsz, nch, height, width = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        y = self.mlp(x)

        return y




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
		numpy.array[numpy.float32]: An array with the data of the image in NCHW format
	
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
	return input_array



class DLLibrary(ABC):

	"""
	A class to store a generic Deep Learning Model.

	Attributes:
		self._data_path : str
		    A string to reprensent the path to the file where the model is stored
		self._data_loaded : 
		    The model once it is loaded
		self._number_of_parameters : int
		    The number of parameters

	Methods:
		__init__()
		get_number_of_parameters()
		free_model_data()
		_load_data(data_path)
		_compute_number_of_parameters()
		run(input_data)

	"""
	
	def __init__(self,data_path):
		"""
		Initialisation.

		First store the data path.
		Then load the data from the file at data_path,
		compute the number of parameters of the model, store it,
		and finally free the memory where the model was loaded.

		Args:
			data_path (string): the path leading to the file to load
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
		pass

	@abstractmethod
	def _compute_number_of_parameters(self):
		"""overridden by subclass"""
		pass

	@abstractmethod
	def run(self,input_data):
		"""overridden by subclass"""
		pass

	@abstractmethod
	def inference_emissions(self,input_data):
		"""overridden by subclass"""
		pass


class PyTorchLibrary(DLLibrary):

	def __init__(self,data_path,enable_GPU=True,model_constructor=None):
		"""
		Initialisation.

		Call the initialization of the parent class.
		The devices used can be specified with enable_GPU to enable/disable GPU usage.
		
		Args:
			data_path (string): the path leading to the file to load
			enable_GPU (bool): states whether GPU can be used or not
			model_constructor (class): the class constructor of the model to use 
									if the file at data_path only contains weights

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
		self._input_shape = None

		

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
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################
		"""
		Find the input shape of accepted by the PyTorch model

		Goes through trial and error, test different input shapes and find one that does not raise errors.

		Returns:
    		list[int]: The shape accepted by the PyTorch model
		
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
							# The shape is assumed to be a square so each side is multiplied by sqrt(coeff)
							sqrt_coeff = int(np.sqrt(coeff))
							input_shape[2] = sqrt_coeff*input_shape[2] 
							input_shape[3] = sqrt_coeff*input_shape[3] 
							dummy_input = np.zeros(input_shape,dtype=np.float32)
							shapes_cannot_be_multiplied = True
		self._input_shape = input_shape

	def inference_emissions(self,test_duration,input_data,input_size=None,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
			input_data : Contains the path to the data to feed to the model
	    	input_size (list[int]): the expected shape of the input (None by default if unkown)
	    	preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)
	    Returns:
	    	float: the average equivalence of CO2 emissions of a run (in kg eq. CO2 per run)

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
		input_array = preprocess(image,input_shape)
		_ = model(torch.from_numpy(input_array).to(self._device))
		t_stop = time.time()
		delta_t = t_stop - t_start

		# To do a runtest that last about test_duration seconds
		nb_iter = int(test_duration/(delta_t))
		if nb_iter < 1:
			nb_iter = 1

		tracker = EmissionsTracker()
		tracker.start()

		for _ in range(nb_iter):
			input_array = preprocess(image,input_shape)
			output = model(torch.from_numpy(input_array).to(self._device))

		emissions: float = tracker.stop()
		self.free_model_data()
		return emissions/nb_iter

	def run(self,input_data,input_size=None,preprocess=default_preprocess):
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################		
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


class ONNXLibrary(DLLibrary):

	def __init__(self,data_path,enable_GPU=True):
		"""
		Initialisation.

		Call the initialization of the parent class.
		The devices used can be specified with enable_GPU to enable/disable GPU usage.
		
		Args:
			data_path (string): the path leading to the file to load
			enable_GPU (bool): states whether GPU can be used or not

		"""
		self._enable_GPU = enable_GPU
		super().__init__(data_path)
		
	def _load_data(self,data_path):
		"""Load the data in self._data_loaded with ONNX API."""
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

	def inference_emissions(self,test_duration,input_data,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
			input_data : Contains the data to feed to the model
			preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)
	    Returns:
	    	float: the average equivalence of CO2 emissions of a run (in kg eq. CO2 per run)

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
		# Approximate how long it takes to run once.
		t_start = time.time()
		input_array = preprocess(image,input_shape)
		_ = ort_sess.run(None, {input_name: input_array})
		t_stop = time.time()
		delta_t = t_stop - t_start

		# To do a runtest that last about test_duration seconds
		nb_iter = int(test_duration/(delta_t))
		if nb_iter < 1:
			nb_iter = 1
		
		tracker = EmissionsTracker()
		tracker.start()

		for _ in range(nb_iter):
			input_array = preprocess(image,input_shape)
			_ = ort_sess.run(None, {input_name: input_array})

		emissions: float = tracker.stop()
		self.free_model_data()
		return emissions/nb_iter

	def run(self,input_data,preprocess=default_preprocess):
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################
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
			data_path (string): the path leading to the file to load
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

	def _load_data(self,data_path):
		"""Load (on the cpu) the data in self._data_loaded with tensorflow.keras API."""
		
		with tf.device(self._device_str):
			# compile=False means the model can be used only in inference (avoid warnings about training)
			# self._data_loaded = tf.keras.models.load_model(data_path, compile=False)
			self._data_loaded = tf.keras.models.load_model(data_path)

	def _compute_number_of_parameters(self):
		"""
		Get the number of parameters using a function from the tensorflow.keras API.

		Returns:
    		int: The number of parameters of the model
		
		"""
		total = self._data_loaded.count_params()
		return total

	def inference_emissions(self,test_duration,input_data,preprocess=default_preprocess):
		"""
		Run the inference model several times to measure the average equivalence of CO2 emissions of a run.

		Args:
	    	test_duration (float): the duration of the tests in seconds
	    	input_data : Contains the path to the data to feed to the model
	    	preprocess (<class 'function'>): the preprocess to apply to the input data 
	    									(a default preprocess is applied if none is specified)

	    Returns:
	    	float: the average equivalence of CO2 emissions of a run (in kg eq. CO2 per run)

		"""
		self._load_data(self._data_path)
		model = self._data_loaded
		model_config = model.get_config()
		# We want inputshape = (batch size, number of color channel, width, height)
		# batch size can be None since the model has some flexibility
		input_shape = list(model_config["layers"][0]["config"]["batch_input_shape"])
		
		image = Image.open(input_data)

		if len(input_shape) == 4:
			input_shape[0] = 1
		# Approximate how long it takes to run once.
		t_start = time.time()
		with tf.device(self._device_str):
			input_array = preprocess(image,input_shape)
			_ = model(input_array, training=False)
		t_stop = time.time()
		delta_t = t_stop - t_start

		# To do a runtest that last about test_duration seconds
		nb_iter = int(test_duration/(delta_t))
		if nb_iter < 1:
			nb_iter = 1
		
		tracker = EmissionsTracker()
		tracker.start()

		with tf.device(self._device_str):
			for _ in range(nb_iter):
				input_array = preprocess(image,input_shape)
				_ = model(input_array, training=False)

		emissions: float = tracker.stop()
		self.free_model_data()
		return emissions/nb_iter


	def run(self,input_data,preprocess=default_preprocess):
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################
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
		model_config = model.get_config()
		# We want inputshape = (batch size, number of color channel, width, height)
		# batch size can be None since the model has some flexibility
		input_shape = list(model_config["layers"][0]["config"]["batch_input_shape"])
		print("input_shape : ",input_shape)
		
		image = Image.open(input_data)

		if len(input_shape) == 4:
			input_shape[0] = 1
		
		with tf.device(self._device_str):
			input_array = preprocess(image,input_shape)
			output = model(input_array, training=False)
			
		self.free_model_data()
		return output



class DLModelFactory:

	"""
	A class to create instances of the appropriate sub-classes of DLLibrary for given files.

	Attributes:
		self._creators : dict of str: class constructor
			A dictionnary with :
				input: The extension of a DL model format (example : ".pt")
		    	output: The DLLibrary sub-class that correpond to this format

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
			format (str): Contains the extension corresponding to a file format
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




# Execute when the module is not initialized from an import statement.
if __name__ == '__main__':	
	factory = DLModelFactory()
	factory.register_predefined_formats()
	if (len(sys.argv) > 1):  # Check if there is at least one file to analize
		print ('Number of models:', len(sys.argv)-1, '.')
		print("")
		for k in range(1,len(sys.argv)):
			#########################
			#						#
			#	Work in progress	#
			#						#
			#########################
			print ('model :', sys.argv[k])
			# Check if the given file is supported
			try :
				model_constructor = factory.get_model(str(sys.argv[k]))
			except ValueError:
				print("unsupported file format")
				print("")
				continue
			# just to test .pt model with constructor in argument
			try:
				model = model_constructor(str(sys.argv[k]),model_constructor=Net)
			except TypeError:
				model = model_constructor(str(sys.argv[k]))
			print("/======================================\\")
			#print(model.run("../cat.jpg"))
			print("emissions :",model.inference_emissions(1,"../glacier.jpg")," kg eq. CO2 per run")
			print("total :",model.get_number_of_parameters(),"parameters")
			print(model.run("../glacier.jpg"))
			print("\\======================================/")
			print("")
			
	else :
		print("")
		print("No arguments found")
		print("Usage: python greenAI.py file_names")
		print("")

