#!/usr/bin/env python3

from abc import ABC, abstractmethod

import os
import sys
import warnings
from threading import Thread , Timer
import subprocess as sp

import numpy as np
#import cv2
from PIL import Image

import torch
import torch.nn as nn 
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import tensorflow as tf

import re  # regex

import time

from dllibrary import *
from energyprofiler import *

from codecarbon import EmissionsTracker
from codecarbon import track_emissions

# Execute when the module is not initialized from an import statement.
if __name__ == '__main__':	


	# to test .pt model saved as weight only

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


	# tf.debugging.set_log_device_placement(True)

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
			input_file = "../images/glacier.jpg"
			test_duration = 20
			delay_between_measures = 5
			print("start test(s) of ",test_duration," seconds (+ potential additionnal time depending on the delay between 2 measures)")
			profiler = PyJoulesProfiler(delay_between_measures)
			print("PyJoulesProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			profiler = EnergyUsageProfiler(delay_between_measures)
			print("EnergyUsageProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			profiler = LikwidProfiler(delay_between_measures)
			print("LikwidProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			profiler = CodecarbonProfiler(delay_between_measures,save_to_file=False)
			print("CodecarbonProfiler (use a tracker from codecarbon)")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			profiler = PerfProfiler(delay_between_measures)
			print("PerfProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			try:
				profiler = NvidiaProfiler(delay_between_measures)
				print("NvidiaProfiler")
				print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			except :
				print("NvidiaProfiler not supported")
			print("total :",model.get_number_of_parameters(),"parameters")
			print(model.run(input_file))
			print("\\======================================/")
			print("")
			
	else :
		print("")
		print("No arguments found")
		print("Usage: python greenAI.py paths/to/model/files")
		print("")

