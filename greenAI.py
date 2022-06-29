#!/usr/bin/env python3

from abc import ABC, abstractmethod

import os
import sys
import warnings
from threading import Thread , Timer
import subprocess as sp

import numpy as np

import re  # regex

import time

from dllibrary import *
from energyprofiler import *


# Execute when the module is not initialized from an import statement.
if __name__ == '__main__':

	# import a model class to test a .pt file
	from pytorchtestmodel import Net

	factory = DLModelFactory()
	factory.register_predefined_formats()

	enable_GPU = False
	model_names = []
	for i in range(1,len(sys.argv)):
		if sys.argv[i][0] == "-":
			if "g" in sys.argv[i]:
				print("enable_GPU")
				enable_GPU = True
		else:
			model_names.append(sys.argv[i])

	if (len(model_names) > 0):  # Check if there is at least one file to analize
		print ('Number of models:', len(model_names), '.')
		print("")
		for k in range(len(model_names)):
			#########################
			#						#
			#	Work in progress	#
			#						#
			#########################

			print ('model :', model_names[k])

			# Check if the given file is supported
			try :
				model_constructor = factory.get_model(str(model_names[k]))
			except ValueError:
				print("unsupported file format")
				print("")
				continue

			# just to test .pt model with constructor in argument
			try:
				model = model_constructor(str(model_names[k]),model_constructor=Net,enable_GPU=enable_GPU)
			except TypeError:
				model = model_constructor(str(model_names[k]),enable_GPU=enable_GPU)
			print("/======================================\\")
			
			# random image
			input_file = "../images/glacier.jpg"

			test_duration = 12#0
			delay_between_measures = 5
			safe_delay = 6#0

			print("start test(s) of ",test_duration," seconds (+ potential additionnal time depending on the delay between 2 measures)")
			
			# a bunch of profilers
			try:
				nb_of_steps = model.train("","../training/small_data_set/train","../training/small_data_set/test")
				print("trained in ",nb_of_steps,"steps")
			except:
				print("training not supported")
			
			# pb with nvidia-smi power.draw
			try:
				profiler = NvidiaProfiler(delay_between_measures)
				print("NvidiaProfiler")
				print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file),profiler.get_unit()," per run")
			except :
				print("NvidiaProfiler not supported")

			profiler = EnergyUsageProfiler(delay_between_measures)
			print("EnergyUsageProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file,safe_delay=safe_delay),profiler.get_unit()," per run")
			print("training footprint :",model.training_energy_consumption(profiler,test_duration,"../training/small_data_set/train","../training/small_data_set/test"),profiler.get_unit()," per run")
			
			profiler = PyJoulesProfiler(delay_between_measures)
			print("PyJoulesProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file,safe_delay=safe_delay),profiler.get_unit()," per run")
			print("training footprint :",model.training_energy_consumption(profiler,test_duration,"../training/small_data_set/train","../training/small_data_set/test"),profiler.get_unit()," per run")
			profiler = LikwidProfiler(delay_between_measures)
			print("LikwidProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file,safe_delay=safe_delay),profiler.get_unit()," per run")
			print("training footprint :",model.training_energy_consumption(profiler,test_duration,"../training/small_data_set/train","../training/small_data_set/test"),profiler.get_unit()," per run")
			profiler = CodecarbonProfiler(delay_between_measures,save_to_file=False)
			print("CodecarbonProfiler (use a tracker from codecarbon)")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file,safe_delay=safe_delay),profiler.get_unit()," per run")
			print("training footprint :",model.training_energy_consumption(profiler,test_duration,"../training/small_data_set/train","../training/small_data_set/test"),profiler.get_unit()," per run")
			profiler = PerfProfiler(delay_between_measures)
			print("PerfProfiler")
			print("footprint :",model.inference_energy_consumption(profiler,test_duration,input_file,safe_delay=safe_delay),profiler.get_unit()," per run")
			print("training footprint :",model.training_energy_consumption(profiler,test_duration,"../training/small_data_set/train","../training/small_data_set/test"),profiler.get_unit()," per run")

			# count the number of parameters
			print("total :",model.get_number_of_parameters(),"parameters")

			# get the output vector of an inference run
			print(model.run(input_file))
			print("\\======================================/")
			print("")
			
	else :
		print("")
		print("No arguments found")
		print("usage: python greenAI.py [option] paths/to/model/files")
		print("Options:")
		print("    -g : allow gpu usage")

