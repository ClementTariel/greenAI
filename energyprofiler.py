#!/usr/bin/python

from abc import ABC, abstractmethod

import os
import sys
import warnings
from threading import Thread , Timer
import subprocess as sp

import numpy as np

import time



class EnergyProfiler(ABC):
	"""
	A class to profile energy consumption.

	Attributes:
		self._delay: float
		    A float to give the delay between 2 consecutive power consumption measures

	Methods:
		__init__()
		__enter__()
		__exit__()
		_background_measures()
		start()
		stop()
		get_power_profile
	"""

	def __init__(self,delay):
		"""
		Initialisation.

		Store the delay to use between 2 consecutive power consumption measures

		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		self._energy_consumption = 0
		self._delay = delay
		if self._delay <= 0:
			raise ValueError("delay must be > 0 !")
		self._profiling = False

	def _background_measures(self):
	    """
		This function print the energy profile every self._delay secs.

		while self._profiling is True this function keeps calling itself
		and print the energy profile measured.
		"""
	    if (self._profiling):
		    Timer(self._delay, self._background_measures).start()
		    power_measure = self.get_power_profile()
		    self._power_sample.append(power_measure)
		    self._timestamps.append(time.time())
		    print(power_measure)

	def __enter__(self):
		"""Start the profiling"""
		self.start()

	def __exit__(self, exc_type,exc_value, exc_traceback):
		"""Stop the profiling"""
		self.stop()

	def start(self):
		"""Launches background mesuring of energy consumption."""
		self._energy_consumption = 0
		self._power_sample = []
		self._timestamps = []
		self._profiling = True
		self._background_measures()

	def stop(self):
		"""Stops background mesuring of energy consumption."""
		self._profiling = False
		self._energy_consumption = np.trapz(self._power_sample, self._timestamps)
		return self._energy_consumption

	@abstractmethod
	def get_power_profile(self):
		"""overridden by subclass"""
		pass

class NvidiaProfiler(EnergyProfiler):

	def __init__(self,delay):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		super().__init__(delay)

	def get_power_profile(self):
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################

		""""""
		output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
		#COMMAND = "nvidia-smi --query-gpu=power.draw --format=csv"
		COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
		try:
		    memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
		# print(memory_use_values)
		return memory_use_values[0]

