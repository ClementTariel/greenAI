#!/usr/bin/python

from abc import ABC, abstractmethod

import os
import sys
import warnings
import threading
from threading import Thread , Timer, Lock
import subprocess as sp
import time

import re  # regex

import numpy as np


from codecarbon import EmissionsTracker
from codecarbon import track_emissions

import energyusage

lock = Lock()


def output_to_list(terminal_output):
	"""
	Convert the terminal output of a sp.check_output command to a list of String.

	Args:
    	terminal_output: the raw output of the subprocess.check_output of a command

    Returns:
    	list[str]: The list of all the lines of the terminal's output, in a String type
	"""
	return terminal_output.decode('ascii').split('\n')


class EnergyProfiler(ABC):
	"""
	A class to profile energy consumption.

	Attributes:
		self._delay: float
		    A float to give the delay between 2 consecutive power consumption measures

	Methods:
		__init__(delay)
		__enter__()
		__exit__()
		_background_measures()
		start()
		stop()
		get_unit()
		get_energy_profile()
		take_measures()
		battery_check()
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
		with lock:
			self._profiling = False

	def _background_measures(self):
		"""
		This function get the energy profile every self._delay secs.

		While self._profiling is True this function keeps making measures
		and add the energy measured to the total energy consumption.
		"""
		with lock:
			currently_profiling = self._profiling
		while (currently_profiling):
			time_offset = time.time()
			self._get_energy_profile_thread = Thread(target=self.get_energy_profile)
			self._get_energy_profile_thread.daemon = True
			self._get_energy_profile_thread.start()
			next_delay = self._delay+time_offset-time.time()
			if next_delay > 0:
				#print("next_delay : ",next_delay,"seconds.")
				time.sleep(next_delay)
			else:
				print(-next_delay,"seconds behind schedule")
			with lock:
				currently_profiling = self._profiling
		self._get_energy_profile_thread.join()

	def get_energy_profile(self):
		"""Make energy measures and keep track of the time"""
		self.take_measures()
		# just to be sure get_energy_profile has enough time to finish before any value is returned
		with lock:
			self._last_background_lauch_timestamp = time.time()

	def __enter__(self):
		"""Start the profiling"""
		self.start()

	def __exit__(self, exc_type,exc_value, exc_traceback):
		"""Stop the profiling"""
		self.stop()

	def battery_check(self):
		"""Check if the battery is charging and if it is, warn the user."""
		self._last_background_lauch_timestamp = time.time()
		battery_state = "unknown"
		try:
			# original command : upower -i $(upower -e | grep '/battery') | grep --color=never -E "state|to\ full|to\ empty|percentage"
			# the command is split into several parts to avoid issue when interacting with the terminal
			paths = sp.check_output("upower -e | grep 'battery'".split()).decode('ascii').split('\n')
			battery_info_path = ""
			for path in paths:
				if '/battery' in path:
					battery_info_path = path
			regex_pattern = "state|to\ full|to\ empty|percentage"
			COMMAND = "upower -i "+battery_info_path+" | grep"
			battery_use_info = sp.check_output(COMMAND.split(),stderr=sp.STDOUT)
			battery_use_info = output_to_list(battery_use_info)
			for line in battery_use_info:
				local_info = line.split()
				if (len(local_info)>1 and local_info[0] == "state:"):
					battery_state = local_info[1] 
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		print("state of the battery : ",battery_state)
		if battery_state == "charging" or battery_state == "unknown":
			warnings.warn("The current state of the battery is : "+battery_state+", that could strongly impact the energy profiling")
		
	def start(self):
		"""Launches background mesuring of energy consumption."""
		# Warn the user if the battery is charging
		self.battery_check()
		self._energy_consumption = 0
		self._last_measure_time_stamp = None
		with lock:
			self._profiling = True
		self._background_measures_thread = Thread(target=self._background_measures)
		self._background_measures_thread.daemon = True
		self._background_measures_thread.start()

	def stop(self):
		"""Stops background mesuring of energy consumption."""
		with lock:
			self._profiling = False
		# delay_before_stopping = self._last_background_lauch_timestamp + self._delay - time.time()
		# if delay_before_stopping < 0:
		# 	print("Could not stop process in time. Return"+str(delay_before_stopping)+" seconds late.")
		# 	with lock:
		# 		print("self._profiling :",self._profiling)
		# 	delay_before_stopping = 0
		# time.sleep(delay_before_stopping)
		self._background_measures_thread.join()
		return self._energy_consumption

	def get_unit(self):
		"""Return a string with the unit used"""
		return self._unit

	@abstractmethod
	def take_measures(self):
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
		self._unit = "J"

	def take_measures(self):
		#########################
		#						#
		#	Work in progress	#
		#						#
		#########################

		"""Get power profile using nvidia-smi and deduce the energy profile"""

		#COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
		print("TEMPORARY VALUE FOR TEST ONLY. THESE VALUES AREN'T THE REAL POWER MESAURES")
		COMMAND = "nvidia-smi --query-gpu=power.draw --format=csv"
		try:
		    power_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))
		    print(power_use_info)
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		power_value = power_use_info[1].split()[0]
		print(power_value)
		if power_value=="[N/A]":
			warnings.warn("nvidia-smi --query-gpu=power.draw returned 'N/A'")
			with lock:
				self._energy_consumption = "N/A"
			return
		else:
			power_measure = int(power_value)
		print(power_measure)
		with lock:
			if self._last_measure_time_stamp is None:
				self._last_measure_time_stamp = new_measure_time_stamp
			energy = power_measure * (new_measure_time_stamp - self._last_measure_time_stamp)
			self._energy_consumption += energy
		print(power_measure," W")
		print(energy," J = ",energy/(3600*1000),"kWh")
		with lock:
			print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")
			self._last_measure_time_stamp = new_measure_time_stamp


class PerfProfiler(EnergyProfiler):

	def __init__(self,delay):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		super().__init__(delay)
		self._unit = "J"
		# In case the user don't want to allow the use of perf from other programms
		# and want to use sudo to give root privileges to this command only.
		TEST_COMMAND = "perf stat -e power/energy-cores/,power/energy-ram/,power/energy-gpu/,power/energy-pkg/,power/energy-psys/ sleep 0"
		self._sudo_privilege = ""
		try:
			sp.check_output(TEST_COMMAND.split(),stderr=sp.STDOUT)
		except sp.CalledProcessError as e:
			regexp = re.compile(r'not have permission')
			result = regexp.search(str(e.output))
			if result:
				self._sudo_privilege = "sudo "
		

	def take_measures(self):
		"""Get energy profile with perf stat (need kernel.perf_event_paranoid <=0 or roor privileges)."""
		with lock:
			COMMAND = self._sudo_privilege+" perf stat -e power/energy-cores/,power/energy-ram/,power/energy-gpu/,power/energy-pkg/,power/energy-psys/ sleep "+str(self._delay)
		try:
		    energy_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[3:-2]
		    #print(energy_use_info)
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		energy = 0
		# print(energy_use_info)
		for output_log in energy_use_info:
			if "energy-pkg" in output_log or "energy-ram" in output_log:
				str_value = output_log.split("Joule")[0]
				[int_part,float_part] = str_value.split(",")
				energy += float(int_part) + float("0."+float_part)
		self._energy_consumption += energy
		print(energy," J = ",energy/(3600*1000),"kWh")
		print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")



class CodecarbonProfiler(EnergyProfiler):

	def __init__(self,delay,save_to_file=False):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		super().__init__(delay)
		self._unit = "kgCO2eq"
		self._tracker = EmissionsTracker(measure_power_secs=delay,save_to_file=save_to_file)

	def take_measures(self):
		pass

	def start(self):
		"""Launches background mesuring of energy consumption."""
		self._energy_consumption = 0
		with lock:
			self._profiling = True
		self._tracker.start()

	def stop(self):
		"""Stops background mesuring of energy consumption."""
		with lock:
			self._profiling = False
		self._energy_consumption = self._tracker.stop()
		return self._energy_consumption


class LikwidProfiler(EnergyProfiler):

	def __init__(self,delay):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		super().__init__(delay)
		self._unit = "J"
		

	def take_measures(self):
		"""Get energy consumption with likwid-powermeter."""
		with lock:
			COMMAND = "likwid-powermeter -s "+str(self._delay)+"s"
		try:
		    energy_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))
		    #print(energy_use_info)
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		energy = 0
		# last one is the time elapsed, not an energy and the one before is an empty line
		# print(energy_use_info)
		intersting_domain = False
		for output_log in energy_use_info:
			line = output_log.split(":")
			try:
				if line[0].split()[0] == "Domain":
					intersting_domain = (line[0] == "Domain PKG" or line[0] == "Domain DRAM")
				if intersting_domain and line[0] == "Energy consumed":
					energy += float(line[1].split()[0])
			except:
				pass
		with lock:
			self._energy_consumption += energy
		print(energy," J = ",energy/(3600*1000),"kWh")
		with lock:
			print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")
