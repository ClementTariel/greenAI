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

from pyJoules.energy_meter import EnergyContext
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.device.rapl_device import RaplDramDomain
from pyJoules.device.rapl_device import RaplUncoreDomain
from pyJoules.device.rapl_device import RaplCoreDomain
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

import torch.multiprocessing as mp

import multiprocessing

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
		self._energy_consumption: float
			Keeps track of the energy consumption measured
		self._profiling: bool
			Indicates whether the profiler is currently profiling or not
		self._get_energy_profile_thread: Thread
			A Thread that make a single reading in parallel of the execution
		self._last_background_lauch_timestamp: float
			A float to keep track of time and make sure energy measurements are realized regularly
			It is refreshed every time a measurement is realized
		self._background_measures_thread: Thread
			The thread that wait and launch regularly measurements

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
		evaluate(f,*args,**kwargs)
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
		self._last_measure_power = None
		with lock:
			self._profiling = True
		self._background_measures_thread = Thread(target=self._background_measures)
		self._background_measures_thread.daemon = True
		self._background_measures_thread.start()

	def stop(self):
		"""Stops background mesuring of energy consumption."""
		with lock:
			self._profiling = False
		self._background_measures_thread.join()
		return self._energy_consumption

	def get_unit(self):
		"""Return a string with the unit used"""
		return self._unit

	def evaluate(self,f,*args,**kwargs):
		"""Return a string with the unit used"""
		self.start()
		t_start = time.time()
		result = f(*args,**kwargs)
		self.stop()
		t_stop = time.time()
		delta_t = t_stop - t_start
		return delta_t,self._energy_consumption,result

	@abstractmethod
	def take_measures(self):
		"""overridden by subclass"""
		pass



class NvidiaProfiler(EnergyProfiler):

	POWER_COMMAND = "nvidia-smi --query-gpu=power.draw --format=csv"
	MEMORY_COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
	UTIL_COMMAND = "nvidia-smi --query-gpu=utilization.gpu --format=csv"
	NAME_COMMAND = "nvidia-smi --query-gpu=gpu_name --format=csv"
	MAX_POWER_COMMAND = "nvidia-smi --query-gpu=power.limit --format=csv"

	def __init__(self,delay,power_cap=75):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures.
		    power_cap: float
		    	A float to give the maximum power usage of the GPU used.
		    	It is only used if power readings are not available.

		"""
		super().__init__(delay)
		self._unit = "J"
		self.power_cap=power_cap
		self._using_util = False
		self.check_power_readings_availibility()


	def check_power_readings_availibility(self):
		"""
		Test if nvidia-smi can read power draw.

		Try to read power consumption from nvidia-smi.
		If it works set the power reading command to read power draw.
		If it does not, set the power reading command to read GPU utilization.
		From the percentage of utilization and the power cap, an estimation
		of the power usage can be made.
		Then the energy consumption can be apprimated by the average power draw
		times the duration of the execution.
		"""
		print("Making sure GPU readings are available ...")
		power_readings_available = True
		self._COMMAND = __class__.POWER_COMMAND
		try:
		    power_use_info = output_to_list(sp.check_output(self._COMMAND.split(),stderr=sp.STDOUT))
		    print(power_use_info)
		except sp.CalledProcessError as e:
			power_readings_available = False
			warnings.warn("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		if power_readings_available:
			power_value = power_use_info[1].split()[0]
			print("Readings available :",power_value)
			if power_value=="[N/A]":
				warnings.warn("'{}' returns 'N/A'".format(self._COMMAND))
				power_readings_available = False
		if not power_readings_available:
			print("power reading not available")
			self._COMMAND = __class__.UTIL_COMMAND
			self._using_util = True
			try:
			    power_use_info = output_to_list(sp.check_output(self._COMMAND.split(),stderr=sp.STDOUT))
			except sp.CalledProcessError as e:
				raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

	def take_measures(self):
		"""
		Get power profile using nvidia-smi and deduce the energy profile

		Use nvidia-smi to query power draw of the GPU.
		In case these readings are not available, it reads the GPU utilisation instead.
		By multipliying the percentage of utilisation by the maximum power usage of the GPU
		it produces an estimation of the actual power draw.
		"""

		COMMAND = self._COMMAND
		new_measure_time_stamp = time.time()
		try:
		    power_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		power_value = power_use_info[1].split()[0]
		if power_value=="[N/A]":
			warnings.warn("'{}' 'N/A'".format(COMMAND))
			with lock:
				self._energy_consumption = "N/A"
			return
		else:
			power_measure = float(power_value)
		with lock:
			if self._last_measure_time_stamp is None:
				self._last_measure_time_stamp = new_measure_time_stamp
				self._last_measure_power = power_measure
			energy = (new_measure_time_stamp - self._last_measure_time_stamp) * (self._last_measure_power + power_measure) / 2
			if self._using_util:
				energy *= self.power_cap/100
			self._energy_consumption += energy
		if self._using_util:
			print("~",power_measure*self.power_cap/100," W")
		else:
			print(power_measure," W")
		print(energy," J = ",energy/(3600*1000),"kWh")
		with lock:
			print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")
			self._last_measure_time_stamp = new_measure_time_stamp
			self._last_measure_power = power_measure

	def get_device_name(self):
		"""
		Get the name of the GPU

		Uses a command of nvidia-smi to get the name of the GPU

		Returns:
			str: The name of the GPU
		"""
		try:
			output = sp.check_output(__class__.NAME_COMMAND.split(),stderr=sp.STDOUT)
			return output.decode('ascii').split('\n')[1]
		except sp.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		
	def get_max_power(self):
		"""
		Get the maximum power of the GPU

		Uses a command of nvidia-smi to get the maximum power of the GPU

		Returns:
			float: The maximum power of the GPU
		"""
		try:
			output = sp.check_output(__class__.MAX_POWER_COMMAND.split(),stderr=sp.STDOUT)
			max_power = output.decode('ascii').split('\n')[1]
		except sp.CalledProcessError as e:
			raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		if max_power is None or max_power=="" or "N/A" in max_power:
			return self.power_cap
		return float(max_power)


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
		# In case the user don't want to allow the use of perf from other program
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
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		energy = 0
		for output_log in energy_use_info:
			if "energy-pkg" in output_log or "energy-ram" in output_log:
				str_value = output_log.split("Joule")[0]
				[int_part,float_part] = str_value.split(",")
				energy += float(int_part) + float("0."+float_part)
		self._energy_consumption += energy
		print(energy," J = ",energy/(3600*1000),"kWh")
		print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")

	def get_device_name():
		INFO_COMMAND = "cat /proc/cpuinfo"
		cat = sp.Popen(INFO_COMMAND.split(), stdout=sp.PIPE)  # Change stdout to PIPE
		output = sp.check_output("grep name".split(), stdin=cat.stdout)  # Get stdin from cat.stdout
		return output.decode('ascii').split('\n')[0].split(": ")[1]


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
		self._energy_ram = 0

	def take_measures(self):
		"""Get energy consumption with likwid-powermeter."""
		with lock:
			COMMAND = "likwid-powermeter -s "+str(self._delay)+"s"
		try:
		    energy_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
		energy = 0
		energy_ram = 0
		intersting_domain = False
		for output_log in energy_use_info:
			line = output_log.split(":")
			try:
				if line[0].split()[0] == "Domain":
					intersting_domain = (line[0] == "Domain PKG" or line[0] == "Domain DRAM")
					is_RAM = (line[0] == "Domain DRAM")
				if intersting_domain and line[0] == "Energy consumed":
					energy += float(line[1].split()[0])
					if is_RAM:
						energy_ram += float(line[1].split()[0])
			except:
				pass
		with lock:
			self._energy_consumption += energy
			self._energy_ram += energy_ram
		print(energy," J = ",energy/(3600*1000),"kWh")
		with lock:
			print("total : ",self._energy_consumption," J = ",self._energy_consumption/(3600*1000),"kWh")

	def get_energy_ram(self):
		"""
		Get the energy consumption of the RAM only
		
		Returns:
			float: the energy consumption of the RAM only
		"""
		return self._energy_ram

	def get_device_name(self):
		"""
		Get the name of the CPU

		Uses the data at /proc/cpuinfo to get the name of the CPU

		Returns:
			str: The name of the CPU
		"""
		INFO_COMMAND = "cat /proc/cpuinfo"
		cat = sp.Popen(INFO_COMMAND.split(), stdout=sp.PIPE)  # Change stdout to PIPE
		output = sp.check_output("grep name".split(), stdin=cat.stdout)  # Get stdin from cat.stdout
		return output.decode('ascii').split('\n')[0].split(": ")[1]

	def get_max_power(self):
		"""
		Get the maximum power of the CPU

		Uses a Likwid command to get the maximum power of the CPU

		Returns:
			float: The maximum power of the CPU
		"""
		COMMAND = "likwid-powermeter -i"
		try:
			power_max = sp.check_output(COMMAND.split(),stderr=sp.STDOUT).decode('ascii').split('domain PKG')[1].split('Maximum Power:')[1].split()[0]
			return float(power_max)
		except sp.CalledProcessError as e:
		    raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

	def get_ram_size(self):
		"""
		Get the size of the memory

		Returns:
			float: The size of the memory in Gb
		"""
		INFO_COMMAND = "free -m"
		cat = sp.Popen(INFO_COMMAND.split(), stdout=sp.PIPE)  # Change stdout to PIPE
		output = sp.check_output("grep Mem".split(), stdin=cat.stdout)  # Get stdin from cat.stdout
		return float(output.decode('ascii').split('\n')[0].split(": ")[1].split()[0])/1000


class PyJoulesProfiler(EnergyProfiler):

	def __init__(self,delay,cpu_domains=[0],ram_domains=[0],gpu_domains=[]):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures

		"""
		super().__init__(delay)
		self._unit = "uJ"
		self._id = int(time.time()*10**9)
		self._csv_temp_file_name  = '.temp_result'+str(self._id)+'.csv'
		self._csv_handler = CSVHandler(self._csv_temp_file_name)
		self._domains = []
		for i in cpu_domains:
			self._domains.append(RaplPackageDomain(i))
		for i in gpu_domains:
			self._domains.append(RaplUncoreDomain(i))
		for i in ram_domains:
			self._domains.append(RaplDramDomain(i))
		
	def take_measures(self):
		"""Do nothing because the measures are taken automatcally with pyjoules"""
		print("energy profiling...")

	def _background_measures(self):
		"""
		This function get the energy profile every self._delay secs.

		While self._profiling is True this function keeps making measures
		and add the energy measured to the total energy consumption.
		"""
		with EnergyContext(handler=self._csv_handler, domains=self._domains, start_tag='start') as ctx:
			measures_count = 0
			with lock:
				currently_profiling = self._profiling
			while (currently_profiling):
				time_offset = time.time()
				ctx.record(tag='measure_'+str(measures_count))
				self._get_energy_profile_thread = Thread(target=self.get_energy_profile)
				self._get_energy_profile_thread.daemon = True
				self._get_energy_profile_thread.start()
				next_delay = self._delay+time_offset-time.time()
				if next_delay > 0:
					time.sleep(next_delay)
				else:
					print(-next_delay,"seconds behind schedule")
				with lock:
					currently_profiling = self._profiling
				measures_count += 1
			ctx.record(tag='not_profiling')
			self._get_energy_profile_thread.join()

	def stop(self):
		"""Stops background mesuring of energy consumption."""
		with lock:
			self._profiling = False
		self._background_measures_thread.join()
		self._csv_handler.save_data()
		energy = 0
		first_line = True
		with open(self._csv_temp_file_name, encoding = 'utf-8') as f:
			for line in f:
				if first_line:
					first_line = False
				else:
					output = line.split(";")
					print(output[1]+" : "+output[3]+" "+self._unit)
					energy += float(output[3])
		self._energy_consumption = energy
		os.remove(self._csv_temp_file_name)
		return self._energy_consumption


class EnergyUsageProfiler(EnergyProfiler):

	def __init__(self,delay):
		"""
		Initialisation.

		Call the initialization of the parent class.
		
		Args:
			delay: float
		    	A float to give the delay between 2 consecutive power consumption measures
		    	It is not used due to the architecture of energyusage but it need to be given
		    	to avoid a crash and to give a time reference to threads

		"""
		super().__init__(delay)
		self._unit = "kWh"

	def take_measures(self):
		pass

	def evaluate(self,f,*args,**kwargs):
		"""
		Call energyusage.evaluate to get energy consumption of f

		Args:
			f: function
		    	The function to evaluate
		"""
		try:
		    mp.set_start_method('spawn')
		except RuntimeError:
		    pass
		try:
		    multiprocessing.set_start_method('spawn', force=True)
		except RuntimeError:
		    pass
		time_used, energy, result_of_f = energyusage.evaluate(f,*args,**kwargs,energyOutput=True)
		print(energy," kWh = ",energy*(3600*1000)," J")
		self._energy_consumption = energy
		return time_used, energy, result_of_f

