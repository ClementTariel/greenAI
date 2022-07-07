import requests
import webbrowser
import json
import time
import os
import sys

from energyprofiler import NvidiaProfiler, LikwidProfiler

# A consumption of 0.3725 W per GB of RAM is assumed
POWER_PER_GIGA_OF_RAM = 0.3725

# pablo's website (see https://github.com/PabloGamiz/)
base_url = "https://pablogamiz.pythonanywhere.com/"
# The url used for different requests
get_cpus_request = base_url+"cpus/CPU/"
get_gpus_request = base_url+"cpus/GPU/"
post_new_device_request = base_url+"calculationData/"
delete_device_with_wrong_specs_request = base_url+"softwareCalculationData/"
get_grade_request = base_url+"classificationDataC/"

def get_cpus():
	"""
	A request to get the list of the data about CPUs supported by pablo's website

	Returns:
		list[dict] : a list with dictionnaries. Each dictionnary contains data about one CPU
	"""
	print("get cpus available ...")
	cpus_response = requests.get(get_cpus_request)
	print("done, response : ",cpus_response)
	cpus = json.loads(cpus_response.text)
	return cpus

def get_gpus():
	"""
	A request to get the list of the data about GPUs supported by pablo's website

	Returns:
		list[dict] : a list with dictionnaries. Each dictionnary contains data about one GPU
	"""
	print("get gpus available ...")
	gpus_response = requests.get(get_gpus_request)
	print("done, response : ",gpus_response)
	gpus = json.loads(gpus_response.text)
	return gpus

def post_new_device(device_name,device_type,power_cap):
	"""
	A request to add data about a new device (CPU or GPU) on pablo's website

	Args:
    	device_name (str): The name of the device
    	device_type (str): The type of the device ("CPU" or "GPU")
    	power_cap (float): The maximum power the device can use

    Returns:
		<class 'requests.models.Response'>: The response of the web request
	"""
	properties = {
		"object":"Sistema software",
		"value_type":device_name,
		"object_type":device_type,
		"value1":str(power_cap),
		"value2":"0.0",
		"value3":"0.0"
	}
	response = requests.post(post_new_device_request, data=properties)
	return response

def delete_device_with_wrong_specs(device_name,device_type):
	"""
	A request to remove data about a device on pablo's website

	Args:
    	device_name (str): The name of the device
    	device_type (str): The type of the device ("CPU" or "GPU")

    Returns:
		<class 'requests.models.Response'>: The response of the web request
	"""
	response = requests.delete(delete_device_with_wrong_specs_request+"Sistema software/"+device_type+"/"+device_name)
	return response

def get_grade(value):
	"""
	A request to remove data about a device on pablo's website

	Args:
    	value (float): The computed value for the evaluated category

    Returns:
		<class 'requests.models.Response'>: The response of the web request
											(`json.loads(response.text)["calification"]` to get the grade)
	"""
	response = requests.get(get_grade_request+"1"+"/"+str(value))
	return response

def utilisation_percentage(energy,duration,power_max):
	"""
	Return the average percentage of use

	Uses the energy consumption and duration values provided to estimate an average power consumption.
	The ratio between this average power consumption and the maximum power of the device gives the
	average percentage of use of the considered device during the test realized.

	Args:
    	energy (float): The energy consumption measured (in Joules)
    	duration (float): The duration of the test in which energy consumption was measured (in seconds)
    	power_max (float): The maximum power the device tested can use

    Returns:
		float: The average pourcentage of use of the device during the test
	"""
	return 100*energy/(duration*power_max)


if __name__ == '__main__':

	pue = 2  # average Power Usage Effectiveness

	# arbitrarily chosen
	delay_between_measures = 1
	safe_delay = 1
	calbration_time = 1


	gpu_profiler = NvidiaProfiler(delay_between_measures)
	cpu_profiler = LikwidProfiler(delay_between_measures)

	print("wait for",safe_delay,"seconds then measure idle consumption for",calbration_time,"seconds.")
	time.sleep(safe_delay)

	gpu_start_time = time.time()
	gpu_profiler.start()
	time.sleep(delay_between_measures)  # To share the delay caused by overlapping
	cpu_start_time = time.time()
	cpu_profiler.start()

	#
	# 	mesure idle power consumption
	#
	time.sleep(calbration_time)

	gpu_energy = gpu_profiler.stop()
	gpu_stop_time = time.time()
	cpu_energy = cpu_profiler.stop()
	cpu_stop_time = time.time()
	ram_energy = cpu_profiler.get_energy_ram()
	
	# The conversion works because energy is in J for both of the profilers
	gpu_idle_power = gpu_energy/(gpu_stop_time-gpu_start_time)
	ram_idle_power = ram_energy/(cpu_stop_time-cpu_start_time)
	cpu_idle_power = cpu_energy/(cpu_stop_time-cpu_start_time) - ram_idle_power
	total_idle_power = cpu_idle_power + gpu_idle_power + ram_idle_power

	#
	# 	mesure effective power consumption
	#

	gpu_start_time = time.time()
	gpu_profiler.start()
	time.sleep(delay_between_measures)  # To share the delay caused by overlapping
	cpu_start_time = time.time()
	cpu_profiler.start()

	script_found = False
	if len(sys.argv) > 1:
		launch_script = "python3 "
		for i in range(1,len(sys.argv)):
			if not(";" in sys.argv[i] or "|" in sys.argv[i] or "&" in sys.argv[i]):  # to make sure only the python script is executed
				if sys.argv[i].endswith(".py"):
					script_found = True
				if script_found:
					launch_script += str(sys.argv[i])+" "
		os.system(launch_script)
	if not script_found:
		# some code here ...
		def stupid_fib(n):
			if n<2:
				return 1
			return stupid_fib(n-1)+stupid_fib(n-2)
		n=37
		print("fib("+str(n)+") = ",stupid_fib(n))

	gpu_energy = gpu_profiler.stop()
	gpu_stop_time = time.time()
	cpu_energy = cpu_profiler.stop()
	cpu_stop_time = time.time()
	ram_energy = cpu_profiler.get_energy_ram()

	#
	# 	Compute metrics
	#

	# The conversion works because energy is in J for both of the profilers
	gpu_power = gpu_energy/(gpu_stop_time-gpu_start_time)
	ram_power = ram_energy/(cpu_stop_time-cpu_start_time)
	cpu_power = cpu_energy/(cpu_stop_time-cpu_start_time) - ram_power
	total_power = cpu_power + gpu_power + ram_power
	
	cpu_max_power = float(cpu_profiler.get_max_power())
	gpu_max_power = float(gpu_profiler.get_max_power())
	ram_max_power = cpu_profiler.get_ram_size() * POWER_PER_GIGA_OF_RAM
	total_power_max = cpu_max_power + gpu_max_power + ram_max_power

	total_efficiency = (total_power - total_idle_power) * pue * 0.001
	total_optim = (total_power - total_idle_power) / total_power_max

	response_efficiency = get_grade(total_efficiency)
	response_optim = get_grade(total_optim)

	error_when_loading_results = False
	try :
		efficiency_grade = json.loads(response_efficiency.text)["calification"]
		optim_grade = json.loads(response_optim.text)["calification"]

		print("\n"+"="*20+"\n")
		print("efficiency	: "+efficiency_grade+" ("+str(total_efficiency)+")")
		print("optimisation	: "+optim_grade+" ("+str(total_optim)+")")
		print("\n"+"="*20+"\n")
	except:
		error_when_loading_results = True
		print("ERROR ")
		print("response for efficiency :",response_efficiency)
		print("response for optimisation :",response_optim)

	while error_when_loading_results and input("Try again ? (y/n)") in ["y","Y","yes","Yes","YES"]:
		try :
			efficiency_grade = json.loads(response_efficiency.text)["calification"]
			optim_grade = json.loads(response_optim.text)["calification"]

			print("\n"+"="*20+"\n")
			print("efficiency	: "+efficiency_grade+" ("+str(total_efficiency)+")")
			print("optimisation	: "+optim_grade+" ("+str(total_optim)+")")
			print("\n"+"="*20+"\n")
		except:
			error_when_loading_results = True
			print("ERROR ")
			print("response for efficiency :",response_efficiency)
			print("response for optimisation :",response_optim)

	# #
	# # old code to add the devices to the website and compute the metrics necessary
	# #

	# # convert metrics

	# cpu_name = cpu_profiler.get_device_name()
	# gpu_name = gpu_profiler.get_device_name()

	# cpu_percentage = int(utilisation_percentage(cpu_energy,cpu_stop_time-cpu_start_time,cpu_max_power))
	# gpu_percentage = int(utilisation_percentage(gpu_energy,gpu_stop_time-gpu_start_time,gpu_max_power))

	# # chekc if the devices are available on the website

	# cpus = get_cpus()
	# gpus = get_gpus()

	# cpu_available = False
	# for cpu in cpus:
	# 	if cpu["value_type"] == cpu_name:
	# 		cpu_available = True
	# 		if float(cpu["value1"]) != float(cpu_max_power):
	# 			print("values do not match : ",float(cpu["value1"])," vs ",float(cpu_max_power))
	# 			print(delete_device_with_wrong_specs(cpu_name,"CPU"))
	# 			cpu_available = False
	# 		break
	# if not cpu_available:
	# 	response = post_new_device(cpu_name,"CPU",cpu_max_power)
	# 	print(response)
	# 	print(response.text)

	# gpu_available = False
	# for gpu in gpus:
	# 	if gpu["value_type"] == gpu_name:
	# 		gpu_available = True
	# 		if float(gpu["value1"]) != float(gpu_max_power):
	# 			print("values do not match : ",float(gpu["value1"])," vs ",float(gpu_max_power))
	# 			print(delete_device_with_wrong_specs(gpu_name,"GPU"))
	# 			gpu_available = False
	# 		break
	# if not gpu_available:
	# 	response = post_new_device(gpu_name,"GPU",gpu_max_power)
	# 	print(response)
	# 	print(response.text)

	# print("-"*20)
	# print('"Indica el tipus d\'objecte per al que vols realitzar el càlcul" : "Sistema software"')
	# print("-"*20)
	# print('"Quina es la CPU emprada en l\'execució" : "'+cpu_name+'"')
	# if gpu_percentage == 0:
	# 	print('"Quina es la GPU emprada en l\'execució" : "None"')
	# else:
	# 	print('"Quina es la GPU emprada en l\'execució" : "'+gpu_name+'"')
	# print('"Indica el tamany en GB de la memòria" : "0" (The consumption of the RAM is actually included with the CPU consumption)')
	# print('"Indica el valor del PUE" : "1" (1 by default but it can be changed)')
	# print('"Indica el percentatge de CPU abans de l\'execució" : "0"')
	# print('"Indica el percentatge de GPU abans de l\'execució" : "0"')
	# print('"Indica el percentatge de memòria abans de l\'execució" : "0"')
	# print('"Indica el nombre de falles total desde el deployment" : "0"')
	# print('"Indica el percentatge de CPU durant de l\'execució" : "'+str(cpu_percentage)+'" (Might be over 100% because RAM is included')
	# if gpu_percentage == 0:
	# 	print('"Indica el percentatge de GPU abans de l\'execució" : "100" (There is no GPU anyways)')
	# else:
	# 	print('"Indica el percentatge de GPU abans de l\'execució" : "'+str(gpu_percentage)+'"')
	# print('"Indica el percentatge de memòria abans de l\'execució" : "100"')
	# print('"Indica el nombre de dies des del deployment" : "1"')  # (??? You can put whatever you want I think)
	
	# url = "https://greenidcard-e3d5d.web.app/#/calculeficiencia"
	# webbrowser.open(url)


