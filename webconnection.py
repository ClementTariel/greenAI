import requests
import webbrowser
import json
import subprocess as sp
from urllib.parse import quote


from energyprofiler import NvidiaProfiler, LikwidProfiler

base_url = "https://pablogamiz.pythonanywhere.com/"
get_cpus_request = base_url+"cpus/CPU/"
get_gpus_request = base_url+"cpus/GPU/"
post_new_device_request = base_url+"calculationData/"

def post_example():
	url = "https://www.facebook.com/login/device-based/regular/login/?login_attempt=1&lwv=110"
	myInput = {'email':'mymail@gmail.com','pass':'mypaass'}
	x = requests.post(url, data = myInput)
	y = x.text
	f = open("home.html", "a")
	f.write(y)
	f.close()
	webbrowser.open('file:///root/python/home.html')

def get_cpus():
	print("get cpus available ...")
	cpus_response = requests.get(get_cpus_request)
	print("done, response : ",cpus_response)
	cpus = json.loads(cpus_response.text)
	return cpus

def get_gpus():
	print("get gpus available ...")
	gpus_response = requests.get(get_gpus_request)
	print("done, response : ",gpus_response)
	gpus = json.loads(gpus_response.text)
	return gpus

def post_new_device(device_name,device_type,power_cap):
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


if __name__ == '__main__':
	delay_between_measures = 1
	GPU_profiler = NvidiaProfiler(delay_between_measures)
	CPU_profiler = LikwidProfiler(delay_between_measures)

	cpus = get_cpus()
	cpus_names = []
	for e in cpus:
		cpus_names.append(e["value_type"])
	print(cpus_names)

	gpus = get_gpus()
	gpus_names = []
	for e in gpus:
		gpus_names.append(e["value_type"])
	print(gpus_names)

	cpu_name = CPU_profiler.get_device_name()
	cpu_max_power = CPU_profiler.get_max_power()

	gpu_name = GPU_profiler.get_device_name()
	gpu_max_power = GPU_profiler.get_max_power()

	if not cpu_name in cpus_names:
		response = post_new_device(cpu_name,"CPU",cpu_max_power)
		print(response)
		print(response.text)

	if not gpu_name in gpus_names:
		response = post_new_device(gpu_name,"GPU",gpu_max_power)
		print(response)
		print(response.text)

	# first_page = "https://greenidcard-e3d5d.web.app/#/calculeficiencia"
	# response = requests.get(first_page)
	# print(response)
	# print(response.text)


