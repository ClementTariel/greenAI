# greenAI

## APIs to measure the energy consumption of some image classification Convolutional Neural Network model

This project provides an API to run some general functions of a CNN model based only on its save file and an API to gather energy measurements during the execution of a script.

- CNN API :
	Support ONNX, PyTorch and TensorFlow models with a square image input
	
	Supported functions on a model :
	- count of the number of parameters of the model
	- run the model in inference mode		
	- measure the average energy consumption during inference mode (need the Energy Profiler API to do so)
	- training (except for ONNX)
	- measure the average energy consumption during training (need the Energy Profiler API to do so)

- Energy Profiler API :
	Need the absence of background process to avoid interference (else the background process will use the computer resources and increase the energy consumption)
	
	Supported features :
	- measures the energy consumed during the execution of a given function
	- measures the energy consumed in a context (or between a start and a stop point)

Additionnaly webconnection.py can give an eco-friendliness grade based on an other project (see https://github.com/PabloGamiz/) by sending requests to https://pablogamiz.pythonanywhere.com/
	
## how to install and use

The project have been to designed to be supported on Linux machines.
The requirements for the project are in requirement.txt

example of the use of the CNN model API :
```
path_to_the_model = "path/to/the/model.onnx"

factory = DLModelFactory()
# The predefined format are :
# ".pt", ".pt" saved with jit, ".onnx", ".h5" and TensorFlow SavedModels (ending with "/")
factory.register_predefined_formats()

model_constructor = factory.get_model(path_to_the_model)
model = model_constructor(path_to_the_model,enable_GPU=False)
n = model.get_number_of_parameters()
print("My model has ",n,"parameters.")
output = model.run("path/to/an/image")
```

example of use of the Energy Profiler API :
```
def fib(n,show_every_step=False):
	a = 1
	b = 1
	while n > 1 :
		a = a+b
		b = a-b
		n = n-1
	return a

delay_between_measures = 0.1  # in seconds
profiler = PyJoulesProfiler(delay_between_measures)
n = 10**8
total_time, energy_consumed, result = profiler.evaluate(fib,n,show_every_step=True)
print("The result is",result,", it took",total_time,"seconds and consumed",energy_consumed,profiler.get_unit())
```

Note that to use most of the profilers you will need to run your script with root privileges.

The greenAI.py script can be used as a test (to make sure there are no import errors for example).

example of use of webconnection.py :

`$ python3 webconnection.py script_to_evaluate.py [arg1] [arg2] ...`

The help is also available with `$ python3 webconnection.py -h`

## how to add new model type and new profilers

Each API has a parent abstract class, to add a new model type or a new profiler you can just create a new class with the abstract class as parent.

example of use of a new model type :
```
from dllibrary import DLLibrary

def MyCustomModelClass(DLLibrary):
	# implement the class
	# ...

factory = DLModelFactory()
factory.register_format(".custom_extension",MyCustomModelClass)

path_to_the_model = "path/to/the/model.custom_extension"

model_constructor = factory.get_model(path_to_the_model)
model = model_constructor(path_to_the_model)

# Use the model

```

## known issues

- For some PyTorch models a reference to the class of the model is needed to build it. The class has to be imported because if it is directly implemented in the code making some profiling tests it may crash because of the way PyTorch handles subprocesses.
- When a profiler stops measuring energy consumption there is a delay before it actually stops, because the delay between two measures always divides the total measuring time (for example if the profiler measures energy consumption every 5s and that the measured function stops after 57s, there will be a 3s delay to reach 60s because 57 is not a multiple of 5). If several profilers are used in parallel with the same delay between measures, it also causes each new profiler to make one more measure than the previous one.
- training is not supported for ONNX models


