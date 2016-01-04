
## Python and Arduino code demonstrating odour classification on the SpiNNaker neuromorphic board.

This work was conducted as part of a lab rotation taking place in the School of Informatics at the University of Sussex, UK,  under the supervision of Dr. Michael Schmuker.

## Acknowledgements:

The spiking neural network classifier used for this project, which includes the BuildAndRunClassifier.py, Classifier.py and ModellingUtils.py files, was created by Dr. Alan Diamond and the original code can be found at: https://github.com/alandiamond/spinnaker-neuromorphic-classifier

The neuromorphic platform used was the SpiNNaker spiNN-3 development board developed at the University of Manchester.

Special thanks to Prof. Thomas Nowotny, Dr. Esin Yavuz and Dr. Michael Schmuker at the University of Sussex for their help. 


## How to use:

1. make_eNose_baseline

Run this script before training and before every subsequent testing phase.
 
It does not require any input parameters.
It outputs the baseline values for each of the sensors employed.

It creates 3 folders if they don’t exist already:
	- recordings; contains all training and test recordings of all odours combined used for crossvaldidation
	- recordingsTrain; contains only the recordings used to train the network
	- recordingsTest; contains only the recordings used to test the network

The baseline values are subtracted from all the training and testing values. As these values differ from one day to the other and between different parts of the day, it is recommended to calculate the baseline before recording new data for testing. These baseline values have to be manually set in the Settings-eNoseClassifier.txt file. 

2. eNose_logger

Run this script to record data coming from the electronic nose.

If ran without input parameters, it records a number of e-nose samples and saves the data in a ‘test.csv’ file within the ‘recordings’ folder located at the MASTER_PATH specified in the Settings-eNoseClassifier.txt file.

Possible input parameters:
	- name of the logger file; specify the name in the format 0-1.csv, 1-1.csv, etc, where the first digit represents the odour class currently being recorded and the second digit is the number of the recording session.
	- name of the folder where the recorded data is saved. By default, this folder is the ‘recordings’ folder. If you to do a single training and testing of the network, change the name of the recording folder first to match the folder used to save the training data, then to the one used to save the testing data. Otherwise, record all the sessions in the default ‘recordings’ folder and divide the data into training and testing sets manually. 
	- number of times the e-nose is sampled for each recording session
	- a boolean indicating whether to do crossvalidation or not

3. run_eNose

Run this script to do crossvalidation on the whole data set or one instance of training and testing.

It uses all the parameters found in the ModelParams-eNoseClassifier.txt and Settings-eNoseClassifier.txt files. 
It calls the BuildAndRun Classifier script. 
It plots raw sensor data, input spike trains and network activity.

Possible input parameters:
	- a boolean indicating whether to do crossvalidation or not
	- a list of filenames for each distinct class recording used for plotting the raw sensor data
	- the folder in which these files are found
	- the names of the odours used for classification
By default, the list of filenames is ['0-1.csv', '1-1.csv', '2-1.csv’], found in the 'recordings' folder. The odour classes I used are ['No odour', 'Ethanol', 'Orange essential oil’]. These parameters can also be changed directly in the script. 

The following parameters can only be modified inside the ModelParams-eNoseClassifier.txt or in the Settings-eNoseClassifier.txt:

ModelParams-eNoseClassifier.txt:

'MAX_NEURONS_PER_CORE':250,
'MAX_STDP_NEURONS_PER_CORE':30,
'CORES_ON_BOARD':64,
'NUM_CLASSES':3, #number of odour classes. I included the no-odour case
'NUM_INPUT_NEURONS':3, #number of sensors used
'NUM_VR':30, #number of total VRs (in this case 10 per sensor)
'RATE_ALPHA':0.0001, #regulates spiking rate
'NETWORK_SCALE':3, #regulates network size
'CLUSTER_SIZE':3, #regulates network size
'RN_NOISE_RATE_HZ':100,
'RN_NOISE_SOURCE_POP_SIZE':6,
'WEIGHT_RATECODE_TO_CLUSTER_RN':0.4,
'MIN_DELAY_RATECODE_TO_CLUSTER_RN':1,
'MAX_DELAY_RATECODE_TO_CLUSTER_RN':30,
'WEIGHT_POISSON_TO_CLUSTER_RN':0.2,
'DELAY_POISSON_TO_CLUSTER_RN':1,
'WEIGHT_RN_PN':1.5,
'DELAY_RN_PN':1,
'WEIGHT_WTA_PN_PN':0.02,
'DELAY_WTA_PN_PN':1,
'CONNECTIVITY_WTA_PN_PN':0.40,
'STARTING_WEIGHT_PN_AN':0.01,
'DELAY_PN_AN':1,
'CONNECTIVITY_PN_AN':0.5,
'STDP_TAU_PN_AN':10.0,
'STDP_WMIN_PN_AN':0.0,
'STDP_WMAX_PN_AN':0.3,
'STDP_SCALING_PN_AN':0.01,
'WEIGHT_WTA_AN_AN':0.06,
'DELAY_WTA_AN_AN':1,
'CONNECTIVITY_WTA_AN_AN':0.5,
'WEIGHT_CLASS_ACTIVITY_TO_CLUSTER_AN':0.2,
'MIN_DELAY_CLASS_ACTIVITY_TO_CLUSTER_AN':1,
'MAX_DELAY_CLASS_ACTIVITY_TO_CLUSTER_AN':30


Settings-eNoseClassifier.txt:

'NUM_FOLDS':5, # number of folds for cross validation
'NUM_OBSERVATIONS':12, #total number of files used for training
'NUM_OBSERVATIONS_TEST':3, #number of files used for testing
#specify the 2 parameters above even in the case of crossvalidation
'SPIKE_TRAIN_LENGTH':100, #length of each Poisson train created out of one sample value
'NUM_BASELINE_SAMPLES':100, #number of samples used to calculate the baseline values
'NUM_LOG_SAMPLES':15, #number of samples taken for each recording
'OBSERVATION_EXPOSURE_TIME_MS':1500, # number of samples x length of single Poisson spike train
'NUM_REPETITIONS':1, #how many times the training set should be repeated. Be careful not to overfit
'LEARNING':True, #whether the network is in the learning or testing phase
'RECORD_POP_INPUT':False,
'RECORD_POP_NOISE_SOURCE':False,
'RECORD_POP_RN':True,
'RECORD_POP_PN':True,
'RECORD_POP_AN':True,
'CROSSVALIDATION':True, #whether to do crossvalidation or single training/testing instance
'BASELINE_FILENAME':'baseline_samples.csv', 
'BASELINE_VALUES':[2086.24,  2733.5 ,  1809.02], #change these values before each testing phase
'SERIAL_PORT':'/dev/tty.usbmodem1411', #serial port used to connect to the arduino
'SENSOR_NAMES':['Elapsed_time','Temperature','Humidity','TGS2600','TGS2602','TGS2610'], #Arduino data field names
'SPIKE_SOURCE_VR_RESPONSE_TRAIN':'VrResponse_SpikeSourceData_Train.csv', #file containing spike trains used for training the network, calculated from e-nose sample data
'SPIKE_SOURCE_VR_RESPONSE_TEST':'VrResponse_SpikeSourceData_Test.csv',#file containing spike times used for testing the network, calculated from e-nose sample data
'SPIKE_SOURCE_CLASS_ACTIVATIONS':'ClassActivation_SpikeSourceData.csv', #file containing teacher signal spike times
'CLASSIFICATION_RESULTS_PATH':'WinningClassesByObservation.csv', #file containing the labels of the winning classes
'SPIKE_COUNT_RESULTS_PATH':'WinningCountsByObservation.csv', #spike counts of the winner AN populations
'CLASS_LABELS_TRAIN':'ClassLabelsTrain.csv', #file containing the labels of the classes used for training
'CLASS_LABELS_TEST':'ClassLabelsTest.csv',#file containing the labels of the classes used for testing
'CROSSVALIDATION_LOGGER_FOLDER':'recordings’, #folder used to save all the recorded data
'TRAIN_LOGGER_FOLDER':'recordingsTrain', #folder used to save the recorded data for training
'TEST_LOGGER_FOLDER':'recordingsTest', #folder used to save the recorded data for testing
'MASTER_PATH':'/Volumes/LocalDataHD/i/il/il75/Google Drive/neuromorphic/spiNNaker/eNose/', #main path, where all the above files and the recording folders are created
'RUN_COMPLETE_FILE':'PynnRunCompleted.txt',
'CACHE_DIR’:’./cache' #path to where the PN-AN connections modified during training are saved. These weights will be loaded into the model during the testing phase. This folder is quite big, make sure you have enough space at the specified path.


4. BuildAndRunClassifier

Runs the spiking neural network.
Plot the network activity.

Input parameters:
	- ModelParams-eNoseClassifier parameters
	- Settings-eNoseClassifier parameters
	- number of the current fold. If single training/testing instance, this parameters is None.

Output:
	- percent of correct decisions

5. Classifier

Contains functions used to build the network. Called by BuildAndRunClassifier

6. ModellingUtils

Contains further modelling tools


For details regarding the spiking neural model, visit Alan Diamond’s GitHub repository at https://github.com/alandiamond/spinnaker-neuromorphic-classifier. 

For further information regarding this work, contact iulialexandralungu@gmail.com



## Troubleshooting

1. SyntaxError: EOL while scanning string literal
- check the single quotes in the ModelParams or Settings files (straight vs. curly!)
- MacOS Mavericks or later uses smart quotes which might be the culprit for this error	

2. Can not connect to board
- change the IP you use to connect to the board (add 1 at the last digit of the board’s IP)
- if you have problems with the WiFi and the Thunderbolt connections, set the service order so that the wifi goes first and then the Thunderbolt. Otherwise the computer will try to connect through the internet through SpiNNaker. Also, disable the automatic connection in the ethernet advanced 802.1x settings. 

3. Spyder cannot import SpiNNaker modules:
- add the path of the module in spyder’s PYTHONPATH

4. pip does not find the modules required to install spyNNaker:
- use pip search to find the updated name of the modules

5. The spike_io script blocks the UDP port it connects to (local address 19999).
- remove pylab.show() from the scripts. It prevents the figures from closing correctly.
- add the following to your .spynnaker.cfg file: 
[Database]
create_database = True 

6. Error when trying to build the classifier using the sensor data set (spinn_front_end_common.utilities.exceptions.ConfigurationException: Invalid neuron id in postsynaptic population 9)
- don’t use old saved weights, incompatible with the current model you are trying to run. Make sure you are training a new instance of the network if you have made changes to the neural model.

7. time_stamp_in_ticks = int((timeStamp * 1000.0) /
TypeError: can't multiply sequence by non-int of type 'float'
- the spike times are strings and not integers



## Resources for getting started:

1. SpiNNaker software can be found at  http://spinnakermanchester.github.io/
2. Instructions are found at https://github.com/SpiNNakerManchester/SpiNNakerManchester.github.io/tree/master/2015.004.LittleRascal
3. Link that may be useful for playing with sensors: http://www.instructables.com/id/How-To-Smell-Pollutants/
4. https://www.teachengineering.org/view_activity.php?url=collection/cub_/activities/cub_air/cub_air_lesson09_activity3.xml 
5. Sensor evaluation and comparison: http://www.takingspace.org/evaluating-low-cost-gas-sensors/ 
6. Arduino code for a project that uses many sensors together: https://github.com/empierre/arduino 
7. Sensor basics: http://playground.arduino.cc/Main/MQGasSensors 
8. https://en.m.wikipedia.org/wiki/Electrochemical_gas_sensor
9. http://playground.arduino.cc/Main/MQGasSensors#wiring
