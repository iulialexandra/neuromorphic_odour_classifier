#author Iulia-Alexandra Lungu (iulialexandralungu@gmail.com)

import os
import fnmatch
import pdb
import numpy as np
import random
import pandas as pd
import copy
import matplotlib.pyplot as plt
import ModellingUtils as utils
import matplotlib.patches as mpatches
import BuildAndRunClassifier as eNoseClassifier
import make_eNose_baseline
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

#pdb.set_trace()

colors = [ '#F7E7BD',#cream
           'darkblue',
           '#FF9900',#orange
           '#55bb33', #greenish
           '#5533bb', #blueish
           '#cc3333',#reddish
           '#99ff99', #greenish
           '#9999ff', #blueish
           '#227722', #greenish
           '#222277', #blueish
           'lightgreen',
           'lightblue',
           'darkgreen',
           '#A4A28B',#olive
           '#B59A73',#coffee
           '#447799'
           ]

def poisson_spike_generator(rate, spikeLengthSample, timeStep=1):
    '''Generates Poisson spike train.
    Input:
        -spiking rate
        -the length of the spike train
        -timeStep
    Output:
        -spike train, 1 for spikes, 0 for non-spikes
    '''
    times = np.arange(0, spikeLengthSample, timeStep)	# a vector with each time step	
    vt = [random.random() for i in times]
    #print vt
    #print rate*timeStep
    spikes = (rate*timeStep) > vt
    return spikes


def sensor_spike_train(filePath, fileName, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues):
    """ Creates Poisson spike trains for each sensor for one recording.
    Input:
        -path of .csv file containing the arduino data for one recording session
        -name of .csv file containing the arduino data for one recording session
        -number of times the e-nose was sampled during one recording session
        -number of input neurons (= number of sensors)
        -number of virtual receptors
        -length of Poisson process for each sample
        -alpha parameter regulating spiking rate
        -baseline values for each sensor
    Ouput:
        -spike times matrix
    """
    with open(os.path.join(filePath, fileName), 'rb') as logFile:
        samples = pd.read_csv(logFile)
        #for each sensor, create a Poisson spike train using all the samples
        #and a 100ms long Poisson spike train for each sample
        spikeTrains = np.zeros((nrVR, nrSamples*spikeLengthSample))
        for sample in range(nrSamples):
            for idx, neuron in enumerate(range(3, 3+nrInputNeurons)):
                    normalizedVal = max(0, samples.iloc[sample, neuron]-baselineValues[idx])
                    rate = alpha*(normalizedVal/5.)
                    #print rate
                    for vr in range(nrVR/nrInputNeurons):
                        spikeTrains[idx*(nrVR/nrInputNeurons)+vr, sample*spikeLengthSample:(sample+1)*\
                             spikeLengthSample] = poisson_spike_generator(rate, spikeLengthSample)      
        return spikeTrains
                 
                 
def concatenate_recordings(logPath, fileList, outFileName, odourClassesRecorded, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues):
    """Concatenates spike times vectors for all recording session.
    Input:
        -path of .csv files containing the arduino data for the desired recording sessions
        -name of .csv files containing the arduino data for the desired recording session
        -name of the output file where the concatenated spike times are saved
        -class labels for the recordings to be concatenated
        -number of times the e-nose was sampled during one recording session
        -number of input neurons (= number of sensors)
        -number of virtual receptors
        -length of Poisson process for each sample
        -alpha parameter regulating spiking rate
        -baseline values for each sensor
    Ouput:
        -saves the concatenated spike times matrices to file
    """
    spikeTrainsRec = np.zeros((nrVR, nrSamples*spikeLengthSample*len(fileList)))
    for idx, fileName in enumerate(fileList):
        spikeTrainsRec[:, nrSamples*spikeLengthSample*idx:nrSamples*spikeLengthSample*(idx+1)] = \
            sensor_spike_train(logPath, fileName, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues)

    with open(outFileName, 'wb') as csvfile:
        #timesWriter = csv.writer(csvfile, dialect = 'excel', delimiter=',', quoting = csv.QUOTE_ALL)
        for neuron in range(nrVR):
            spikeTimes = np.nonzero(spikeTrainsRec[neuron, :])[0]
            np.savetxt(csvfile, [spikeTimes], delimiter=',', fmt = '%u')


def make_class_activation_spikes(classActivationsFile, odourClassesRecorded, maxSimTime):
    '''Saves the teaching signal for all desired recordings concatenated to file
    Input:
        -the name of the file where the teacher spike times are saved
        -the class labels for the desired recordings 
        -maximum simulation time for each recording (number of samples for each recording) x (length of Poisson spike train for each sample)
    Output:
        -saves the teacher signal spikes to file
    '''
    with open(classActivationsFile, 'wb') as csvfile:
        #classActivations = csv.writer(csvfile)
        classes = list(np.unique(odourClassesRecorded))
        spikes = [[]] * len(classes)
        for idx, recClass in enumerate(odourClassesRecorded):
            spikeTimes = list(np.arange(idx*maxSimTime + 1, (idx+1)*maxSimTime, 3))
            ext = copy.copy(spikes[classes.index(recClass)])
            ext.extend(spikeTimes)
            spikes[classes.index(recClass)]=ext
            #classActivations.writerow(spikeTimes)
        for cls in classes:
            np.savetxt(csvfile, [spikes[classes.index(cls)]], delimiter=',', fmt = '%u')
        #print spikeTimes


def make_labels_file(labelsFilename, odourClassesRecorded):
    '''Saves the labels for the desired recordings to file.
    Input:
        -the name of the file used to save the labels
        -the labels of the desired recordings
    Output:
        -saves the labels to file
    '''
    with open(labelsFilename, 'wb') as csvfile:
        #classLabels= csv.writer(csvfile, dialect = 'excel', delimiter=',', quoting = csv.QUOTE_ALL)
        np.savetxt(csvfile, [odourClassesRecorded], delimiter=',', fmt = '%u')


def cross_validate(paramsClassifier, settingsClassifier, baselineValues):
    '''Performs crossvalidation on the data set.
    Input:
        -classifier parameters
        -classifier settings values
        -sensor baseline values
    Output:
        -crossvalidation network performance scores vector
    '''
    
    nrInputNeurons = paramsClassifier['NUM_INPUT_NEURONS']
    nrVR = paramsClassifier['NUM_VR']
    nrClasses = paramsClassifier['NUM_CLASSES']
    alpha = paramsClassifier['RATE_ALPHA']
    
    logFolder = settingsClassifier['CROSSVALIDATION_LOGGER_FOLDER'] 
    nrFolds = settingsClassifier['NUM_FOLDS']
    masterPath = settingsClassifier['MASTER_PATH']
    spikeSourceVRTrain = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TRAIN']
    spikeSourceVRTest = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TEST']
    classLabelsTrainFile = settingsClassifier['CLASS_LABELS_TRAIN']
    classLabelsTestFile = settingsClassifier['CLASS_LABELS_TEST']
    classActivationsFile = settingsClassifier['SPIKE_SOURCE_CLASS_ACTIVATIONS']
    nrSamples = settingsClassifier['NUM_LOG_SAMPLES']
    spikeLengthSample = settingsClassifier['SPIKE_TRAIN_LENGTH']


    maxSimTime = nrSamples*spikeLengthSample
    logPath = os.path.join(masterPath, logFolder)  
    
    files = os.listdir(logPath)
    fileList = [[]] * nrClasses

    for cls in range(nrClasses):
        pattern = str(cls)+'-*.csv'
        tempFileList = [name for name in files if fnmatch.fnmatch(name, pattern)]
        fileList[cls] = tempFileList
    
    fileArray = np.array(fileList)
    nrTotObs = np.shape(fileArray)[1]
    scores = np.zeros((1, nrFolds))
    
    for fold in range(nrFolds):
        indices = range(0, nrTotObs)
        indxTest = range(nrTotObs/nrFolds*fold, nrTotObs/nrFolds*(fold+1))
        indxTrain = np.setdiff1d(indices,indxTest)
        
        trainFiles = np.ravel(fileArray[:, indxTrain])
        testFiles = np.ravel(fileArray[:, indxTest])
        
        np.random.shuffle(trainFiles)
        np.random.shuffle(testFiles)
        
        odourClassesTrain = [int(fileName[0]) for fileName in trainFiles]
        odourClassesTest = [int(fileName[0]) for fileName in testFiles]
       
        concatenate_recordings(logPath, trainFiles, spikeSourceVRTrain, \
            odourClassesTrain, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues)
        concatenate_recordings(logPath, testFiles, spikeSourceVRTest, \
            odourClassesTest, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues)
        
        make_class_activation_spikes(classActivationsFile, odourClassesTrain, maxSimTime)
        
        make_labels_file(classLabelsTrainFile, odourClassesTrain)
        make_labels_file(classLabelsTestFile, odourClassesTest)
        
        settingsClassifier['NUM_OBSERVATIONS'] = len(odourClassesTrain)
        settingsClassifier['NUM_OBSERVATIONS_TEST'] = len(odourClassesTest)

        settingsClassifier['LEARNING'] = True
        eNoseClassifier.runClassifier(paramsClassifier, settingsClassifier, fold)
        
        print 'Training completed'
        raw_input("Press Enter to proceed to testing...")
        
        settingsClassifier['LEARNING'] = False
        scores[0, fold] = eNoseClassifier.runClassifier(paramsClassifier, settingsClassifier, fold)
        
        print 'Testing for fold ' + str(fold) + ' completed'
        
        raw_input("Press Enter to continue with the next fold...")
    return scores
    
        
def train_classifier(paramsClassifier, settingsClassifier, baselineValues):
    '''Performs network training.
    Input:
        -classifier parameters
        -classifier settings values
        -sensor baseline values
    Output:
        -saves the trained PN-AN connection weights to file
    '''
    nrInputNeurons = paramsClassifier['NUM_INPUT_NEURONS']
    nrVR = paramsClassifier['NUM_VR']
    spikeLengthSample = settingsClassifier['SPIKE_TRAIN_LENGTH']
    alpha = paramsClassifier['RATE_ALPHA']
    
    logFolder = settingsClassifier['TRAIN_LOGGER_FOLDER'] 
    masterPath = settingsClassifier['MASTER_PATH']
    spikeSourceVRTrain = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TRAIN']
    classLabelsTrainFile = settingsClassifier['CLASS_LABELS_TRAIN']
    classActivationsFile = settingsClassifier['SPIKE_SOURCE_CLASS_ACTIVATIONS']
    nrRepetitions = settingsClassifier['NUM_REPETITIONS']
    nrSamples = settingsClassifier['NUM_LOG_SAMPLES']

    maxSimTime = nrSamples*spikeLengthSample
    logPath = os.path.join(masterPath, logFolder)  
    
    files = os.listdir(logPath)
    pattern = '*.csv'
    fileList = [name for name in files if fnmatch.fnmatch(name, pattern)]       
    fileArray = np.ravel(fileList * nrRepetitions)
    np.random.shuffle(fileArray)
    odourClassesTrain = [int(fileName[0]) for fileName in fileArray]
    
    concatenate_recordings(logPath, fileArray, spikeSourceVRTrain, odourClassesTrain, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues)
    make_class_activation_spikes(classActivationsFile, odourClassesTrain, maxSimTime)
    make_labels_file(classLabelsTrainFile, odourClassesTrain)
    
    settingsClassifier['LEARNING'] = True
    eNoseClassifier.runClassifier(paramsClassifier, settingsClassifier, None)
    

def test_classifier(paramsClassifier, settingsClassifier, baselineValues):
    '''Performs network testing.
    Input:
        -classifier parameters
        -classifier settings values
        -sensor baseline values
    '''

    nrInputNeurons = paramsClassifier['NUM_INPUT_NEURONS']
    nrVR = paramsClassifier['NUM_VR']
    alpha = paramsClassifier['RATE_ALPHA']
    
    logFolder = settingsClassifier['TEST_LOGGER_FOLDER'] 
    masterPath = settingsClassifier['MASTER_PATH'] 
    spikeSourceVRTest = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TEST']
    classLabelsTestFile = settingsClassifier['CLASS_LABELS_TEST']
    spikeLengthSample = settingsClassifier['SPIKE_TRAIN_LENGTH']
    nrSamples = settingsClassifier['NUM_LOG_SAMPLES']

    
    logPath = os.path.join(masterPath, logFolder)  
    files = os.listdir(logPath)
    pattern = '*.csv'
    fileList = [name for name in files if fnmatch.fnmatch(name, pattern)]   
    fileArray = np.ravel(np.array(fileList))
    np.random.shuffle(fileArray)
    odourClassesTest = [int(fileName[0]) for fileName in fileArray]
    concatenate_recordings(logPath, fileArray, spikeSourceVRTest, odourClassesTest, nrSamples, nrInputNeurons, nrVR, spikeLengthSample, alpha, baselineValues)
    make_labels_file(classLabelsTestFile, odourClassesTest)
    
    settingsClassifier['LEARNING'] = False
    eNoseClassifier.runClassifier(paramsClassifier, settingsClassifier, None)
    
    
def plot_sensor_data(logPath, logFileNames, nrInputNeurons, nrSamples, fieldnames, odourNames):
    '''Plot the raw sensor data, before baseline substraction.
    Input:
        -path of the sensor data file
        -name of the sensor data file
        -number of input neurons (= number of sensors)
        -the fieldnames of the data read from the arduino
        -names of the odours used
    '''
    sensorReading = np.zeros((nrInputNeurons, len(logFileNames)*nrSamples))
    odourClasses = np.zeros(len(logFileNames))
    bckndNames =[[]]*len(logFileNames)
    
    for fileIdx, logFileName in enumerate(logFileNames):
        odourClasses[fileIdx] = logFileName[0]
        with open(os.path.join(logPath, logFileName), 'rb') as logFile:
            samples = pd.read_csv(logFile)
            for idx, neuron in enumerate(range(3, 3+nrInputNeurons)):
                sensorReading[idx, fileIdx*nrSamples:(fileIdx+1)*nrSamples] = samples.iloc[:, neuron]
                
                
    plt.figure(figsize=(20,20))
    
    markers = ['-', '-.', '--']
    for n in range(nrInputNeurons):
        plt.plot(sensorReading[n, :], label=fieldnames[n + 3], linestyle = markers[n], linewidth = 10) #the first 3 fields are not of interest for this classification
    
    handles1, labels1 = plt.gca().get_legend_handles_labels()
    first_legend = plt.legend(handles1, labels1, loc=1, prop={'size':20})


    for j, odourClass in enumerate(odourClasses):
        plt.axvspan(j*nrSamples, j*nrSamples+nrSamples, facecolor=colors[int(odourClass)], alpha=0.3)
        bckndNames[j] = mpatches.Patch(color=colors[int(odourClass)], label=odourNames[int(odourClass)])



    plt.gca().add_artist(first_legend)      
    plt.legend(handles=bckndNames, loc = 4, prop={'size':20}) 
    plt.title('Sensor data', fontsize=40)
    plt.xlabel('Samples', fontsize=40)
    plt.ylabel('Sensor voltage[mV]', fontsize=40)
    plt.tick_params(labelsize=40)
    plt.savefig('Sensor_data_plot.pdf')
    plt.close()
    

# def PCA_sensor_data(logPath, logFileNames, nrInputNeurons, nrSamples, fieldnames, odourNames, baselineValues, ax):  
#     sensorReading = np.zeros((nrInputNeurons, nrSamples, len(logFileNames))) 
#     odourClasses = np.zeros(len(logFileNames))   
    
#     for fileIdx, logFileName in enumerate(logFileNames):
#         odourClasses[fileIdx] = logFileName[0]
#         with open(os.path.join(logPath, logFileName), 'rb') as logFile:
#             samples = pd.read_csv(logFile)
#             for idx, neuron in enumerate(range(3, 3+nrInputNeurons)):
#                 sensorReading[idx, :, fileIdx] = samples.iloc[:, neuron]
                
    

#     ax.plot(sensorReading[0,:, 0], sensorReading[1,:, 0], sensorReading[2,:, 0],
#             'o', markersize=15, color=colors[0], alpha=0.5, label=odourNames[int(odourClasses[0])])
#     ax.plot(sensorReading[0,:, 1], sensorReading[1,:, 1], sensorReading[2,:, 1],
#             'o', markersize=15, alpha=0.5, color=colors[1], label=odourNames[int(odourClasses[1])])
#     ax.plot(sensorReading[0,:, 2], sensorReading[1,:, 2], sensorReading[2,:, 2],
#             'o', markersize=15, alpha=0.5, color=colors[2], label=odourNames[int(odourClasses[2])])

     
    
def plot_spike_sources(filePath, fileName, nrInputNeurons, nrVR, observationTime, totalSimulationTime, classLabels, odourNames):
    '''Plot the Poisson spike source matrix
    Input:
        -path of the spike times file
        -name of the spike times file
        -number of input neurons (= number of sensors)
        -number of virtual receptors
        -length of the Poisson spike train for each sample
        -maximum simulation time for each recording (number of samples for each recording) x (length of Poisson spike train for each sample)
        -class labels
        -names of the odours used
    '''
    bckndNames =[[]]*len(odourNames)
    
    spikeTimes = utils.readSpikeSourceDataFile(os.path.join(filePath, fileName))['spike_times']
    plt.figure(figsize=(20,20))
    for idx, line in enumerate(spikeTimes):
        for x in line:
            plt.plot(x, idx, 'ko', markersize = 2)
    for j in range(idx, nrVR):
        plt.plot(0, j, 'k,')


    for j, classLabel in enumerate(classLabels):
        plt.axvspan(j*observationTime, j*observationTime+observationTime, facecolor=colors[int(classLabel)], alpha=0.3)
        
    for idxO, odour in enumerate(odourNames):
        bckndNames[idxO] = mpatches.Patch(color=colors[idxO], label=odour)

    
    plt.legend(handles=bckndNames, loc ='best', prop={'size':20}) 
    plt.xlabel('Simulation time[ms]', fontsize=20)
    plt.ylabel('%i Virtual receptors per sensor'%(nrVR/nrInputNeurons), fontsize=20)
    plt.tick_params(labelsize=20)
    plt.title('VR spike times for classes %s'%str(classLabels), fontsize=20)
    
    
    plt.savefig(fileName+'.pdf')

    plt.close()

#---------------------------------------------------------------------------------------------------------------    
if __name__ == "__main__":
    paramsClassifier = eval(open("ModelParams-eNoseClassifier.txt").read())
    settingsClassifier  = eval(open("Settings-eNoseClassifier.txt").read())

    fieldnames = settingsClassifier['SENSOR_NAMES']    
    crossvalidation = settingsClassifier['CROSSVALIDATION'] 
    masterPath = settingsClassifier['MASTER_PATH']  
    nrInputNeurons = paramsClassifier['NUM_INPUT_NEURONS']
    nrVR = paramsClassifier['NUM_VR']

    if nrVR%nrInputNeurons != 0:
        print 'VR number mismatch. The number of virtual receptors should be \
                a multiple of the number of sensors(input neurons)'
        pass
    
    spikeSourceVRTrain = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TRAIN']
    spikeSourceVRTest = settingsClassifier['SPIKE_SOURCE_VR_RESPONSE_TEST']
    
    classLabelsTrain = settingsClassifier['CLASS_LABELS_TRAIN']
    classLabelsTest = settingsClassifier['CLASS_LABELS_TEST']
    
    spikeLengthSample = settingsClassifier['SPIKE_TRAIN_LENGTH']
    nrSamples = settingsClassifier['NUM_LOG_SAMPLES']
    observationTime = settingsClassifier['OBSERVATION_EXPOSURE_TIME_MS']
    
    nrObsTrain = settingsClassifier['NUM_OBSERVATIONS']
    nrObsTest = settingsClassifier['NUM_OBSERVATIONS_TEST']

    nrFolds = settingsClassifier['NUM_FOLDS']
    baselineValues =settingsClassifier['BASELINE_VALUES']
    
    odourRecordings = ['0-1.csv', '1-1.csv', '2-1.csv']
    plotRecFolder = 'recordings'
    odourNames = ['No odour', 'Ethanol', 'Orange essential oil']

     numArgumentsProvided = len(sys.argv) - 1
    if numArgumentsProvided >=1 :
        crossvalidation = eval(sys.argv[1])
    if numArgumentsProvided >=2 :
        odourRecordings = eval(sys.argv[2])   
    if numArgumentsProvided >=3 :
        plotRecFolder = eval(sys.argv[3]) 
    if numArgumentsProvided >=4 :c
        odourNames = eval(sys.argv[4]) 
    
    logPath = os.path.join(masterPath, plotRecFolder)
    print 'Plotting sensor data..............................................'
    plot_sensor_data(logPath, odourRecordings, nrInputNeurons, nrSamples, fieldnames, odourNames)


    
    if crossvalidation:
        scores = cross_validate(paramsClassifier, settingsClassifier, baselineValues) 
        print scores
        plt.figure()
        plt.bar(range(len(scores[0])), scores[0])
        plt.title('%i-fold crossvalidation scores'%nrFolds)
        plt.ylabel('Percentage correct')
        plt.gray()
        plt.savefig('Crossvalidation_scores.pdf')
        plt.close()
    else:
        settingsClassifier['NUM_REPETITIONS'] = 1
        print 'Training classifier...........................................'
        train_classifier(paramsClassifier, settingsClassifier, baselineValues)
        print 'Training completed'
        raw_input("Press Enter to proceed to testing...")
        
#        baselineValues = make_eNose_baseline.baseline_init_eNose()
        test_classifier(paramsClassifier, settingsClassifier, baselineValues)
    
    

    
    #plot VR spike source for training
    totalSimulationTime = float(observationTime*nrObsTrain)                                                        
    classLabels = utils.loadListFromCsvFile(classLabelsTrain,True)
    plot_spike_sources(masterPath, spikeSourceVRTrain, nrInputNeurons, nrVR, observationTime, totalSimulationTime, classLabels, odourNames)
    
    #plot VR spike source for testing
    totalSimulationTime = float(observationTime*nrObsTest)                                                        
    classLabels = utils.loadListFromCsvFile(classLabelsTest,True)
    plot_spike_sources(masterPath, spikeSourceVRTest, nrInputNeurons, nrVR, observationTime, totalSimulationTime, classLabels, odourNames)
    

