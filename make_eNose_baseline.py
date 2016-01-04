#author Iulia-Alexandra Lungu (iulialexandralungu@gmail.com)

import numpy as np
import os.path
import sys
import pandas as pd
from eNose_logger import eNose_logger

def baseline_init_eNose():
    """Creates folders if non-existent, calculates baseline for the sensors.
    Returns the baseline values.
    """
    
    settingsClassifier = eval(open("Settings-eNoseClassifier.txt").read())
    paramsClassifier = eval(open("ModelParams-eNoseClassifier.txt").read())
    
    fieldnames = settingsClassifier['SENSOR_NAMES']
    baselineFileName = settingsClassifier['BASELINE_FILENAME']
    masterPath = settingsClassifier['MASTER_PATH']
    serialPort = settingsClassifier['SERIAL_PORT']
    nrSamples = settingsClassifier['NUM_BASELINE_SAMPLES']
    nrInputNeurons = paramsClassifier['NUM_INPUT_NEURONS']

    #create folders if they don't exist yet
    recFolder = settingsClassifier['CROSSVALIDATION_LOGGER_FOLDER']
    logPath = os.path.join(masterPath, recFolder)
    if not os.path.exists(logPath): os.makedirs(logPath)
    
    #if not using crossvalidation, separate your recordings into training
    #and test ones, by moving them into 2 separate folders
    trainFolder = settingsClassifier['TRAIN_LOGGER_FOLDER']
    logPath = os.path.join(masterPath, trainFolder)
    if not os.path.exists(logPath): os.makedirs(logPath)
    
    testFolder = settingsClassifier['TEST_LOGGER_FOLDER']
    logPath = os.path.join(masterPath, testFolder)
    if not os.path.exists(logPath): os.makedirs(logPath)  
    
    #record baseline samples
    nose = eNose_logger(masterPath, baselineFileName, serialPort, fieldnames)
    for i in range(nrSamples):
        nose.logSensors()   
    nose.close_connection()

    #calculate the baseline response for each sensor        
    with open(os.path.join(masterPath, baselineFileName), 'rb') as logFile:
        samples = pd.read_csv(logFile)
        baseSamples = np.zeros((nrInputNeurons, nrSamples))
        
        for idx, neuron in enumerate(range(3, 3+nrInputNeurons)):
            baseSamples[idx, :] = samples.iloc[:, neuron]
                
    baselineValues = np.mean(baseSamples, 1)
    return baselineValues