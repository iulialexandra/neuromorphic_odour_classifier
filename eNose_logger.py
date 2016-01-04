#author Iulia-Alexandra Lungu (iulialexandralungu@gmail.com)

import serial
import time
import os.path
import sys
import csv
import re
import pandas as pd


class eNose_logger(object):
    """Logs data from serial port into a text file.
    Input:
        -path of the file used to save sensor data
        -name of the file used to save sensor data
        -the serial port used to connect to arduino
        -the fieldnames of the data read from the arduino
    
    Outputs: 
        -time elapsed since arduino was started
        -temperature
        -humidity
        -chemo-sensor data
        
    """
    
    def __init__(self, logPath, logFileName, serialPort, fieldnames):
        self.logPath = logPath
        self.logFileName = logFileName
        # open the output file (for appending)
        self.out = open(os.path.join(logPath, logFileName), 'w' )
        self.fieldnames = fieldnames
        self.writer = csv.DictWriter(self.out, fieldnames = fieldnames, dialect = 'excel', quoting = csv.QUOTE_NONNUMERIC)
        self.writer.writeheader()
        self.serialPort = serialPort
        self._connect_arduino()
        
    def _connect_arduino(self):
        #create the serial connection through the port arduino is connected to
        self.serialConnection = serial.Serial(self.serialPort, timeout=2.0)
        print "Arduino connected"
        #give arduino some time to wake up
        time.sleep(2)
        #clear the buffer
        self.serialConnection.flushInput()
        
        
    def logSensors(self):
       #tell arduino to check sensor values once
       self.serialConnection.write("Sample")
       data = self.serialConnection.readline().strip()
       if data:
           data = re.split(r'\t+', data.rstrip('\t'))
           datadict = {}
           for idx, col in enumerate(self.fieldnames):
               datadict[col] = data[idx]
           self.writer.writerow(datadict)
           return data
       else:
           print 'Arduino not connected'
           return 0

    def close_connection(self):
       self.serialConnection.flushInput()
       self.serialConnection.close()
       self.out.close()
       
       
# if run as top-level script
if __name__ == "__main__":
    try:
        
        settingsClassifier = eval(open("Settings-eNoseClassifier.txt").read())
        #change the name of the file used to record sensor data for each session
        #and each odour
        logFileName = 'test.csv'

        fieldnames = settingsClassifier['SENSOR_NAMES']
        masterPath = settingsClassifier['MASTER_PATH']
        serialPort = settingsClassifier['SERIAL_PORT']
        nrSamples = settingsClassifier['NUM_LOG_SAMPLES']

 
        numArgumentsProvided = len(sys.argv) - 1
        if numArgumentsProvided >=1 :
            logFileName = eval(sys.argv[1])
        if numArgumentsProvided >=2 :
            recFolder = eval(sys.argv[2])   
        if numArgumentsProvided >=3 :
            nrSamples = eval(sys.argv[3]) 
        
        #by default, all the recordings are saved in the 'recordings' folder
        #if not using crossvalidation, separate the train and test samples by hand
        #in the 2 folders created when calling the make_eNose_baseline script
        recFolder = settingsClassifier['CROSSVALIDATION_LOGGER_FOLDER']
        logPath = os.path.join(masterPath, recFolder)

        nose = eNose_logger(logPath, logFileName, serialPort, fieldnames)
        #sample the sensors a'nrSamples' number of times
        for i in range(nrSamples):
            nose.logSensors()
            
        nose.close_connection()
        
    except KeyboardInterrupt:
        pass