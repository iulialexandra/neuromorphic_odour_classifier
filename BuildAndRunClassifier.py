# author: Alan Diamond. Github repo:
# https://github.com/alandiamond/spinnaker-neuromorphic-classifier


import matplotlib.pyplot as plt
import Classifier as classifier
import ModellingUtils as utils
import sys
import os
import pdb

#pdb.set_trace()
def runClassifier(params, settings, fold):    
    classifier.printParameters('Model Parameters',params)
    classifier.printParameters('Classifier Settings',settings)
    
    populationsInput = list()
    populationsNoiseSource = list()
    populationsRN = list()
    populationsPN = list()
    populationsAN = list()
    projectionsPNAN = list() #keep handle to these for saving learnt weights
    
    if settings['LEARNING']:
        totalSimulationTime = float(settings['OBSERVATION_EXPOSURE_TIME_MS'] * 
                                settings['NUM_OBSERVATIONS'])
    else:
        totalSimulationTime = float(settings['OBSERVATION_EXPOSURE_TIME_MS'] *
                                settings['NUM_OBSERVATIONS_TEST'])
        
    print 'Total Simulation Time will be', totalSimulationTime
    
    DT = 1.0 #ms Integration timestep for simulation
    
    classifier.setupModel(params, settings, DT, totalSimulationTime, 
                          populationsInput, populationsNoiseSource,
                          populationsRN,populationsPN,populationsAN,projectionsPNAN)
    
    utils.recordPopulations(populationsInput,settings['RECORD_POP_INPUT'])
    utils.recordPopulations(populationsNoiseSource,settings['RECORD_POP_NOISE_SOURCE'])
    utils.recordPopulations(populationsRN,settings['RECORD_POP_RN'])
    utils.recordPopulations(populationsPN,settings['RECORD_POP_PN'])
    utils.recordPopulations(populationsAN,settings['RECORD_POP_AN'])
    
    #run the model for the whole learning or the whole testing period
    classifier.run(totalSimulationTime)
    
    fig1 = plt.figure(figsize=(20,20))
    plt.xlabel('Time[ms]', fontsize = 16)
    plt.ylabel('Neurons', fontsize = 16)
    title = 'Testing'
    if settings['LEARNING']:
        title = 'Training'
    title = title + ' - Odour Classification - ' + str(params['NUM_VR']) + \
                                                    ' Virtual Receptors'
    fig1.suptitle(title, fontsize = 18)
    
    indexOffset = 0
    indexOffset = 1 + utils.plotAllSpikes(populationsInput,
                            totalSimulationTime, indexOffset,
                            settings['RECORD_POP_INPUT'])
                            
    indexOffset = 1 + utils.plotAllSpikes(populationsNoiseSource,
                            totalSimulationTime, indexOffset,
                            settings['RECORD_POP_NOISE_SOURCE'])
                            
    indexOffset = 1 + utils.plotAllSpikes(populationsRN,
                                          totalSimulationTime,
                                          indexOffset,settings['RECORD_POP_RN'])
                                          
    indexOffset = 1 + utils.plotAllSpikes(populationsPN,
                                          totalSimulationTime,
                                          indexOffset,settings['RECORD_POP_PN'])
                                          
    indexOffset = 1 + utils.plotAllSpikes(populationsAN,
                                          totalSimulationTime,
                                          indexOffset,settings['RECORD_POP_AN'])
                                          
    
        
    filename = 'RasterPlot-Testing-fold' + str(fold)+'.pdf'
    if settings['LEARNING']:
        filename = 'RasterPlot-Training-fold' + str(fold)+'.pdf'
    plt.savefig(filename)
    plt.close()
    
    
    (fig2, (ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(20,20), sharex=True)
    plt.axes(ax1)
    utils.plotAllSpikes(populationsRN,totalSimulationTime,0, settings['RECORD_POP_RN'])
    plt.axes(ax2)
    utils.plotAllSpikes(populationsPN,totalSimulationTime,0, settings['RECORD_POP_PN'])
    plt.axes(ax3)
    utils.plotAllSpikes(populationsAN,totalSimulationTime,0, settings['RECORD_POP_AN'])
    ax1.set_title('RN layer spikes', fontsize = 30)
    ax2.set_title('PN layer spikes', fontsize = 30)
    ax3.set_title('AN layer spikes', fontsize = 30)
    ax3.set_xlabel('Simulation time[ms]', fontsize = 30)
    ax3.set_ylabel('Neuron indices', fontsize = 30)
    ax3.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    ax1.tick_params(labelsize=20)



    
    filename = 'Separated_RasterPlot-Testing-fold' + str(fold)+'.pdf'
    if settings['LEARNING']:
        filename = 'Separated_RasterPlot-Training-fold' + str(fold)+'.pdf'
    plt.savefig(filename)
    plt.close()
                                              
#        fig.add_subplot(2,1,2)
#        utils.plotAllSpikes(populationsAN,totalSimulationTime, 0, settings['RECORD_POP_AN'])
    
    #if in the learning stage
    if settings['LEARNING']:
        #store the weight values learnt via plasticity, these will be reloaded as 
        #static weights for test stage
        classLabels = utils.loadListFromCsvFile(settings['CLASS_LABELS_TRAIN'],True)
        classifier.saveLearntWeightsPNAN(settings, params, projectionsPNAN,
                                         len(populationsPN),len(populationsAN))
        winningClassesByObservation, winningSpikeCounts = classifier.calculateWinnersAN(settings,populationsAN, classLabels)
        scorePercent = classifier.calculateScore(winningClassesByObservation,classLabels)
    
                           
    else:
        #save the AN layer spike data from the testing run.
        #This data will be interrogated to find the winning class (most active AN pop)
        #during the presentation of each test observation
        #classifier.saveSpikesAN(settings,populationsAN)
        classLabels = utils.loadListFromCsvFile(settings['CLASS_LABELS_TEST'],True)
        winningClassesByObservation, winningSpikeCounts = classifier.calculateWinnersAN(settings,populationsAN, classLabels)
        scorePercent = classifier.calculateScore(winningClassesByObservation, classLabels)
        utils.saveListAsCsvFile(winningClassesByObservation,settings['CLASSIFICATION_RESULTS_PATH'])
        utils.saveListAsCsvFile(winningSpikeCounts,settings['SPIKE_COUNT_RESULTS_PATH'])

    classifier.end()
    
    #write a marker file to allow invoking programs to know that the Python/Pynn run completed
    utils.saveListToFile(['Pynn Run complete'],settings['RUN_COMPLETE_FILE'])
    
    print 'PyNN run completed.'
    return scorePercent  

# if run as top-level script
if __name__ == "__main__":
    
    params = eval(open("ModelParams-eNoseClassifier.txt").read())
    settings = eval(open("Settings-eNoseClassifier.txt").read())
    
    #clear marker file
    if utils.fileExists(settings['RUN_COMPLETE_FILE']):
        os.remove(settings['RUN_COMPLETE_FILE'])
    
    #Override default params with any passed args
    numArgumentsProvided = len(sys.argv) - 1
    
    if numArgumentsProvided >=1 :
        settings['LEARNING'] = eval(sys.argv[1])
    if numArgumentsProvided >=2 :
        params['NUM_VR'] = int(sys.argv[2])
    if numArgumentsProvided >=3 :
        params['NUM_CLASSES'] = int(sys.argv[3])
    if numArgumentsProvided >=4 :
        settings['SPIKE_SOURCE_VR_RESPONSE_TRAIN'] = sys.argv[4]
    if numArgumentsProvided >=5 :
        settings['SPIKE_SOURCE_VR_RESPONSE_TEST'] = sys.argv[5]    
    if numArgumentsProvided >=6 :
        settings['SPIKE_SOURCE_CLASS_ACTIVATIONS'] = sys.argv[6]
    if numArgumentsProvided >=7 :
        settings['NUM_OBSERVATIONS'] = int(sys.argv[7])
    if numArgumentsProvided >=8 :
        settings['NUM_OBSERVATIONS_TEST'] = int(sys.argv[8])
    if numArgumentsProvided >=9 :
        settings['OBSERVATION_EXPOSURE_TIME_MS'] = int(sys.argv[9])
    if numArgumentsProvided >=10 :
        settings['CLASS_LABELS_TRAIN'] = sys.argv[10]
    if numArgumentsProvided >=11 :
        settings['CLASS_LABELS_TEST'] = sys.argv[11]
    if numArgumentsProvided >=12 :
        settings['CLASSIFICATION_RESULTS_PATH'] = sys.argv[12]
        
    runClassifier(params, settings, 0)
