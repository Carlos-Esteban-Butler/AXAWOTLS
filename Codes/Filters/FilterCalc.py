import numpy as np
import os
import json
from copy import deepcopy

scriptdir = os.path.dirname(os.path.realpath(__file__))
# now import the set parameters for the events
setfilepath = os.path.join(os.path.dirname(scriptdir), "Parameters", "set.json")
with open(setfilepath, 'r') as f:
    eventDict = json.load(f)

# routine to find and print all the unique combinations of filter type and thickness. Take into account different angles due to spectrometer geometry as well
def unique_combinations():
    combo = []
    for j in range(1, len(eventDict)):
        j = str(j)
        spec = list(eventDict[j]["spect"].keys())
        type = []; thick = []
        type = eventDict[j]["spect"][spec[0]]["filterType"]
        thick = eventDict[j]["spect"][spec[0]]["filterThickness"]
        for i in range(len(type)):
            tmp = [type[i], thick[i], spec[0]]
            combo.append(tmp)
        type = eventDict[j]["spect"][spec[1]]["filterType"]
        thick = eventDict[j]["spect"][spec[1]]["filterThickness"]
        for i in range(len(type)):
            tmp = [type[i], thick[i], spec[1]]
            combo.append(tmp)
        if spec[1] == "ADP":
            type = eventDict[j]["spect"][spec[1]]["filterType1"]
            thick = eventDict[j]["spect"][spec[1]]["filterThickness1"]
            for i in range(len(type)):
                tmp = [type[i], thick[i], spec[1]]
                combo.append(tmp)
    resultTuple = list(sorted(set(tuple(element) for element in combo))) # use tuple logic to find unique combinations
    result = [list(element) for element in resultTuple] # convert list of tuples to list of lists
    result = [i[:-1] for i in result] # trim off the spectrometer info
    resultNoSpec = list(sorted(set(tuple(element) for element in result))) # use tuple logic to find unique combinations
    print(resultNoSpec)
    print(len(resultNoSpec))

    return result

unique = unique_combinations()