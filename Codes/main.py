#==============AXAWOTLS: Analysis of X-ray Absorption for WDM Observation and Testing of Locally-made Spectrometers===================
#======================================== Author: Carlos Butler =======================================================================
#======================= Email in case of questions: carlosbutler210@gmail.com ========================================================

# This code handles all the actual commands of the code. The functions are found in the various libraries.
# It can extract and save the raw data in a data folder in the spectrometer event folder as a csv, as well as run data analysis
# with the csv data.

# set the parameters ===================================================================================================================
# for these options, 1 is on, 0 is off

# ---------------------------------------------------------------------------------------------------------------------------------------
import numpy as np # need to import early for some parameters
# these pertain to the main code, which yields a spectrum at the end
produce_spectra = 0 # whether to produce spectra. Turning off skips to next section

new_data     =    1 # whether to take any new data from tiffs. if off, will skip all prompts about taking new data unless no data exists
do_offset    =    0 # whether to give option to manually add energy offsets. If add offset, need to run produce_spectra again to apply changes

# the following pertains to post processing
ppl_placeholder = 0 # placeholder in case want to add something in future

# these are for the spectrum_processing_lib
# parameters for absorption coeffecient calculation and plot
do_crystal_calibration  =  1 # whether to do crystal calibration. Only choose no if the crystals are high enough quality
choose_calibration_event = 1 # requires do_crystal_calibration = 1. Whether to choose new calibration event. If no, will get from txt file

# ---------------------------------------------------------------------------------------------------------------------------------------
# these pertain to further data analysis the calculation of the resolution and simple graphing modules
data_analysis     =     1 # whether to do further data analysis. Required to produce spectra first

# main on and off switches of data analysis
simple_graphing   =     1 # whether to graph the basic spectra (as in plots without analysis in them)
absorption     =        0 # whether to output the absorption coefficient graphs again. Parameters for this are above
resolution     =        0 # whether to calculate the resolution by fitting. Keep binning off for this
relative_reflectivity = 0 # whether to calculate relative reflectivity.
conversion_efficiency = 0 # whether to calculate conversion efficiency for a chosen line. Turn off binning for this
source_size    =        0 # whether to calculate source size from knife edge individually. Turn this on if want to test source size calculation
# which event settings to use, set to spec_raw if want to use constant background and spec_1 if want second channel of double channel spectrometer
#events = [[1, "spec", "SUCC"], [1, "spec", "DUCC"]]
events = [[4, "spec", "FSSR"]]

# basic processing parameters. Processing done in the same order as parameters are given.
DoFilters  =     1 # whether to correct for the filters
d_thick_rel = 0.05 # the relative error of the filter thicknesses, in percent/100 (only used in systematic error, which is off by default)
DoDetector =     1 # whether to correct the detector influences (here a greateyes x-ray camera)
physical_units = 1 # whether to convert to photons/(steradian*eV). Requires DoDetector and DoFilters to be on.
DoBinning  =     1 # whether to bin the data
numPixelPerBin = 8 # controls the number of pixels per bin for binning process. Needs to be number in 2 power series, i.e. 2,4,8,16...

# parameters for simple graphing
plot_together = 1 # whether to plot all chosen events together.
in_pixel = 0 # whether to show raw data, so x axis in pixel

# parameters for resolution analysis
with_full_spectrum =  0 # whether to graph full spectrum over the resolution graph (outputs a plot with two subplots)
manual_fitting_area = 0 # whether to manually select the area to fit over
edge        =         0 # whether the fit is around the k-edge. False means using He alpha line
# neccessary if chose to use the edge. Otherwise do not need to adjust
use_transmission   =  1 # whether you are using a transmission spectrum for the edge. False will use abs. coefficient without crystal calibration, so only use for double channel spec
# activates for edge = 0, so fitting to a peak
peak_fitting_func = ["g"] # choices are: g (gauss), l (lorentz), and v (voigt)
fit_bounds = [[-np.inf, -np.inf, -np.inf],[np.inf, np.inf, np.inf]] # the bound of the fit parameters. for gauss is [Amplitude, E_central, y-shift]. Number is percent of the inital guess. np.inf means not bound

# parameters for relative reflectivity calculation. Will automatically pull the other spectrometer's data from event
use_derivative = 0 # This one is buggy atm, keep at 0. Whether to use dN_ph/dE in calculation. 0 will make you choose an energy region to sum over for both spectra
sanity_check = 0 # Only use for double channel spectrometers. if this is on, the rel R_int calc uses the same spectrometer name.


# begin processing ======================================================================================================================

import os
import matplotlib.pyplot as plt
import json
# libraries made by author
import spectrum_processing_lib as spl
import data_analysis_lib as dal
import general_lib as gl

# first pass the relevant parameters to the post processing and spectrum processing libraries
spectrum_process_p = (do_crystal_calibration, choose_calibration_event, new_data, DoBinning, numPixelPerBin)
post_process_p = (ppl_placeholder)
spl.pass_parameters(post_process_p, *spectrum_process_p) # function to pass parameters to spectrum processing lib
# post_process_p passed on by spl to ppl

# now import the set parameters for the events
setPara = gl.import_event_parameters()
# and the spectrometer parameters
specPara = gl.import_event_parameters(get_spectrometer_parameters=True)

if produce_spectra:
    # Have user select the event and which spectrometers to make spectra
    eventNum = str(input("Event: ")); print()
    # add 0 for single digit numbers
    if(float(eventNum) < 10):
        extraZeros = "000"
    else: extraZeros = "00"
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    eventdir = os.path.join(os.path.dirname(scriptdir), "Events", extraZeros + eventNum)
    spectrometers = list(setPara[eventNum]["spect"].keys()) # save the spectrometer names
    # ask user which spectrometers to make a spectrum for
    j = 1
    for i in spectrometers:
        print(j,".", i)
        j += 1
    print(j, ". All")
    whichSpect = int(input("Make spectrum for which spectrometer(s)?: "))

    # prepare dict to easily grab the processed data later
    rawData = {"spect1" : {}, "spect2": {}}

    # Begin process of getting and saving raw data from images
    print("\nFirst select the spectrum, then the background. If double channel, go from top to bottom.")
    print(r"r resets the box, s saves the box, e ends the selecting, c increases the contrast and v decreases it.")
    print()
    # prep the new_data choice
    if new_data == 0: new_data = False
    else: new_data = True
    # get the raw data for the spectrometers chosen by the user
    if (whichSpect == 1 or whichSpect == 2):
        rawData = spl.pullAndSaveRawData("spect" + str(whichSpect), spectrometers[whichSpect-1], rawData, eventNum)
    elif (whichSpect == 3):
        rawData = spl.pullAndSaveRawData("spect1", spectrometers[0], rawData, eventNum)
        print()
        rawData = spl.pullAndSaveRawData("spect2", spectrometers[1], rawData, eventNum)


    # =====================================================================================================================================================
    # now process the data depending on the case. Then make a Fallunterscheidung.
    # first extract the shot info into lists. index 0 is spect1, 1 is spect2. Spect1 should always be the control spectrometer
    type = []
    disp = []
    for sp in spectrometers:
        type.append(setPara[eventNum]["spect"][sp]["type"])
        disp.append(int(setPara[eventNum]["spect"][sp]["disp"]))

    # import E offset dict
    offset = gl.import_event_parameters(get_E_offset = True)

    # go through all data available, skipping the empty dictionaries.
    j = 0
    spectra = {spectrometers[0]: {}, spectrometers[1]: {}} # initialize a dictionary that will save the processed spectra
    for sp in rawData:
        if rawData[sp]: # this checks if the dict is filled
            if (j > 0 and type[j] == "source"): # make a new figure for second source spectrum if desired
                plottgt = input("\nPlot both spectrometers together? (y or n): ")
                if (plottgt == "n"): plt.figure() # if dont want to plot results of both spectrometers together, will make new figure
            if (whichSpect == 2):
                j = 1 # fixes an oversight in the structure. now abs will still work even if chose only spect2
            if (j == 0 and type[j] == "source"): plt.figure() # make a new figure for first source spectrum. Note that figure for abs is made later
            
            labelAddition = "" # used later for the plotting of double channel spectra
            specName = rawData[sp]["name"] # pull name of the spectrometer from rawData dict
            spectra[specName] = spl.make_spectrum(rawData, sp, eventNum, offset, disp[j], setPara) # process the spectra

            # for the case of a source only measurement -------------------------------------------------------------------------------------------------------------
            if (type[j] == "source"):
                if (specPara[specName]["channels"] == "double"): # for these spectrometers need an additional plot
                    labelAddition = " channel 1"; labelAddition1 = " channel 2" 
                    plt.plot(spectra[specName]["spec1"][0], spectra[specName]["spec1"][1], label = specName + labelAddition1) # plot extra spectra for double channel spectrometer
                plt.plot(spectra[specName]["spec"][0], spectra[specName]["spec"][1], label = specName + labelAddition) # plot main spectra
                # add some graphing details
                plt.xlabel("Photon Energy [eV]")
                plt.ylabel("Count [-]")
                plt.title("Event " + eventNum + ": Shot on " + setPara[eventNum]["target"] + " at " + str(setPara[eventNum]["energy"]) + "J")
                plt.grid()
                plt.legend()
                j+=1
                continue
            
            # for a absorption spectroscopy measurement ----------------------------------------------------------------------------------------------------------------
            # note that this relies on the abs spectrum being the second one of event, i.e. "spect2" in rawData
            if (type[j] == "abs"):
                # get the names of the spectrometers
                transSpecName = specName # rename for clarity, corresponds to spectrometer taking the spectrum transmitted through sample
                sourceSpecName = spectrometers[0] # get the first spectrometer name. note that for double channel spectrometers this is not used
                if specPara[transSpecName]["channels"] == "double": sourceSpecName = transSpecName
                if (not rawData["spect1"] and specPara[transSpecName]["channels"] != "double"): # if both spectrometers not chosen in beginning, get data of source spectrum now. Don't for double channel
                    print("\nExtracting the source spectrum: ")
                    rawData = spl.pullAndSaveRawData("spect1", sourceSpecName, rawData, eventNum) # pull raw data
                    spectra[sourceSpecName] = spl.make_spectrum(rawData, "spect1", eventNum, offset, disp[j-1], setPara) # process the source spectra
                aluThickness = setPara[eventNum]["spect"][transSpecName]["thickness"] # thickness of sample
                if do_crystal_calibration: # if chose to do a crystal calibration
                    # grab the calibration event number and data
                    calEventNum = spl.get_calibration_event_num(eventNum, transSpecName)
                    calSpectra = spl.get_calibration_data(calEventNum, transSpecName, sourceSpecName)
                    # run calculation of absorption coefficient, as well as plot everything. plotting is done in function since it plots stuff from each calculation step
                    spl.absorption_coefficient(spectra, calSpectra, transSpecName, sourceSpecName, aluThickness, eventNum, calEventNum, setPara)
                # otherwise get absorption coefficient without calibratin
                else: spl.absorption_coefficient_wo_calibration(spectra, transSpecName, sourceSpecName, aluThickness, eventNum)
                
    print("\nIf you want the graphs, save them in the window(s).")
    plt.show()

    # =================================================================================================================================================
    # ask user if want to build in an energy offset. This is to accommodate for deviations within a single dispersion number event group. do_offset must be turned on
    if do_offset == 1:
        print("\nIf needed, you can assign an energy offset to the spectrometer(s) now.")
        print("Assign to which spectrometer(s):")
        j = 1
        for sp in rawData:
            if rawData[sp]: # checks if dict input exists
                print(j,".", rawData[sp]["name"])
                j += 1
        print(j, ". none")
        if (j == 3):
            j += 1
            print(j, ". both")
        userOffset = int(input("Change offset for: "))

        if (userOffset < 3 and j == 4): # j == 4 to avoid case where j=2 means none
            spectNum = "spect" + str(userOffset)
            spl.offset_change(spectNum, rawData, offset, eventNum)
        if (j == 4 and userOffset == 4):
            spl.offset_change("spect1", rawData, offset, eventNum)
            spl.offset_change("spect2", rawData, offset, eventNum)
        if (j == 2 and userOffset == 1): # for the case of only one spectrometer
            for sp in rawData:
                if rawData[sp]: # checks if dict input exists
                    spl.offset_change(sp, rawData, offset, eventNum)

        # save the offset for future use.
        with open(os.path.join(scriptdir, "Parameters", "E_offset.json"), 'w') as f: 
            json.dump(offset, f)

# data analysis section ============================================================================================================================================

if data_analysis:
    # send the choices to the data_analysis_lib
    data_analysis_p = (plot_together, in_pixel, manual_fitting_area, edge, use_transmission, peak_fitting_func, fit_bounds, relative_reflectivity,use_derivative, with_full_spectrum, events, DoBinning, numPixelPerBin, DoFilters, physical_units, d_thick_rel, DoDetector, sanity_check)
    dal.pass_parameters(data_analysis_p)
    # do basic graph of the spectrum 
    if simple_graphing: dal.basic_graphing()
    # graph absorption coefficient with extra options (filters, binning, etc.) if desired
    if absorption: dal.calculate_absorption_coefficient(do_crystal_calibration)
    # run resolution anaylsis 
    if resolution: dal.analyze_resolution()
    # calculate relative reflectivity if chosen
    if relative_reflectivity: dal.calculate_relative_reflectivity()
    # get conversion efficiency if desired
    if conversion_efficiency: dal.calculate_conversion_efficiency()
    # run source size calculation, including plot
    if source_size:
        for event in events:
            if specPara[event[2]]["knife edge"] == "yes": # check if even has a knife edge
                plt.figure()
                dal.get_source_size(event)
                plt.legend()
            else: print("Event "+str(event[0])+": "+event[2]+" has no knife edge")
    # show all the results
    plt.show()

