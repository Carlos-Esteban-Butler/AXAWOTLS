from scipy.optimize import curve_fit
from scipy.special import erf, voigt_profile
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d
from copy import deepcopy
import sys
# libraries made by author
import general_lib as gl
import post_processing_lib as ppl
import spectrum_processing_lib as spl

# pull the set.json and e_offset.json and specPara.json dicts
setPara = gl.import_event_parameters()
specPara = gl.import_event_parameters(get_spectrometer_parameters=True)
E_offset = gl.import_event_parameters(get_E_offset = True)

# grab the parameters from spectrum_processing_lib. First initalize global variables. Also identify the y-label for graphs.
plot_together=in_pixel=manual_fitting_area=edge=use_transmission=peak_fitting_func=fit_bounds=relative_reflectivity=use_derivative=with_full_spectrum=events=DoBinning=numPixelPerBin=DoFilters=physical_units=d_thick_rel=DoDetector=sanity_check = 0
def pass_parameters(data_analysis_p):
    global plot_together, in_pixel, manual_fitting_area, edge, use_transmission, peak_fitting_func, fit_bounds, relative_reflectivity, use_derivative, with_full_spectrum, events, DoBinning, numPixelPerBin, DoFilters, physical_units, d_thick_rel, DoDetector, sanity_check # make global
    plot_together, in_pixel, manual_fitting_area, edge, use_transmission, peak_fitting_func, fit_bounds, relative_reflectivity, use_derivative, with_full_spectrum, events, DoBinning, numPixelPerBin, DoFilters, physical_units, d_thick_rel, DoDetector, sanity_check = data_analysis_p # set parameters
    # set all the event info to string
    events = np.array(events).astype(str)
    fit_bounds = np.array(fit_bounds)/100 # change percentage to decimal
    return

# parameters for general behavior of data analysis
use_title = 1 # whether to have a title on the graph
delta_pix = 0.0135 # width of pixel. Assumed to be same for all cameras [mm]

# for source size calculation
magnification = 3 # This magnification depends on the ray distance from knife edge to detector and to source. Specific to the spectrometer [-]
spec_with_knife_edge = "SUCC" # define which spectrometer has the knife edge you want to use

# options for resolution calculation
#FWHM_test_list = np.linspace(0.1, 7, 1000) # FWHM to test [eV]
FWHM_test_list = np.linspace(0.5, 6, 1000) # FWHM to test [eV]
manual_chi2_limit = 1 # max chi2 allowed for a model to be acceptable
cut_off_saturated = 0
mirror_values_left = 1 # choose to mirror a peak to fill in the left side
use_min_chi_value = 1 # if to use minimal chi2 of the fits instead of the uppper_chi2_limit. Do if not reaching 1
deviation_from_min_chi = 1.5 # how far the chi2 value is allowed to deviate from min_chi2. is max_chi2=min_chi2*deviation_from_min_chi
# parameters for graph output
plot_worst_acceptable_fit = 1 # whether to plot the worst acceptable fit as well. Only for comparison
# choose limits for fitting in resolution calculation
if edge:
    fit_lower_limit = 1557.6; fit_upper_limit = 1561 # to use the edge [eV]
else: # lower limit is first in list, upper is second [eV]
    resolution_limits = {
        "DUCC" : [1597.5, 1601.5],
        "SUCC" : [1597, 1603],
        "FSSR" : [1597.8, 1602]
    }
# to select a feature more exactly goes with cut_off_saturated
feature_limits = [1597.9, 1598.72]
# doppler broadening info
sigma_doppler = 0.474 # from FLYCHK of T=1000eV and n_e=4e21cm-3 
d_sigma_doppler = 0.1 # uncertainty estimated from FLYCHK simulation with Te = 500 to 1025 eV

# initialize the ylabel parameter outside the function
ylabel = "Counts [-]" # will be used if no corrections are turned on.

# this function grabs the distances that the x-rays are allowed to diverge over for a spectrometer for each energy
# spec is spectrometer name and dists is the list for holding the distance values
def get_diverging_distances(spec, data, dists):
    C = 12398 # Umrechnungsfaktor of wavelength to energy
    # right now the equation corresponds to FSSR-1D. If a different geometry with spherical crystal will need to change this
    if specPara[spec]["crystal bend"] == "spherical": 
        # R is crystal-curvature radius [mm], C is E --> 1/lambda conversion factor [eV --> 1/angstrom], twod is lattice spacing of mica [angstrom]
        # a_0 is source-crystal distance for central ray [mm], n is diffraction order [-], Ecentral is central energy [eV]
        R =  specPara[spec]["radius of curvature"]; twod = specPara[spec]["lattice spacing"]
        a_0 = specPara[spec]["a0"]; n = specPara[spec]["order"]; Ecentral = specPara[spec]["E-central"]
        # formulas from dispersion section of Carlos Butler's master thesis
        s = np.sqrt(a_0**2 + R**2 - 2*a_0*R*n*C/(twod*Ecentral))
        a = R*n*C/twod *1/data[0] + np.sqrt((R*n*C/twod)**2 * 1/data[0]**2 + s**2 - R**2)
        dists.append(a) # since rays allowed to diverge until incident on crystal, after which is focused
    # now for flat crystal spectrometers
    if specPara[spec]["crystal bend"] == "flat":
        n = specPara[spec]["order"]; twod = specPara[spec]["lattice spacing"]
        L = specPara[spec]["length"]; Ecentral = specPara[spec]["E-central"]
        twod = twod/n
        theta0 = np.arcsin(C/(twod*Ecentral)) # central Bragg angle
        d_calc = lambda E_i: L*(1/np.sqrt((twod/C)**2*E_i**2-1) - 1/np.sqrt((twod/C)**2*Ecentral**2-1))
        W = twod*data[0]/(n*C)*(L*np.tan(theta0) + d_calc(data[0])) # W is total distance ray travels to detector
        # to check if output is reasonable for bug fixing
        # plt.plot(data[0], W)
        # plt.show()
        dists.append(W)
    return dists

# correction functions --------------------------------------------------------------------------------------------------------
# for all correction and processing functions, if force is on, then will automatically go through, regardless of user input

# function to bin the data. Note that it uses the default of binned_statistic, which is mean for the values
# kwarg manual_num_pix_per_bin is if a function wants to override the user input num of bins.
def bin_data(data, force = False, manual_num_pix_per_bin = 0):
    # if chose not to bin, then break the function
    if not DoBinning and not force: return data
    # bin the data, outputting the binned count values and edges of the bins. Misc is a placeholder and not used
    if manual_num_pix_per_bin == 0: numBins = 2048/numPixelPerBin # set number of bins 
    else: numBins = 2048/manual_num_pix_per_bin # if manually want to set
    binnedValues, binEdges, misc = binned_statistic(data[0], data[1], bins = numBins)
    binCentered = (binEdges[:-1]+binEdges[1:])/2 # calculates the center of each bin
    binnedData = np.vstack((binCentered, binnedValues)) # creates new array with binned data
    return binnedData

# correct out the influence of the filters. with_error gives additional output of systematic error
def filter_correction(data, eventNum, specName, extra, force = False, with_error = False):
    # if chose not to correct for the filters, then break the function
    if not DoFilters and not force and not with_error: return data
    # if error chosen but DoFilters not, then return a dummy for the error on the data
    elif not DoFilters and not force and with_error: return data, np.vstack((data[0], np.ones(len(data[1]))))
    # first get the directory of the script for use later
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # consider if is the second channel of a double spectrometer or not
    add = ""
    if (extra): add = "1" # if is for the extra spectra of double channel, add a one to the key
    # get the filter info from the set parameters dict
    filTypes = setPara[eventNum]["spect"][specName]["filterType" + add]
    filThick = setPara[eventNum]["spect"][specName]["filterThickness" + add]
    # for each filter on spectrometer (or spectrometer arm), correct the data
    T_at_He_alpha_list = []
    total_trans = np.ones(len(data[0])) # initialize array for total transmission
    counts = deepcopy(data[1]) # to use for error calculation later
    for i in range(len(filTypes)):
        filename = filTypes[i] + "_" + str(filThick[i]).replace(".", ",") + ".csv" # find file name for respective thick and type
        filterfilepath = os.path.join(scriptdir, "Filters", filename) # get file path to the filter csv file
        transData = gl.extract_csv(filterfilepath, filter=True) # pull the tranmission data from the csv
        filter_func = interp1d(transData[0], transData[1]) # interpolate a function from the filter data
        transmission = filter_func(data[0]) # define transmission
        total_trans *= transmission # incorporate into total transmission
        data[1] = data[1]/transmission # correct the data by dividing by the tranmission for the given energy
        T_at_He_alpha_list.append(filter_func(1598.4))
    # in case want to check values
    # T_at_He_alpha = np.prod(T_at_He_alpha_list)
    # print("\nTotal transmission of "+specName+" for event "+eventNum+": "+str(T_at_He_alpha))
    # guess a relative uncertainty of the filter thickness (percent/100), then get the uncertainties of the counts
    d_trans_rel = np.abs(np.log(total_trans)*d_thick_rel) # calculate the uncertrainty of Trans. Formula is from T=e^(-mu d)
    d_counts = counts/total_trans*d_trans_rel # calculate the uncertainty of the counts due to d_trans
    d_data = np.vstack((data[0],d_counts)) # make two row array to include energy values as well
    #print(d_counts[gl.find_nearest_index(data[0], 1600)], data[1][gl.find_nearest_index(data[0], 1600)])
    if with_error: return data, d_data
    return data

# Remove the influences of the camera so that N_ph results, which is the number of photons landing on each pixel respectively. 
# Be careful about binning, as this will change the meaning of N_ph
# reverse means to do calculation backwards, so photons --> counts
def detector_correction(data, force = False, reverse = False):
    if not DoDetector and not force: return data
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # extract the quantum efficiency and photoelectrons per photon data from the csvs in the Camera folder
    qe = gl.extract_csv(os.path.join(scriptdir, "Camera", "quantum_efficiency.csv") , camera=True)
    e_per_photon = gl.extract_csv(os.path.join(scriptdir, "Camera", "photoelectrons_over_energy.csv") , camera=True)
    # set the qe to a constant value for now
    qe[1] = qe[1]/qe[1]*77.5
    # fit the e_per_photon linearly. np.polyfit return the prefactors of the polynomial, starting with the highest degree.
    e_per_photon_fit_p = np.polyfit(e_per_photon[0], e_per_photon[1], 1)
    # set some camera parameters. All are from Philipp
    E_hole = 1/e_per_photon_fit_p[0] # [eV/e-]
    gain = 0.4 # [counts/e-]
    # now do calculation. N_ph is number of photons
    if not reverse: 
        N_ph = np.zeros(data.shape) # initialize
        N_ph[0] = deepcopy(data[0]) # copy the E values into the first row of N_ph
    elif reverse: 
        N_counts = np.zeros(data.shape)
        N_counts[0] = deepcopy(data[0]) # copy the E values into the first row of N_counts
    for i in range(len(data[0])): # for each energy, calculate the number of photons. Choose the closest data point for the qe, then convert % to decimal
        if not reverse:
            N_ph[1][i] = data[1][i]/gain * E_hole/data[0][i] * 1/(qe[1][gl.find_nearest_index(qe[0], data[0][i])] / 100)
        else:
            N_counts[1][i] = data[1][i]*gain*data[0][i]*(qe[1][gl.find_nearest_index(qe[0], data[0][i])] / 100) / E_hole
    # selection = (e_per_photon[0]<data[0].max()) & (e_per_photon[0]>data[0].min()) # isolate the feature
    # e_per_photon = e_per_photon[:, selection]
    # plt.plot(e_per_photon[0], e_per_photon[1], "o")
    # plt.plot(data[0], e_per_photon_func(data[0]))
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()
    if not reverse: return N_ph
    else: return N_counts

# convert to physical units, here photons/(sterdian*eV), using an assumed R_int and formula from Paul's paper.
# note that this only makes sense if detector and filter correction is done first. This is accounted for in the code
# this calculation currently assumes a constant for delta E, R_int. Only works bc changes of spectrum are small compared to inherent resolution
def convert_to_physical_units(data, spec):
    if not DoDetector or not DoFilters or not physical_units: return data # if detector and filter corrections not done, don't convert
    # grab diverging distances for each spectrometer
    dists = [] # initialize distance
    dists = get_diverging_distances(spec, data, dists) # append onto dists [mm]
    # define the neccessary variables
    delta_E = data[0][2]-data[0][1] # use the energy separation of first two data points, excepting the very first point
    R_int = specPara[spec]["R_int"]
    if specPara[spec]["crystal bend"] == "spherical": collecting_width = specPara[spec]["crystal width"] # change to the width of crystal, which is the collecting area
    elif specPara[spec]["crystal bend"] == "flat": collecting_width = delta_pix
    R_int *= 10**(-6) # convert micro rad to rad
    photons_units = data[1]*dists[0]/(delta_E*collecting_width*R_int)
    data_units = np.vstack((data[0],photons_units))
    return data_units

# fetching data and additional calculations --------------------------------------------------------------------------------------------------------------

# function to calculate the statistical error due to the ccd camera. Is a combination of noise of background and poisson noise
# returns an array with [eV, stat_error], where the error is still of the pure counts
def get_statistical_error(event):
    # first, get the box data to calculate the number of pixel rows of the spectrum. 
    box_label = deepcopy(event[1])
    if "_raw" in box_label: box_label = box_label.replace("_raw", "") # remove the "_raw" if chose raw data previously. Aligns with box_info.txt convention
    box_data = spl.control_box_data(event[0], event[2], read = box_label)
    num_rows_spectrum = np.abs(box_data[0][1] - box_data[0][0]) # number of rows that were averaged over to find spectrum
    # get the standard deviation of the data from the background. Just take one value for the entire energy range, as doesn't significantly change
    box_back = spl.control_box_data(event[0], event[2], read = "back")# get the box from box_info.txt
    std_array = spl.fullImageProcessing(event[2], event[0], type="std", manual_box=box_back) # get the std deviations for each pixel number along x
    std = np.mean(std_array[1]) # take the mean of the std deviations for convenience. This could be improved in the future
    if specPara[event[2]]["crystal bend"] == "spherical": std*=np.sqrt(num_rows_spectrum) # since summed over pixel rows for spherically bent crystal geometry
    elif specPara[event[2]]["crystal bend"] == "flat": std/=np.sqrt(num_rows_spectrum) # otherwise continue to flat crystal calculation
    # next, derive the poisson noise ------------------------------------------------------------------------------------------
    if specPara[event[2]]["crystal bend"] == "spherical": num_rows_spectrum = 1 # account for that the FSSR spectra are taken using sum
    # then, get the poisson noise by considering the entire counts of an energy, then convert counts to photons.
    # next, calculate the noise by square root, finally convert back to counts and consider that the counts are from a horizontal lineout.
    data = gl.pull_data(event[0], event[1], event[2]) # pull spectrum data
    data = data.astype(float) # necessary in case we use a constant background. Normally ppl.correct_background() does it automatically
    data[1][data[1] < 0] = 0 # to avoid error with the sqrt later
    E_offset_str = "E_offset"; extra = False
    if event[1] == "spec_1": E_offset_str += "_1"; extra = True
    # data = ppl.calc_E(data, int(event[0]), event[2], setPara[event[0]]["spect"][event[2]]["disp"], E_offset[event[0]][event[2]][E_offset_str], extra=extra)
    full_counts = data[1]*num_rows_spectrum
    full_photons = detector_correction(np.vstack((data[0], full_counts)), force=True)
    full_photons[1] = np.sqrt(full_photons[1])
    poisson_noise = detector_correction(full_photons, force=True, reverse=True)[1] / num_rows_spectrum
    # get the total camera noise --------------------------------------------------------------------------------------------
    stat_error = std + poisson_noise
    #stat_error = poisson_noise
    return np.vstack((data[0], stat_error))

# special function to pull the data. Didn't use gl.pull_data since that doesnt allow pulling of raw data
# outside data refers to raw data pulled from outside the function, for example with temporary data
# In the case of outside data, a constant background will be subtracted. this is bc outside data is used for special handling of TIFFs,
# i.e. summing or vertical lineouts. Default is without error. Error will assume only statistical error
def fetch_data (event, outside_data = np.array([]), error = False, sys_error = False):
    # set the y-label according to the corrections chosen
    global ylabel
    if not DoFilters:
        ylabel = "Counts [-]"
    if DoFilters and DoDetector: 
        ylabel = "Photons on Detector (Filter Corrected) [-]"
        if physical_units: ylabel = r"Number of Photons $\left[ \frac{1}{\text{sr} \cdot \text{eV}}\right]$"
    # if outside data not input, then pull it from the csv files. else use the outside data
    if outside_data.any(): data = outside_data
    else: data = gl.pull_data(event[0], event[1], event[2])
    # if using raw data or outside_data, convert pixel to energy
    if event[1] == "spec_raw" or event[1] == "spec_1_raw" or outside_data.any(): 
        data[1] -= data[1].min() # take a constant background, which is the smallest signal
        data = data.astype(float) # necessary because we used a constant background. Normally ppl.correct_background() does it automatically
        E_offset_str = "E_offset"; extra = False
        if event[1] == "spec_1_raw": E_offset_str += "_1"; extra = True
        data = ppl.calc_E(data, int(event[0]), event[2], setPara[event[0]]["spect"][event[2]]["disp"], E_offset[event[0]][event[2]][E_offset_str], extra=extra)
    extra = False # initialize extra boolean
    if event[1] == "spec_1": extra = True # if doing extra spectrum of double channel spectormeter.
    # do corrections. Note that will only happen if the corresponding user parameter is on. apply corrections to data and d_data independently
    if error: 
        # get the camera noise
        d_data = get_statistical_error(event)
        # systematic error due to unknown filter thickness is there, but unused at the moment
        data, d_data_filter = filter_correction(data, event[0], event[2], extra=extra, with_error=True)
        d_data = filter_correction(d_data, event[0], event[2], extra=extra)
        if sys_error: d_data[1] = d_data[1] + d_data_filter[1] # add on sys_error if chosen
    else: data = filter_correction(data, event[0], event[2], extra=extra)
    data = detector_correction(data)
    data = convert_to_physical_units(data, event[2])
    data = bin_data(data)
    if error:
        d_data = detector_correction(d_data)
        d_data = convert_to_physical_units(d_data, event[2])
        d_data = bin_data(d_data)
        return data, d_data
    return data

# Function to determine the source size using a knife edge
# alone decides if produce graph of error func fit. with_error decides if return source size uncertainty as well 
def get_source_size (event, alone=True, with_error=False):
    FWHM = 0; dFWHM = 0 # initialize the full width at half max and its uncertainty outside the do_fit()
    # define the error function with amplitude A, shift mu and std deviation sigma
    def Error(x, *p):
        A, sigma, mu, c = p
        y = A*erf((x - mu)/(sigma*np.sqrt(2))) + c
        return y
    # define general processing, where the results are recorded at the end, as well as graphed
    def do_fit(func, data, input_guess):
        nonlocal FWHM, dFWHM
        parameters, covariance = curve_fit(func, data[0], data[1], input_guess)
        FWHM = round(2*np.sqrt(2*np.log(2))*parameters[1], 3)
        dFWHM = round(2*np.sqrt(2*np.log(2))*covariance[1][1], 3)
        fit = func(data[0], parameters[0], parameters[1], parameters[2], parameters[3])
        if alone: plt.plot(data[0], fit, '-', label=func.__name__+' fit with FWHM = '+str(FWHM)+" pixels")
        return
    # pull the data into the dictionary. Note that "spect1" is essentially a placeholder at this point
    spec = spl.fullImageProcessing(event[2], event[0], type="vertical", only_choose_vertical=False)
    # correct out constant background
    spec[1] -= spec[1].min()
    data = spec # rename for clarity
    idx_mean = gl.find_nearest_index(data[1], np.mean(data[1]))
    E_central = data[0][idx_mean] # location of the central point for the fit, corresponding to midpoint of edge [eV]
    data[1] -= np.mean(data[1]) # make the mean the zero point of the signal as well
    A = data[1].max() # amplitude to fit to
    c = 0
    do_fit(Error, data, [A, 2, E_central, c])
    # if running function directly from main.py, make graphs
    if alone:
        plt.plot(data[0], data[1], label = "data")
        plt.title("Fit to Knife Edge for Event "+str(event[0]))
        plt.xlabel("Pixel [-]")
        plt.ylabel("Arbitrary Unit [-]")
    # finally get source size. As FWHM is essentially the width of the source image on the chip, need to convert pixel to mm, then apply magnification
    source_size = round(FWHM*delta_pix/magnification, 4)
    d_source_size = round(dFWHM*delta_pix/magnification, 4) # uncertainty of source size
    print("\nSource size from event "+str(event[0])+": "+str(source_size)+r"+-"+str(d_source_size)+" mm")
    if alone: return # no need to return source_size variable
    if with_error: return source_size, d_source_size
    return source_size

# functions that output a result ---------------------------------------------------------------------------------------------------------------

# Function to graph the spectrum by itself
def basic_graphing():
    global ylabel
    plt.figure()
    i = 1 # initialize looping variable to use to stop extra empty figure
    x_min = 3000; x_max = 0 # variables for setting xlim of plot_together plot. 3000 and 0 bc then will def be replaced by first iteration
    for event in events:
        if in_pixel and event[1].find("_raw") == -1: # if in pixel is true and _raw data not originally chosen, then pull raw data and set x axis for graph
            data = gl.pull_data(event[0], event[1] + "_raw", event[2]) 
            plt.xlabel("Pixel Number [-]") 
        else: # else get standard data in energy. note that this will impose a constant background if event[1] has "_raw" in it
            data, d_data = fetch_data(event, error=True)
            plt.xlabel("Photon Energy [eV]") # set x label for all graphs
        plt.ylabel(ylabel) 
        # do graphing of data with corresponding labels
        add_label = ""
        if event[1].find("_1") != -1: add_label = " Spectrum 2" # add extra str to label to denote that second channel for a double channel spectrometer
        if specPara[event[2]]["channels"] == "double" and event[1].find("_1") == -1: add_label = " Spectrum 1"
        # cut data if want
        minE = 1510
        maxE = 1740
        data = data[:, data[0] > minE]
        data = data[:, data[0] < maxE]
        if not in_pixel: 
            d_data = d_data[:, d_data[0] > minE]
            d_data = d_data[:, d_data[0] < maxE]
        #plt.plot(data[0], data[1], lw=2, label=event[2]+" (event "+event[0]+" at "+str(setPara[event[0]]["energy"])+"J)")
        plt.plot(data[0], data[1], lw=2, label=event[2]+add_label+" (event "+event[0]+": "+str(setPara[event[0]]["energy"])+"J on "+str(setPara[event[0]]["target"])+ ")")
        #plt.plot(data[0], data[1], lw=2, label=add_label)
        #plt.errorbar(data[0], data[1], fmt="ro", yerr=d_data[1], label='data', markersize=2.5, ecolor="orange", capsize=2, zorder = 1)
        # if a single plot, then adjust xlim and finish
        if not plot_together:
            plt.xlim(data[0].min()*0.999, data[0].max()*1.001) # so the graph barely misses axis
            plt.title("Event "+event[0]+": "+event[2]+" Spectrum from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"])
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join("Graphs","basic_spectra","basic_spectrum_of_"+setPara[event[0]]["target"]+"_event_"+event[0]+"_on_"+event[2]+".pdf"))
        if not plot_together and len(events) > i: # if not plotting together and havent reached end of events list, then make a new figure
            plt.figure()
        if plot_together:
            if data[0].min() < x_min: x_min = data[0].min()
            if data[0].max() > x_max: x_max = data[0].max()
        i += 1
    if plot_together:
        # some options for visualization
        #plt.fill_between(data[0], min(data[1]), max(data[1]), where=(data[0] > 1585) & (data[0] < 1610), alpha = 0.5, label="Integration Area")
        #plt.axvspan(1585, 1610, alpha = 0.2, label = "Integration area", color="palevioletred")
        plt.legend()
        plt.xlim(x_min*0.999, x_max*1.001)
        # plt.title("Spectra from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"]+" with Various Spectrometers")
        plt.title("Spectra of "+setPara[event[0]]["target"]+" at Various Laser Energies")
        plt.grid()
        # for the name of the save file
        events_str = ""
        for event in events: events_str += event[0]+"_"
        events_str = events_str[:-1]
        plt.savefig(os.path.join("Graphs","basic_spectra","spectra_of_"+setPara[event[0]]["target"]+"_events_"+events_str+".pdf"))
    return

# function to calculate the resolution.
# note that it has no arguments. This is because all the needed inputs are given by the user in the beginning of main.py
def analyze_resolution ():
    print("Begin resolution calculation #############################")
    global DoFilters, DoDetector, ylabel
    # turn on filter and detector corrections if not manually chosen, since needed for calculation
    switched_DoFilters = False; switched_DoDetector = False # variables to save if switched or not
    if not DoFilters: DoFilters = 1; switched_DoFilters = True
    if not DoDetector: DoDetector = 1; switched_DoDetector = True
    # initialize results list
    results = {}
    # Define fitting functions. Note that sigma will be the varying parameter, so it is defined nonlocally
    sigma = 1
    # start with the Gaussian function
    def Gauss(x, *p):
        nonlocal sigma
        A, mu, c = p
        y = A*np.exp(-1/(2*sigma**2)*(x-mu)**2) + c
        return y
    # define the lorentz distribution. sigma is equivalent to gamma (standard variable of lorentz) in this case
    def Lorentz(x, *p):
        nonlocal sigma
        A, mu, c = p
        y = A*(sigma**2/((x - mu)**2 + sigma**2)) + c
        return y
    # define the error function with amplitude A, shift mu and std deviation sigma
    def Error(x, *p):
        nonlocal sigma
        A, mu, c = p
        y = A*erf((x - mu)/(sigma*np.sqrt(2))) + c
        if use_transmission: y = A*erf(-(x - mu)/(sigma*np.sqrt(2))) + c # flip function along y axis
        return y
    # define voigt function. This is a convolution of gauss and lorentz
    def Voigt(x, *p):
        nonlocal sigma
        A, mu, c, gamma = p
        y = A*voigt_profile(x-mu, sigma, gamma) + c
        return y
    
    # begin looping over the chosen events
    i = 1 # looping variable for results list
    for event in events:
        result = {"Event|": event[0], "Spec|": event[2], "Gauss|": None, "d_gauss+|": None, "d_gauss-|": None, "sigma_so|": None, "d_so|": None, "Lorentz [eV]|": None, "d_lorentz|": None, "Doppler": None, "d_dopp|": None, "sigma_sp|": None, "d_sp+|": None, "d_sp-|": None}
        # make figure, choosing if a double or single figure
        if with_full_spectrum: 
            fig, (ax1, ax2) = plt.subplots(2, figsize=(8,8))
            #plt.rcParams["figure.figsize"] = (3,20)
        else:
            fig, ax1 = plt.subplots(1, figsize=(8,5))
        fig.supxlabel("Photon Energy [eV]") # set x label for all graphs
        # fetch the data for the event. Use a special function as enables using raw data as well
        data, d_data = fetch_data(event, error=True)
        if not edge: fig.supylabel(ylabel) # set y label for all graphs
        elif edge and use_transmission: fig.supylabel("Transmission [-]")
        # if want to add an artificial error
        if event[2] == "FSSR": d_data[1] += np.mean(data[1])*0.2 # add a percent error due to poor crystal quality
        # do basic graphing if chosen
        if with_full_spectrum:
            # ax2.plot(data[0], data[1], lw=2, label=event[0]+" on "+event[2])
            ax2.plot(data[0], data[1], lw=2, label="Absorption Channel")
            ax2.grid()
            if edge and not use_transmission: 
                data_extra = fetch_data([event[0], "spec_1", event[2]])
                ax2.plot(data_extra[0], data_extra[1], label = "Transmission Channel")
                ax2.legend()
                ax2.set_ylabel(ylabel)
            ax2.set_xlim(data[0].min()*0.999, data[0].max()*1.001) # so the graph barely misses axis
        
        # use reduced chi squared test
        def full_fitting(fit_data, d_fit_data, func, initial_guess):
            nonlocal sigma
            # define function to do the actual fitting
            chi2_results = []
            def do_fit(FWHM_test, plot=False):
                nonlocal sigma
                sigma = FWHM_test/(2*np.sqrt(2*np.log(2)))
                if peak_fitting_func[0] == "l": sigma = FWHM_test/2
                bounds = ([initial_guess[0]*fit_bounds[0][0], initial_guess[1]*fit_bounds[0][1], initial_guess[2]*fit_bounds[0][2]],[initial_guess[0]*fit_bounds[1][0], initial_guess[1]*fit_bounds[1][1], initial_guess[2]*fit_bounds[1][2]])
                nu = len(fit_data[0]) - len(initial_guess) # get degrees of freedom
                parameters, covariance = curve_fit(func, fit_data[0] ,fit_data[1] ,p0 = initial_guess, sigma=d_fit_data[1], absolute_sigma=True, bounds=bounds)
                red_chi_squared = np.sum((fit_data[1] - func(fit_data[0], *parameters))**2/d_fit_data[1]**2)/nu
                if plot:
                    x = np.linspace(data[0].min(), data[0].max(), 200)
                    ax1.plot(x, func(x, *parameters), label="FWHM = "+str(round(FWHM_test,2))+ r", $\chi^2$ = "+str(round(red_chi_squared,2)))
                    print(np.round(100*(np.array(parameters)-np.array(initial_guess))/np.array(initial_guess), decimals=2))
                return red_chi_squared
            # test all the options
            for FWHM_test in FWHM_test_list:
                red_chi_squared = do_fit(FWHM_test)
                chi2_results.append(red_chi_squared)
            # if want to use min_chi2 as limit instead of setting limit
            if use_min_chi_value:
                chi2_limit = min(chi2_results)*deviation_from_min_chi
            else: chi2_limit = manual_chi2_limit
            chi2_results = np.array(chi2_results)
            accepted_FWHM = []
            for i in range(len(FWHM_test_list)):
                if chi2_results[i] <= chi2_limit:
                    accepted_FWHM.append([FWHM_test_list[i], chi2_results[i]])
            accepted_FWHM = np.array(accepted_FWHM)
            FWHM_best = FWHM_test_list[chi2_results.argmin()]
            if accepted_FWHM.size == 0:
                print("Above 1: smallest chi is "+str(chi2_results.min())+" at FWHM of "+str(FWHM_test_list[chi2_results.argmin()]))
            else:
                print("Lowest FWHM fit: ", accepted_FWHM[0],"| Highest FWHM fit: ", accepted_FWHM[-1])
                # FWHM_worst = accepted_FWHM[accepted_FWHM[:,1].argmax()][0]
                if plot_worst_acceptable_fit:
                    print("Deviations from initial guesses for worst fit (percent):")
                    red_chi_squared = do_fit(accepted_FWHM[0][0], plot=True)
                    red_chi_squared = do_fit(accepted_FWHM[-1][0], plot=True)

            print("Deviations from initial guesses for best fit (percent):")
            red_chi_squared = do_fit(FWHM_best, plot=True)
            ax1.errorbar(fit_data[0], fit_data[1], fmt="ro", yerr=d_fit_data[1], label='data', markersize=1.5, ecolor="purple", capsize=2, zorder = 1)
            if accepted_FWHM.size == 0:
                print("Please adjust fitting parameters.")
                plt.legend()
                plt.show()
                sys.exit()
            # get the error of the FWHM, plus is at index 0, minus is index 1
            d_FWHM = [np.abs(accepted_FWHM[:,0].max() - FWHM_best), np.abs(accepted_FWHM[:,0].min() - FWHM_best)]
            return FWHM_best, d_FWHM
        
        # if using the K-edge of the absorption spectrum, gather the absorption array for the event. Note that this should only be used for double channel spec
        # not very useful anymore, but kept in just in case
        if edge and not use_transmission:
            # stop the resolution calculation for this event if not an absorption event
            if "abs" not in setPara[event[0]]["spect"][event[2]]["type"]: print("--------Error: "+str(event[0])+" not an abs event."); continue
            # fetch the data for the other arm. Use a special function as enables using raw data as well
            extra_event = deepcopy(event)
            if "spec_1" in event[1]: extra_event[1] = event[1].replace("spec_1", "spec")
            else: extra_event[1] = event[1].replace("spec", "spec_1")
            data1, d_data1 = fetch_data(extra_event, error=True)
            # set dict for spl.absorption_coefficient_wo_calibration() later
            if "spec_1" in event[1]: spectra = {event[2]: {"spec": data1, "spec1": data}, "Errors": {"spec": d_data1, "spec1": d_data}}
            else: spectra = {event[2]: {"spec": data, "spec1": data1}, "Errors": {"spec": d_data, "spec1": d_data1}}
            # pull absorption coefficient
            aluThickness = setPara[event[0]]["spect"][event[2]]["thickness"]
            Abs = spl.absorption_coefficient_wo_calibration(spectra, event[2], event[2], aluThickness, event[0], return_A=True, plotting=False)
            # calculate d_Abs
            # interpolate the errors from original spectra and spectra themselves. This is neccessary to line up the Abs with the errors properly
            d_trans_func = interp1d(spectra["Errors"]["spec"][0], spectra["Errors"]["spec"][1]) # interpolate a function for error of trans spectrum
            d_source_func = interp1d(spectra["Errors"]["spec1"][0], spectra["Errors"]["spec1"][1]) # interpolate a function for error of source spectrum
            trans_func = interp1d(spectra[event[2]]["spec"][0], spectra[event[2]]["spec"][1]) # interpolate a function for error of trans spectrum
            source_func = interp1d(spectra[event[2]]["spec1"][0], spectra[event[2]]["spec1"][1]) # interpolate a function for error of source spectrum
            # cut Abs E range to make sure interpolations are within range
            selection = (Abs[0]>1552) & (Abs[0]<1565) # stay within range of selection later
            Abs = Abs[:, selection] # cut down the data
            twod = 10.64 # lattice spacing of ADP in angstrom
            deff = aluThickness/np.cos(np.arcsin(12398/(Abs[0]*twod))) # calc the effective sample thickness
            d_deff_rel = 0.05 # set the relative uncertainty of the sample thickness
            d_transmission_rel = np.sqrt((d_source_func(Abs[0])/source_func(Abs[0]))**2 + (d_trans_func(Abs[0])/trans_func(Abs[0]))**2) # from gaussian error propogation
            d_Abs = np.sqrt((d_transmission_rel/deff)**2 + (Abs*d_deff_rel)**2) # also error propogation
            # set the final values
            d_data = d_Abs # set error of data to error of abs spectrum
            data = Abs # set data to the absorption spectrum
        elif edge and use_transmission: # also only really applicable to double channel spectrometers
            # fetch the data for the other arm. Use a fetch_data as enables using raw data as well
            extra_event = deepcopy(event)
            if "spec_1" in event[1]: extra_event[1] = event[1].replace("spec_1", "spec")
            else: extra_event[1] = event[1].replace("spec", "spec_1")
            data1, d_data1 = fetch_data(extra_event, error=True)
            # name the source and transmission spectra
            if "spec_1" in event[1]: source = data; d_source = d_data; trans = data1; d_trans = d_data1
            else: source=data1; d_source = d_data1; trans=data; d_trans = d_data
            # align the two spectra with each other through manually choosing
            print("Choose an E range for alignment of plots. ")
            trans, source = gl.plot_alignment(trans, source) # shifts source spectra to align with transmission
            # iron out the differences in E values, creating a common E_range while cutting down the larger E range spectrum
            trans_common, source_common = ppl.match_energies(trans, source)
            # cut E range to make sure interpolations are within range for error calculatoin
            selection = (trans_common[0]>1550) & (trans_common[0]<1568) # stay within range of selection later
            trans_common = trans_common[:, selection]; source_common = source_common[:, selection] # cut down the data
            # divide the trans by the source to get the transmission
            T = trans_common[1]/source_common[1]
            d_trans_func = interp1d(d_trans[0], d_trans[1])
            d_source_func = interp1d(d_source[0], d_source[1])
            dT = np.sqrt((trans_common[1]/source_common[1]**2*d_source_func(source_common[0]))**2 + (d_trans_func(trans_common[0])/source_common[1])**2)
            # set the data to be fitted to the intensity ratio, i.e. the transmission
            data = np.vstack((trans_common[0], T)); d_data = np.vstack((trans_common[0], dT))
        
        if manual_fitting_area: # if wish to manually choose your region of interest
            limits = gl.select_limits_with_plot(data)
            fit_lower_limit = limits[0]; fit_upper_limit = limits[1]
        else: fit_lower_limit = resolution_limits[event[2]][0]; fit_upper_limit = resolution_limits[event[2]][1]
        # now cut the data for all cases
        selection = (data[0]<fit_upper_limit) & (data[0]>fit_lower_limit) # isolate the feature
        data = data[:, selection] # cut down the data
        d_data = d_data[:, selection] # cut down the error data
        # Do the actual fitting, differentiating between a number of cases
        if edge: # fit the transmission or absorption edge. Transmission means the ratio of intensities, only really suitable for two channel specs
            idx_mean = gl.find_nearest_index(data[1], np.mean(data[1]))
            E_central = data[0][idx_mean] # location of the central point for the fit, corresponding to midpoint of edge [eV]
            c = np.mean(data[1]) # add y axis shift to error func fit
            A = min(data[1].max()-c, c-data[1].min()) # amplitude to fit to 
            FWHM, d_FWHM = full_fitting(data, d_data, Error, [A, E_central, c])
            result["Gauss|"] = round(FWHM, 4); result["d_gauss+|"] = round(d_FWHM[0], 4); result["d_gauss-|"] = round(d_FWHM[1], 4)
        else: # corresponds to fitting a peak
            E_central = data[0][data[1].argmax()] # location of the central point for the fit, corresponding to peak [eV]
            A = data[1].max() # amplitude to fit to
            c = data[1].min() # y-shift of data
            # cut the data more bc of saturation
            feature_selection = (data[0]<feature_limits[0]) | (data[0]>feature_limits[1]) # isolate the feature
            if cut_off_saturated:
                data = data[:, feature_selection] # cut down the data
                d_data = d_data[:, feature_selection] # cut down the data
            # mirror values if necessary. Fills in values on the left side past the fit_lower_limit
            if mirror_values_left:
                data_m = deepcopy(data); d_data_m = deepcopy(d_data) # initialize array for adding mirrored values
                mirror_selection = (data_m[1] < data[1][0])
                data_m = data_m[:, mirror_selection] # cut off data points with y value greater than the max of the left side
                d_data_m = d_data_m[:, mirror_selection] # do the same with error
                data_m[0] =  data_m[0][0] - data_m[0] + data[0][0] # transform the mirrored values to simulate the missing left side values
                d_data_m[0] =  d_data_m[0][0] - d_data_m[0] + d_data[0][0] 
                data = np.concatenate((data_m, data), axis=1) # add the mirrored values onto front of the data array
                d_data = np.concatenate((d_data_m, d_data), axis=1)
            if "g" in peak_fitting_func: 
                FWHM, d_FWHM = full_fitting(data, d_data, Gauss, [A-c, E_central, c]) # parameters are amplitude, then std deviation, then location of peak, then count shift
                result["Gauss|"] = round(FWHM, 4); result["d_gauss+|"] = round(d_FWHM[0], 4); result["d_gauss-|"] = round(d_FWHM[1], 4)
            elif "l" in peak_fitting_func: 
                FWHM, d_FWHM = full_fitting(data, d_data, Lorentz, [A, E_central, c]) # parameters are amplitude, then half-width at half maximum, then location of peak, then count shift
                result["Lorentz [eV]|"] = round(FWHM,4); result["d_lorentz|"] = round(d_FWHM,4)
            else: print("Error: incorrect letter in peak_fitting_func."); sys.exit()
            # if "v" in peak_fitting_func: do_fit(Voigt, data, [A, 2, E_central, 0, 2])

        
        # Now deal with source broadening. No need to do for spherically bent crystal geometries, as focusing
        if specPara[event[2]]["crystal bend"] != "spherical": 
            # define the relevant values, then calculate the source broadening analytically using the fitted source size
            C = 12398
            n = specPara[event[2]]["order"]; twod = specPara[event[2]]["lattice spacing"]
            L = specPara[event[2]]["length"]; Ecentral = specPara[event[2]]["E-central"]
            twod = twod/n
            theta0 = np.arcsin(C/(twod*Ecentral)) # central Bragg angle
            # now get source size for the event
            extra_event = deepcopy(event)
            if specPara[event[2]]["knife edge"] != "yes": extra_event[2] = spec_with_knife_edge # find spectrometer with a knife edge
            print("Choose area to calculate source size.")
            s_z, d_s_z = get_source_size(extra_event, alone = False, with_error=True) # source size [mm]. alone is False so that dont create graphs
            #s_z, d_s_z = 0.1195, 0.0121
            s_z *= 1/np.cos(theta0*np.pi/180) # accounts for projection of source onto detector due to geometry
            d_s_z *= 1/np.cos(theta0*np.pi/180)
            # get delta E from dispersion
            E = np.sqrt((1/(s_z/L + 1/np.sqrt((twod/C)**2*Ecentral**2-1))**2 + 1)*1/((twod/C)**2))
            deltaE = np.abs(E-Ecentral) # is delta E approximation for close to linear dispersion
            sigma_source = deltaE
            # along with deltaE uncertainty from gaussian error propogation
            d_sigma_source = 1/(1/(s_z/L + 1/np.sqrt((twod/C)**2*Ecentral**2-1))**2 + 1)**(1/2)*C/twod * 1/((s_z/L + 1/np.sqrt((twod/C)**2*Ecentral**2-1))**3) * 1/L * d_s_z
            # resolution due to source broadening. Directions: x is dispersive, y is vertical, and z is out of plane. Values same as in ppl.calc_E()
            result["sigma_so|"] = round(sigma_source, 4)
            result["d_so|"] = round(d_sigma_source, 4)
        # control plot appearance
        ax1.grid(which='minor', alpha=0.3)
        ax1.grid(which='major', alpha=0.8)
        geometry = event[2] # only bc I'm lazy to adjust the code from old version
        if use_title: ax1.set_title("Event "+event[0]+": "+geometry+" Resolution from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"])
        if edge and not use_transmission: 
            ax1.set_ylabel(r"Absorption Coefficient [μm$^{-1}$]")
            if use_title: ax1.set_title("Event "+event[0]+": "+geometry+" Absorption Spectrum from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"])
        # ax1.set_title("Comparison of Tb Shots of 10.9J (Event 67) and 83.7J (Event 70)")
        ax1.legend()
        # save the graph
        resol_type = "peak"
        if edge and not use_transmission: resol_type = "absorption"
        elif edge and use_transmission: resol_type = "transmission"
        # ax1.set_xlim(data[0].min()*0.999, data[0].max()*1.001) # so the graph barely misses axis
        plt.savefig(os.path.join("Graphs","resolution",resol_type+"_of_"+setPara[event[0]]["target"]+"_event_"+event[0]+"_on_"+geometry+".pdf"), dpi=1200)
        # calculate the final resolution for the spectrometer
        if not edge:
            result["Doppler"] = sigma_doppler
            result["d_dopp|"] = d_sigma_doppler
            if specPara[event[2]]["crystal bend"] != "spherical": 
                sigma_spec = np.sqrt(FWHM**2-sigma_source**2-sigma_doppler**2)
                d_spec_calc = lambda x: np.sqrt(sigma_source**2*d_sigma_source**2 + sigma_doppler**2*d_sigma_doppler**2 + FWHM**2*x**2)/sigma_spec
                d_spec = [d_spec_calc(result["d_gauss+|"]), d_spec_calc(result["d_gauss-|"])]
            else: 
                sigma_spec = np.sqrt(FWHM**2-sigma_doppler**2)
                d_spec_calc = lambda x: np.sqrt(sigma_doppler**2*d_sigma_doppler**2 + FWHM**2*x**2)/sigma_spec
                d_spec = [d_spec_calc(result["d_gauss+|"]), d_spec_calc(result["d_gauss-|"])]
        elif specPara[event[2]]["crystal bend"] != "spherical": 
            sigma_spec = np.sqrt(FWHM**2-sigma_source**2)
            d_spec_calc = lambda x: np.sqrt(sigma_source**2*d_sigma_source**2 + FWHM**2*x**2)/sigma_spec
            d_spec = [d_spec_calc(result["d_gauss+|"]), d_spec_calc(result["d_gauss-|"])]
        else: sigma_spec = FWHM; d_spec = d_FWHM
        result["sigma_sp|"] = round(sigma_spec, 4)
        result["d_sp+|"] = round(d_spec[0], 4)
        result["d_sp-|"] = round(d_spec[1], 4)
        results[str(i)] = result # save the result to the results dict
        i+=1 # iterate
    # change to dataframe for easier printing
    df_results = pd.DataFrame.from_dict(results, orient="index")
    if peak_fitting_func[0] == "g": df_results = df_results.drop(["Lorentz [eV]|", "d_lorentz|"], axis=1)
    print(df_results)
    # turn filter and detector corrections back off if not on before
    if switched_DoFilters: DoFilters = 0
    if switched_DoDetector: DoDetector = 0
    return

# find the ratio of relative reflectivity between two spectrometers.
def calculate_relative_reflectivity ():
    print("Begin relative reflectivity calculation #########################")
    global DoFilters, DoDetector, physical_units
    # turn on filter and detector corrections if not manually chosen
    switched_DoFilters = False; switched_DoDetector = False; switched_physical_units = False # variables to save if switched or not
    if not DoFilters: DoFilters = 1; switched_DoFilters = True
    if not DoDetector: DoDetector = 1; switched_DoDetector = True
    if physical_units: physical_units = 0; switched_physical_units = True # need to turn it off bc the conversion assumes a R_int
    # loop over events
    for event in events:
        # record name of first spectrometer, then find name of second. If sanity check, then use same spectrometer (double channel spec)
        # else get the second spectrometer from the shot
        spec1 = event[2]
        event2 = deepcopy(event) # second event list for second spectrometer
        if sanity_check: 
            spec2 = spec1
            if event[1] == "spec": event2[1] = "spec_1" # make second event take other channel of spectrometer
            else: event2[1] = "spec"
        else: 
            for spec in setPara[event[0]]["spect"]:
                if spec != spec1: spec2 = spec
        # pull data2 by creating new event list
        event2[2] = spec2
        # so that if second spec is a single channel spec, then is handled correctly
        if specPara[event[2]]["channels"] == "double" and event[1] == "spec_1" and specPara[event2[2]]["channels"] == "single": event2[1] = "spec" 
        # fetch the data
        i = 0 # looping variable
        for temp_event in [event, event2]:
            if i == 0: data1, d_data1 = fetch_data(temp_event, error=True, sys_error=True)
            else: data2, d_data2 = fetch_data(temp_event, error=True, sys_error = True)
            i += 1
        # define the distance over which the rays are allowed to diverge in dist variable
        specs = [spec1, spec2] # facilitates looping
        dists = [] # where the distances will be saved
        i = 0 # looping variable
        for spec in specs:
            if not i: data = data1 # choose the corresponding data
            else: data = data2
            dists = get_diverging_distances(spec, data, dists)
            i += 1
        # now have both datasets as N_ph
        beta = [] # initialize beta list
        d_beta = [] # uncertainty of beta
        if use_derivative:
            # beta corresponds to dN/dE = N_px/deltaE, where N_px=N_ph. Note that the first deltaE value in duplicated
            # to keep length of arrays the same
            beta.append(data1[1]/np.insert(np.diff(data1[0]), 0, data1[0][1]-data1[0][0]))
            beta.append(data2[1]/np.insert(np.diff(data2[0]), 0, data2[0][1]-data2[0][0]))
        else:  # if want to calculate from sum over an energy range. In this case beta is a single float value
            print("\nSelect limits for E range to integrate over:")
            limit1 = gl.select_limits_with_plot(data1, labels=[specs[0]])
            limit2 = gl.select_limits_with_plot(data2, labels=[specs[1]])
            #limit1 = [1583.0, 1610.0]; limit2 = [1583.0, 1610.0]
            # save the uncut data for graphing later
            uncut_data1 = deepcopy(data1)
            uncut_data2 = deepcopy(data2)
            # cut down data according to plots
            condition1 = (data1[0] > limit1[0]) & (data1[0] < limit1[1])
            condition2 = (data2[0] > limit2[0]) & (data2[0] < limit2[1])
            data1 = data1[:, condition1]
            data2 = data2[:, condition2]
            # add to get beta
            beta.append(np.sum(data1[1]))
            beta.append(np.sum(data2[1]))
            # get the error of beta in form d_beta = sqrt(sum(d_data1**2)). Cut down first
            d_data1 = d_data1[:, condition1]
            d_data2 = d_data2[:, condition2]
            d_beta.append(np.sqrt(np.sum(d_data1[1]**2)))
            d_beta.append(np.sqrt(np.sum(d_data2[1]**2)))
            # find the average distance of the chosen range
            av_dist1 = dists[0][gl.find_nearest_index(data1[0],(limit1[1]+limit1[0])/2)]
            av_dist2 = dists[1][gl.find_nearest_index(data2[0],(limit2[1]+limit2[0])/2)]
            # compute the dists error
            d_dists = [] # next calc gives the max deviation from the average distance within the limits
            d_dists.append(max(np.abs(av_dist1-dists[0][gl.find_nearest_index(data1[0],limit1[0])]), np.abs(av_dist1-dists[0][gl.find_nearest_index(data1[0],limit1[1])])))
            d_dists.append(max(np.abs(av_dist2-dists[1][gl.find_nearest_index(data2[0],limit2[0])]), np.abs(av_dist2-dists[1][gl.find_nearest_index(data2[0],limit2[1])])))
            # add on an absolute error from various systematic errors of the setup
            d_dists_sys = 5 # [mm]
            d_dists[0] += d_dists_sys; d_dists[1] += d_dists_sys
            # set to the dists list so that can be used later on and save old dists for plotting later
            full_dists = deepcopy(dists)
            dists[0] = av_dist1
            dists[1] = av_dist2

        # now calculate the result for each spectrometer. variable result = N_total/R_int
        result = [] # initialize result list
        d_result = [] # list of the result uncertainties
        i = 0 # looping variable for index of lists
        plt.figure()
        plt.grid(alpha=0.5, zorder=1)
        for spec in specs:
            # for labeling purposes
            geometry = spec # again, am lazy to change it
            add_label = ""
            # add extra str to label to denote that second channel for a double channel spectrometer
            if specPara[spec]["channels"] == "double" and event[1].find("_1") != -1: add_label = " Spectrum 2" 
            if specPara[spec]["channels"] == "double" and event[1].find("_1") == -1: add_label = " Spectrum 1"
            # run calculation for the type of crystal bending
            if specPara[spec]["crystal bend"] == "flat":
                result.append(beta[i]*4*np.pi*dists[i]/delta_pix)
                d_result.append(4*np.pi/delta_pix*np.sqrt((dists[i]*d_beta[i])**2 + (beta[i]*d_dists[i])**2))
                # now do graphing
                if not i: uncut_data = uncut_data1 # choose the corresponding uncut_data
                else: uncut_data = uncut_data2
                # z_order ensures that the actual plotted lines are in the foreground of graph
                z_order = 2
                if specPara[spec]["channels"] == "double": z_order = 3
                plt.plot(uncut_data[0], uncut_data[1]*4*np.pi*full_dists[i]/delta_pix, label = geometry + add_label, zorder = z_order)
            elif specPara[spec]["crystal bend"] == "spherical":
                crystal_width = specPara[spec]["crystal width"] # width of crystal in mm
                # if wish to add an artificial error
                error_poor_crystal_rel = 0.2 # 20 percent error due to crystal defects
                d_beta[i] += error_poor_crystal_rel * np.mean(beta[i])
                result.append(beta[i]*4*np.pi*dists[i]/crystal_width)
                d_result.append(4*np.pi/crystal_width*np.sqrt((dists[i]*d_beta[i])**2 + (beta[i]*d_dists[i])**2))
                # and do additional calculation for FSSR with artems uncut_data
                if not i: uncut_data = uncut_data1 # choose the corresponding uncut_data
                else: uncut_data = uncut_data2
                # graphing of basic result
                plt.plot(uncut_data[0], uncut_data[1]*4*np.pi*full_dists[i]/crystal_width, label = geometry + add_label, zorder=3)
                # calculation of result with efficiency
                if not i: data = data1 # choose the corresponding data
                else: data = data2
                scriptdir = os.path.dirname(os.path.realpath(__file__))
                filepath = os.path.join(scriptdir, "FSSR", "2023_10_01_FSSR_mica_refl.csv")
                eff = gl.extract_csv(filepath, efficiency=True) # grab the efficiency from file
                eff_func = interp1d(eff[0], eff[1])
                N_ph_total_eff = np.sum(data[1]/eff_func(data[0]))
                R_int_artem = beta[i]*4*np.pi*dists[i]/crystal_width / N_ph_total_eff
                print("R_int from FSSR (event "+str(event[0])+") using theoretical efficiency: "+str(round(R_int_artem*10**6,3))+"+-"+str(round(d_result[i]/N_ph_total_eff*10**6, 3))+" vs. R_lit = 53.6")
            i += 1
        # let user know which event we're talking about
        print("\nR_int ratio results for event "+str(event[0]))
        # now do ratio calculations
        if use_derivative: # plot the results if using derivative, since get in an E range
            ratio = np.zeros(data1[0].shape) # prep ratio array
            # plot the results individually to judge if final ratio looks reasonable
            plt.plot(data1[0], result[0], label = spec1)
            plt.plot(data2[0], result[1], label = spec2)
            plt.figure() # make new figure for plots of results
            # calculate the ratio for each energy value then plot it. Check which spectrum is longer, then match shorter to longer spetrum
            if data1[0][-1] - data1[0][0] < data2[0][-1] - data2[0][0]:  
                for i in range(len(data1[0])):
                    ratio[i] = result[0][i]/result[1][gl.find_nearest_index(data2[0], data1[0][i])]
                plt.plot(data1[0], ratio, label = "Experimental")
            else:
                for i in range(len(data2[0])):
                    ratio[i] = result[0][gl.find_nearest_index(data2[0], data1[0][i])]/result[1][i]
                plt.plot(data2[0], ratio, label = "Experimental")
        else: # only print out the ratio, as its a single value in this case
            ratio = result[0]/result[1]
            d_ratio = np.sqrt((d_result[0]/result[1])**2 + (result[0]/result[1]**2*d_result[1])**2)
            print(r"Experimental R_int ratio ("+specs[0]+"/"+specs[1]+"): "+str(np.round(ratio, 3))+"+-"+str(np.round(d_ratio, 3)))
        lit_R_int = [] # literature R_int values [micro rad]
        for spec in specs:
            lit_R_int.append(specPara[spec]["R_int"]) 
        print(r"Theoretical R_int ratio ("+specs[0]+"/"+specs[1]+"): "+str(lit_R_int[0]/lit_R_int[1]))
        # if using derivative, then draw a line on plot for literature value
        if use_derivative:
            plt.xlabel("Energy [eV]")
            plt.ylabel("Ratio of Integrated Reflectivities [-]")
            plt.axhline(y = lit_R_int[0]/lit_R_int[1], color = 'r', linestyle = '-', alpha = 0.3, label = "Literature")
            plt.legend()
        plt.axvspan(limit1[0], limit2[1], alpha = 0.2, label = "Integration area", color="palevioletred")
        plt.legend()
        plt.xlabel("Photon Energy [eV]")
        plt.ylabel(r"Integrated Reflectivity times Number of Photons [rad]")
        x_min = uncut_data1[0].min()
        x_max = uncut_data1[0].max()
        plt.xlim(x_min*0.999, x_max*1.001)
        plt.ylim(bottom=uncut_data2[1].min()*10000)
        plt.rcParams['axes.axisbelow'] = True
        # plt.title("Event "+event[0]+r": Spectra from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"])
        #plt.title("Spectra of "+setPara[event[0]]["target"]+" at Various Laser Energies")
        # make string of event numbers to identify the save file
        events_str = ""
        for event in events: events_str += event[0]+"_"
        events_str = events_str[:-1]
        plt.savefig(os.path.join("Graphs","R_int_ratio","spectra_of_"+setPara[event[0]]["target"]+"_event_"+events_str+".pdf"))
    # turn filter and detector corrections back off if not on before
    if switched_DoFilters: DoFilters = 0
    if switched_DoDetector: DoDetector = 0
    if switched_physical_units: physical_units = 1 # turn back on if turned off before
    return

# calculate the absorption coefficient, allowing for further options (filters, binning etc) since in data analysis
# manually pass do_crystal_calibration to avoid crossover of dal and spl parameters.
def calculate_absorption_coefficient (do_crystal_calibration):
    print("Begin absorption coefficient output #########################")
    # make inner function to pull data from additional event that arent explicit in events list. Also need to 
    # conform to spl.absorption_coefficient syntax
    def get_data_as_dict (event, transSpecName, sourceSpecName, spectraDict):
        event_extra = deepcopy(event)
        event_extra[2] = sourceSpecName
        if specPara[transSpecName]["channels"] != "double": # if not double channel, then get source spectrum from other spectrometer
            spectraDict[transSpecName] = {}; spectraDict[sourceSpecName] = {} # initialize keys
            # get the spectra. First the transmission spectrum
            spectraDict[transSpecName]["spec"] = fetch_data(event) # need the "spec" to conform to spl.absorption_coefficient syntax
            if event[1] == "spec_1": event_extra[1] = "spec" # needed to prevent bugs if use extra channels for spec in event
            spectraDict[sourceSpecName]["spec"] = fetch_data(event_extra)
        else: # so if spectrometer is double channel
            spectraDict[transSpecName] = {} # initialize transSpecName key
            if event[1] == "spec":
                spectraDict[transSpecName]["spec"] = fetch_data(event)
                event_extra[1] = "spec_1" # means non chosen channel is source
                spectraDict[transSpecName]["spec1"] = fetch_data(event_extra)
            if event[1] == "spec_1":
                spectraDict[transSpecName]["spec1"] = fetch_data(event) 
                event_extra[1] = "spec" # means non chosen channel is transmission
                spectraDict[transSpecName]["spec"] = fetch_data(event_extra)
        return spectraDict
    # begin processing
    for event in events:
        if '_raw' in event[1]: 
            print("Error: Raw spectra can't be used to get absorption spectrum for event " +event[0]+ ".")
            continue # don't do anything if using raw spectra
        eventNum = event[0]; specName = event[2]
        # start analysis. if spectrometer in list doesnt have type "abs", then skip it
        if (setPara[eventNum]["spect"][specName]["type"] == "abs"):
            # get the names of the spectrometers
            transSpecName = specName # rename for clarity 
            for spec in setPara[event[0]]["spect"]: # get the first spectrometer name. note that for double channel this is not used
                if spec != transSpecName: sourceSpecName = spec
            if specPara[transSpecName]["channels"] == "double": sourceSpecName = transSpecName # if double channel then replace source name
            # initialize dict to hold spectra. pull the spectra
            spectra = {}
            spectra = get_data_as_dict(event, transSpecName, sourceSpecName, spectra)
            # get aluminum thickness
            aluThickness = setPara[eventNum]["spect"][transSpecName]["thickness"]
            if do_crystal_calibration:
                # grab the calibration event number and data
                calEventNum = spl.get_calibration_event_num(eventNum, transSpecName)
                calEvent = deepcopy(event)
                calEvent[0] = calEventNum
                calSpectra = {}
                calSpectra = get_data_as_dict(calEvent, transSpecName, sourceSpecName, calSpectra)
                # run calculation of absorption coefficient, as well as plot everything. plotting is done in function since it plots stuff from each step
                spl.absorption_coefficient(spectra, calSpectra, transSpecName, sourceSpecName, aluThickness, eventNum, calEventNum, setPara, isolated_plot=True, final_result=True, ylabel=ylabel)
            # otherwise get absorption coefficient without calibratin
            else: spl.absorption_coefficient_wo_calibration(spectra, transSpecName, sourceSpecName, aluThickness, eventNum, final_result=True, ylabel=ylabel)
    return

# determine conversion efficiency for an emission line. Make sure not to bin for this
def calculate_conversion_efficiency ():
    print("Begin conversion efficiency calculation #########################")
    global DoFilters, DoDetector, physical_units
    # turn on filter and detector corrections if not manually chosen
    switched_DoFilters = False; switched_DoDetector = False; switched_physical_units=False # variables to save if switched or not
    if not DoFilters: DoFilters = 1; switched_DoFilters = True
    if not DoDetector: DoDetector = 1; switched_DoDetector = True
    if physical_units: physical_units = 0; switched_physical_units = True # turn off bc I dont want to reprogramm stuff and still works just as well
    results = [] # list to record results of calculations
    d_results = [] # uncertainty of results
    event_nums = [] # for event numbers
    spec_used = [] # for the corresponding spectrometer
    line_energies = [] # for the energies of the lines used
    for event in events:
        event_nums.append(event[0]); spec_used.append(event[2]) # gather the event info
        data, d_data = fetch_data(event, error=True, sys_error=True)
        # extract the laser energy from the event [J]
        l_energy = setPara[event[0]]["energy"]
        # choose a line for the calculation
        print("\nSelect line for the conversion efficiency:")
        limit = gl.select_limits_with_plot(data, labels=[event[0] + " on " + event[2]])
        #limit = [1591.4, 1604.3]
        # save data for graphing later
        uncut_data = deepcopy(data)
        # integrate over line to get number of photons in that line on detector. Works since corrections make y axis into N_ph
        data = data[:, (data[0] > limit[0]) & (data[0] < limit[1])] # cut down data according to limit
        d_data = d_data[:, (d_data[0] > limit[0]) & (d_data[0] < limit[1])] # cut down error according to limit
        # if want to manually input an error
        if event[2] == "FSSR":
            error_poor_crystal_rel = 0.2 # relative error due to poor mica quality
            d_data[1] += error_poor_crystal_rel*np.mean(data[1]) # increase error of photon count respectively
        # do calculation
        N_ph_detector = np.sum(data[1]) # get photon landing on detector for line
        d_N_detector = np.sqrt(np.sum(d_data[1]**2)) # and the uncertainty
        # now extract the total number of photons emitted by the source in that energy range by assuming a R_int value
        line_energy = data[0][np.argmax(data[1])] # get the energy of the line by finding energy of max counts
        d_line_energy = 1 # estimated error of line energy due to broadening and dispersion [eV]
        dists = [] # initialize dists list for use in getting total num of photons
        spec = event[2] # get the literature R_int value [micro rad]
        R_int = specPara[spec]["R_int"]
        R_int *= 10**(-6) # convert to rad
        # manual relative error of R_int
        d_R_int = R_int*0.5 # 50% relative error estimated from ratio of R_int results for KAP and ADP
        # decide over what length the rays are being collected
        if specPara[spec]["crystal bend"] == "spherical": collecting_length = specPara[spec]["crystal width"]
        else: collecting_length = delta_pix # for flat crystal spectrometers
        # calculate the total number of photons
        disp_dist = get_diverging_distances(spec, np.array([[line_energy],[max(data[1])]]), dists)[0][0] # distance over which the rays of the He alpha energy can disperse
        d_dist = 5 # systematic error of distance [mm]
        N_ph_total = N_ph_detector*4*np.pi*disp_dist / (R_int*collecting_length)
        d_N_ph_total = 4*np.pi/(R_int*collecting_length)*np.sqrt((disp_dist*d_N_detector)**2 + (N_ph_detector*d_dist)**2 + (N_ph_detector*disp_dist/R_int *d_R_int)**2)
        # calculate the conversion efficiency. Convert l_energy to eV and multiply photon num by photon energy to get total energy in line
        conversion_efficiency = (N_ph_total * line_energy) / (l_energy*6.242*10**18)
        d_conversion_eff = 1/(l_energy*6.242*10**18)*np.sqrt((d_N_ph_total*line_energy)**2 + (N_ph_total*d_line_energy)**2)
        results.append(round(conversion_efficiency, 6)) # append to results
        d_results.append(round(d_conversion_eff, 6))
        line_energies.append(round(line_energy, 3)) # record which line is used
        plt.figure()
        plt.grid(zorder=1)
        # also do calculation using efficiency (reflectivity) from Artem
        if spec == "FSSR": 
            scriptdir = os.path.dirname(os.path.realpath(__file__))
            filepath = os.path.join(scriptdir, "FSSR", "2023_10_01_FSSR_mica_refl.csv")
            eff = gl.extract_csv(filepath, efficiency=True) # grab the efficiency from file
            eff_func = interp1d(eff[0], eff[1])
            N_ph_total_eff = np.sum(data[1]/eff_func(data[0]))
            d_N_total_eff = np.sqrt(np.sum((d_data[1]/eff_func(d_data[0]))**2)) # and the uncertainty
            CE_eff = (N_ph_total_eff * line_energy) / (l_energy*6.242*10**18)
            d_CE_eff = 1/(l_energy*6.242*10**18)*np.sqrt((d_N_total_eff*line_energy)**2 + (N_ph_total_eff*d_line_energy)**2)
            print("CE from FSSR (event "+str(event[0])+") using theoretical efficiency: "+str(round(CE_eff, 5))+"+-"+str(round(d_CE_eff, 5)))
        # plot the process
        # for labeling purposes
        geometry = spec
        add_label = ""
        if event[1].find("_1") != -1: add_label = " Spectrum 2" # add extra str to label to denote that second channel for a double channel spectrometer
        if specPara[event[2]]["channels"] == "double" and event[1].find("_1") == -1: add_label = " Spectrum 1"
        # get the energy interval of the CCD camera in order to get physical units on the y axis
        delta_E = uncut_data[0][2]-uncut_data[0][1] # use the energy separation of first two data points, excepting the very first point
        if DoBinning: delta_E /= numPixelPerBin # since take average over the binned pixel
        if specPara[event[2]]["crystal bend"] == "flat":
            plt.plot(uncut_data[0], uncut_data[1]*4*np.pi*disp_dist / (R_int*collecting_length*delta_E), label=geometry+add_label)
        if event[2] in ("FSSR"):
            # cut the data so interpolation range works
            uncut_data = uncut_data[:, (uncut_data[0] > eff[0].min()) & (uncut_data[0] < eff[0].max())]
            plt.plot(uncut_data[0], uncut_data[1]/eff_func(uncut_data[0])/delta_E, label=geometry+add_label)
        plt.axvspan(limit[0], limit[1], alpha = 0.2, label = "Integration area", color="palevioletred")
        plt.legend()
        plt.xlabel("Photon Energy [eV]")
        plt.ylabel(r"Number of Photons $\left[\frac{1}{\text{eV}}\right]$")
        x_min = uncut_data[0].min()
        x_max = uncut_data[0].max()
        plt.xlim(x_min*0.999, x_max*1.001)
        plt.ylim(bottom=uncut_data[1].min()*10000)
        plt.rcParams['axes.axisbelow'] = True
        # plt.title("Event "+event[0]+": Total Emission Spectrum from "+str(setPara[event[0]]["energy"])+"J Shot on "+setPara[event[0]]["target"])
        #plt.title("Spectra of "+setPara[event[0]]["target"]+" at Various Laser Energies")
        plt.savefig(os.path.join("Graphs","Conversion_efficiency","spectra_of_"+setPara[event[0]]["target"]+"_event_"+event[0]+"_on_"+geometry+"_"+add_label+".pdf"), bbox_inches="tight")
    # change to dataframe for easier printing
    df_results = pd.DataFrame(np.transpose(np.vstack((event_nums, spec_used, line_energies, results, d_results))), columns=["Event" , "Spec", "E of line [eV]", "CE [-]", "d_CE"])
    print(df_results)
    # turn filter and detector corrections back off if not on before
    if switched_DoFilters: DoFilters = 0
    if switched_DoDetector: DoDetector = 0
    if switched_physical_units: physical_units = 1 # turn back on if turned off before
    return