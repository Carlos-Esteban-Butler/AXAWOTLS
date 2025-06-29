# this library is responsible for the post processing of the data, including the conversion of pixel into E

import os
import numpy as np
import sys
import json
from scipy.interpolate import interp1d
from copy import deepcopy

import general_lib as gl
import spectrum_processing_lib as spl

setPara = gl.import_event_parameters()
specPara = gl.import_event_parameters(get_spectrometer_parameters=True)

# grab the parameters from spectrum_processing_lib
ppl_placeholder = 0 # replace this if need a parameter eventually
def pass_parameters(*post_process_p):
    global ppl_placeholder # make global
    for parameter in post_process_p: # set the int values to bool
        if parameter == 0: parameter = False
        elif parameter == 1: parameter = True
    ppl_placeholder = post_process_p # set parameters
    return

# now define energy conversions based on dispersion and general functions for data processing. --------------------------------------------------------------------------------------
# E offset is defined by user after seeing spectra (comes from random small shifts in experimental setup)
# kwarg extra is used for double channel spectrometers. For unfocusing a spherically bent crystal spectrometer, just make a new disp number
def calc_E(data, eventNum, specName, dispNum, E_offset, extra=False):
    C = 12398 # Umrechnungsfaktor of wavelength [angstrom] to energy [eV]
    E_offset = float(E_offset) # ensure that E_offset is a float
    # import d_shift dict
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    dshiftpath = os.path.join(scriptdir, "Parameters", "d_shift.json")
    with open(dshiftpath, 'r') as f:
        disp = json.load(f)
    # set d_shift according to event by disp num
    if extra: d_shift_list = disp[specName+" extra"]
    else: d_shift_list = disp[specName]
    if len(d_shift_list) < dispNum: d_shift_list.append(0) # add an element to list if created a new dispersion number
    d_shift = d_shift_list[dispNum-1] # grab the d_shift
    # now run the calculations
    if (specPara[specName]["crystal bend"] == "spherical"): # if the FSSR is unfocused, just make a new dispersion number and copy the d_shift from previous shot
        # values for coefficients of quadratic dispersion calculated by mmpxrt. Focused and unfocused (by 5mm increase in b0) case
        if setPara[eventNum]["spect"][specName]["focus"] == "yes": a = -0.02193; b = 10.57; c = 1600
        elif setPara[eventNum]["spect"][specName]["focus"] == "no": a = -0.00867; b = 9.62; c = 1600 
        else: print("Error: parameter focus in set.json is undefined"); sys.exit()
        d = d_shift - (data[0]-1023.5)*(-0.01347) # minus because of the difference in definition of d between mmpxrt and mine
        d_calc = lambda E_i: -(-b + np.sqrt(b**2-4*a*(c - E_i)))/(2*a) # - to compensate difference in d definitions
        # shift using user defined offset. Useful if want to do non standard d_shift determination/fine tuning
        if E_offset != 0: d -= d_calc(E_offset+1600) # plus the central energy, since d(E_central) = 0. Brings it into the correct coor system
        E = a*d**2 + b*d + c
    # now for flat crystal spectrometers. Since dispersion formula is the same for our cases, only need to pull the spectrometer parameters
    if (specPara[specName]["crystal bend"] == "flat"):
        n = specPara[specName]["order"]; twod = specPara[specName]["lattice spacing"]/n
        L = specPara[specName]["length"]; Ecentral = specPara[specName]["E-central"]
        # run the calculation with analytical dispersion formula
        d = (data[0]-1023.5)*(-0.01347) - d_shift
        F = 1/np.sqrt((twod/C)**2*Ecentral**2-1)
        D = (twod/C)
        d_calc = lambda E_i: L*(1/np.sqrt((twod/C)**2*E_i**2-1) - 1/np.sqrt((twod/C)**2*Ecentral**2-1))
        # plus the central energy, since d(E_central) = 0. Brings it into the correct coor system
        if E_offset != 0: d += d_calc(E_offset+Ecentral) # shift using user defined offset. Useful if want to do non standard d_shift determination/fine tuning
        E = np.sqrt((1/(d/L + F)**2 + 1)*1/(D**2))

    # check and calculate d_shift if needed. Calculated by finding He alpha singlet line of Al at 1598.4 eV
    if d_shift == 0:
        user_disp_confirmation = input("\nCalculating new dispersion shift for "+specName+". Make sure you are using the correct Al shot. Continue? (y or n): ")
        if (user_disp_confirmation != "y"): sys.exit()
        else:
            E_disp_0 = E
            disp_0_func = interp1d(E_disp_0, data[1]) # interpolate a function from the d_shift=0 data
            E_test = np.linspace(E_disp_0.min(), E_disp_0.max(), int((E_disp_0.max()-E_disp_0.min())*100)) # test every 0.001 eV
            E_test = E_test[(E_test < 1670) & (E_test > 1530)] # keep within 70eV of HeAlpha on each side
            E_HeAlphaPeak = E_test[disp_0_func(E_test).argmax()] # finds index of max counts, then uses index to get corresponding E value
            d_shift = d_calc(E_HeAlphaPeak) - d_calc(1598.4)
            # dump new d shift into the json file to load later
            d_shift_list[dispNum-1] = d_shift
            if extra:
                disp[specName+" extra"] = d_shift_list
            else: 
                disp[specName] = d_shift_list
            with open(dshiftpath, 'w') as f: 
                json.dump(disp, f)
            # rerun the function with the new disp dictionary
            data = calc_E(data, eventNum, specName, dispNum, E_offset, extra)
            return data
    # set x axis of data
    data[0] = E
    return data

# ===============================================================================================================================
# the following functions are neccessary for the abs spectroscopy.

# cut off spectra according to a min and max energy
def cut_spectra(array, minE, maxE):
    # cut off top end of spectrum
    ShortArray = np.delete(array, np.arange(gl.find_nearest_index(array[0,:], maxE), len(array[0,:])), axis=1)
    # cut off low end of spectrum
    ShortestArray = np.delete(ShortArray, np.arange(0 ,gl.find_nearest_index(array[0,:], minE)), axis=1)
    return ShortestArray

# matches the E values of spectrum with smaller E range to larger
def match_energies(spec1, spec2): # minE is minimum of smaller E range. Same for maxE
    switched = False
    # if spec1 E range is smaller than spec2, switch the roles
    if (spec1[0].max() - spec1[0].min()) < (spec2[0].max() - spec2[0].min()):
        temp = spec1
        spec1 = spec2
        spec2 = temp
        switched = True
    spec1_common = cut_spectra(spec1, spec2[0].min(), spec2[0].max())
    spec2_common = np.zeros(spec1_common.shape)
    spec2_common[0, : ] = spec1_common[0, :]
    for i in range(0, len(spec2_common[0])):
        idx_closest = gl.find_nearest_index(spec2[0, :], spec2_common[0, i])
        spec2_common[1, i] = spec2[1, idx_closest]
    if (switched): # if switched above, then return the correct spectra
        return spec2_common, spec1_common
    return spec1_common, spec2_common

# this function lines up all the spectra in an absorption spectrum calculation, then shifts all equally according to the K-edge location
# effectively, this removes the source location flucuations between shots. Use carefully, as it also removes any edge shifts, hiding a potential info source.
# E_test_range denotes the range of energies to test for optimal config
def line_up_spectra(trans, source, calTrans, calSource, E_test_range = 15, isolated_plot = False):
    # save the original signals for output later
    trans_output = deepcopy(trans[1]); source_output = deepcopy(source[1]); calTrans_output = deepcopy(calTrans[1]); calSource_output = deepcopy(calSource[1])
    # first normalize the spectra intensities so that Sigmal_max = 1 while signal > 0
    normalize = lambda signal: signal/signal.max()
    trans[1] = normalize(trans[1]); source[1] = normalize(source[1]); calTrans[1] = normalize(calTrans[1]); calSource[1] = normalize(calSource[1])
    # then line up the spectra by finding the configuration with the smallest difference between spectra
    # hold second spectrum still, then shift the first until it lines up the best, given by the smallest sum of differences, normalized to num of data points
    def find_optimal_E_shift(spec1, spec2):
        E_shifts = np.linspace(-E_test_range/2, E_test_range/2, E_test_range*4) # test each 0.25eV in given test range
        sums = [] # will have each sum for each E shift
        for E_shift in E_shifts:
            summe = 0; N = 0 # summe is sum of absolute diff of spectra signals, while N is number of data points in sum
            test_spec1 = np.vstack((spec1[0] + E_shift, spec1[1]))
            for i in range(len(spec1[0])):
                idx, dist = gl.find_nearest_index(spec2[0], test_spec1[0][i], distance=True)
                if dist < test_spec1[0].sum()/len(test_spec1[0]): # if the dist becomes larger than average E point difference in spectrum, then dont use
                    N += 1 # add to num of data points used
                    summe += np.abs(test_spec1[1][i] - spec2[1][idx]) # contribute to sum of differences
            sums.append(summe/N) # save normalized sum
        return E_shifts[np.array(sums).argmin()] # return the E shift corresponding to the minimum of sum of normalized signal diff
    
    # first align the calibration and current shot source spectra
    calSource[0] += find_optimal_E_shift(calSource, source)
    # then align the transmission and calTrans shots, meaning need to cut off the energies after the K-edge
    deriv_test_trans = trans[:, np.abs(trans[0] - 1558.98) < 10] # for finding the edge using derivative, within 20eV of theoretical edge
    # edge found by finding the location of largest derivative, then identifying the corresponding energy, then getting index for that E in original array
    idx_most_neg_derivative = gl.find_nearest_index(trans[0], deriv_test_trans[0][np.diff(deriv_test_trans[1]).argmin()])
    trans[0] -= trans[0][idx_most_neg_derivative] - 1558.98 # bring the edge approximately into the right place
    test_trans = trans[:, trans[0] < trans[0][idx_most_neg_derivative] - 1.5] # take only E values 1.5 eV below edge (the 1.5 eV accomodates average width of edge)
    test_calTrans = calTrans[:, calTrans[0] < calTrans[0][idx_most_neg_derivative] - 1.5]
    test_calTrans[1] = normalize(test_calTrans[1]) # renormalize to give better agreement with the trans spectrum
    calTrans[0] += find_optimal_E_shift(test_calTrans, test_trans)
    
    # # finally line up the cal transmission spectrum with both source spectra. For this need to manually choose an energy range
    # print("\nChoose an energy range to align the transmission and calibration transmission with: ")
    # # select the limits for the alignment with a plot. Not an isolated plot since no plot has yet been opened. Isolated_plot = True will keep all figures open after function call
    # limits = gl.select_limits_with_plot(*[trans, calTrans], labels = ["transmission", "cal trans"], isolated_plot = isolated_plot)
    # # now align the spectra within the new limits, while renormalizing them to the same level
    # test_trans = trans[:, (trans[0] > limits[0]) & (trans[0] < limits[1])]
    # test_calTrans = calTrans[:, (calTrans[0] > limits[0]) & (calTrans[0] < limits[1])]
    # test_calTrans[1] = normalize(test_calTrans[1]); test_trans[1] = normalize(test_trans[1])
    # calTrans[0] += find_optimal_E_shift(test_calTrans, test_trans)


    # finally line up the cal transmission spectrum with both source spectra. For this need to manually choose an energy range
    print("\nChoose an energy range to align the two source spectra and calibration transmission spectrum with: ")
    # select the limits for the alignment with a plot. Not an isolated plot since no plot has yet been opened. Isolated_plot = True will keep all figures open after function call
    limits = gl.select_limits_with_plot(*[calTrans, calSource], labels = ["cal Transmission", "cal Source"], isolated_plot = isolated_plot)
    # now align the spectra within the new limits, while renormalizing them to the same level
    test_calTrans = calTrans[:, (calTrans[0] > limits[0]) & (calTrans[0] < limits[1])]
    test_calSource = calSource[:, (calSource[0] > limits[0]) & (calSource[0] < limits[1])]
    test_calSource[1] = normalize(test_calSource[1]); test_calTrans[1] = normalize(test_calTrans[1])
    E_shift_final = find_optimal_E_shift(test_calSource, test_calTrans)
    calSource[0] += E_shift_final
    source[0] += E_shift_final

    # recover the original signals
    trans[1] = trans_output; source[1] = source_output; calTrans[1] = calTrans_output; calSource[1] = calSource_output
    return trans, source, calTrans, calSource

#===============================================================================================================================
# now follows the functions for post processing specifically
# simple background correction
def correct_background(Spec, Back, specName, eventNum):
    if specPara[specName]["crystal bend"] == "spherical": # since for focusing spectrometers the pixels are summed over during spectrum extraction,
        # need a separate behaviour 
        box_data = spl.control_box_data(eventNum, specName, read = "spec")
        num_rows_spectrum = np.abs(box_data[0][1] - box_data[0][0]) # number of rows that were summed over to find spectrum
        Back[1] *= num_rows_spectrum # need to do this as averaged over background pix rows, but summed over spectrum pix rows
    CorrX = Spec[0]
    CorrY = Spec[1] - Back[1]
    return np.vstack((CorrX,CorrY))

    