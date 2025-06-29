import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
import re
import json
from copy import deepcopy
import sys
from scipy.signal import savgol_filter


# for reading the csv files, outputs the data as a np.array. Additional kwargs for various specific csv formats of different data
def extract_csv (file_path, filter=False, camera=False, efficiency=False, abs_lit=False):
    delimiter = ","
    if filter: delimiter = " " # if filter, then delimiter is a space
    if abs_lit: delimiter = ";" # if literature absorption values, then ;
    data = []
    # check if data exists
    if not os.path.isfile(file_path): 
        print("Data not available for "+file_path+". Make sure new_data is on.")
        sys.exit()
    # read in the csv files
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        if (filter) or efficiency: # if filter or FSSR efficiency csvs, then skip first two lines
            next(csv_reader)
            next(csv_reader)
        for row in csv_reader:
            data.append(row)
    if camera: data = np.transpose(data)
    if filter: # if filter, then delete all empty elements in list and format to x-values in first row and y in second.
        d = [float(i) for j in data for i in j if i != '']
        data1 = [0,0]
        data1[0] = d[::2]
        data1[1] = d[1::2]
        return np.array(data1).astype(float)
    if efficiency: 
        data = np.transpose(data)
        data = data[1:]
    if abs_lit:
        data = np.transpose(data)
        data = data[:, data[0].argsort()] # sort the energy values to avoid weird behavior when plotting
        data[1]=savgol_filter(data[1], 8, 3) # smooths out the data
    return np.array(data).astype(float)

# creates or extracts the path to a data directory in the event directories
def make_datadir (specName, currentEventNum):
    if(float(currentEventNum) < 10):
        extraZeros = "000"
    else: extraZeros = "00"
    currenteventdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Events", extraZeros + currentEventNum)
    # then make the path to data directory of each spectrometer if it doesnt exist
    datadir = os.path.join(currenteventdir, specName, "Data")
    if not os.path.isdir(datadir):
        os.makedirs(datadir)
    return datadir

# general function for pulling data from csv created by main code
def pull_data (eventNum, csv, specName):
    datadir = make_datadir(specName, str(eventNum))
    datafilepath = os.path.join(os.path.join(datadir, csv+".csv"))
    if not os.path.isfile(datafilepath): 
        print("Error: " + datafilepath +" doesn't exist. Please create the data.")
        sys.exit()
    data = extract_csv(datafilepath)
    data = np.array(data)
    return data

# find the index in an array nearest to a value. Can also return distance to closest value if desired
def find_nearest_index(array, value, distance = False):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    if distance:
        return idx, np.abs(array[idx] - value)
    return idx

# this function allows simultaneous input while graph is open. Isolated plot means that will not close all existing plots,
# but will require an extra "enter" input by the user. Labels input in order that spectra are given
def select_limits_with_plot(*spectra, labels = [], isolated_plot = True):
    limits = []
    if labels == []: # if no labels defined, do general labels
        i = 1
        for spectrum in spectra: labels.append("Spectrum " + str(i)); i += 1
    def submit(chosen_range): # function to call upon submission
        nonlocal limits # so that the limits can be changed in the main function
        limits = [float(i) for i in re.findall(r"[-+]?(?:\d*\.*\d+)", chosen_range)] # read the input into a list, changing into float
        plt.close(fig_l)
    # make an interactable plot of the relevant spectra
    fig_l, ax_l = plt.subplots()
    fig_l.subplots_adjust(bottom=0.2)
    i = 0
    for spectrum in spectra:
        ax_l.plot(spectrum[0], spectrum[1], label = labels[i])
        i += 1
    ax_l.legend()
    ax_l.set_xlabel("Photon Energy [eV]")
    leftmost = np.ceil(min(spectra[0][0]))
    rightmost = np.floor(max(spectra[0][0]))
    major_ticks = np.arange(leftmost, rightmost, 15)
    minor_ticks = np.arange(leftmost, rightmost, 5)
    ax_l.set_xticks(major_ticks)
    ax_l.set_xticks(minor_ticks, minor=True)
    ax_l.grid(which='minor', alpha=0.3)
    ax_l.grid(which='major', alpha=0.8)
    axbox = fig_l.add_axes([0.1, 0.05, 0.8, 0.075])
    text_box = TextBox(axbox, r"$E_{min}, E_{max} = $", textalignment="center")
    text_box.on_submit(submit)
    if isolated_plot: # requires an extra input from user
        fig_l.show()
        input("Press enter when done: ")
    elif not isolated_plot: # doesnt require the extra input, but will open all current plots. Only use if no other plots yet made.
        plt.show()
    # now align the spectra within the new limits, while renormalizing them to the same level
    print("Limits = ", limits)
    return limits

# this function loads either the set parameters dict or the E_offset
def import_event_parameters (get_E_offset = False, get_spectrometer_parameters = False):
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    # now import the set parameters for the events
    setfilepath = os.path.join(scriptdir, "Parameters", "set.json")
    with open(setfilepath, 'r') as f:
        setPara = json.load(f)
    if get_E_offset: 
        with open(os.path.join(scriptdir, "Parameters", "E_offset.json"), 'r') as f:
            E_offset = json.load(f)
        return E_offset
    if get_spectrometer_parameters: 
        with open(os.path.join(scriptdir, "Parameters", "specPara.json"), 'r') as f:
            specPara = json.load(f)
        return specPara
    return setPara

# general function to align two plots by using a temporary plot. x_test_range is how far from each data point is tested for matches
# data1 will be unchanged, while data2 is shifted. If isolated_plot=False, then all figures will be closed after use
def plot_alignment(data1, data2, x_test_range = 15, isolated_plot = True):
    # save the original signals for output later
    data1_output = deepcopy(data1[1]); data2_output = deepcopy(data2[1])
    # first normalize the signals so that Signal_max = 1 while signal > 0
    normalize = lambda signal: signal/signal.max()
    data1[1] = normalize(data1[1]); data2[1] = normalize(data2[1])
    # then line up the plots by finding the configuration with the smallest difference between plots.
    # hold second plot still, then shift the first until it lines up the best, given by the smallest sum of differences, normalized to num of data points
    def find_optimal_shift(dataset1, dataset2):
        shifts = np.linspace(-x_test_range/2, x_test_range/2, x_test_range*4) # test in a given range
        sums = [] # will have each sum for each shift
        for shift in shifts:
            summe = 0; N = 0 # summe is sum of absolute diff of dataset data points, while N is number of data points in sum
            test_dataset1 = np.vstack((dataset1[0] + shift, dataset1[1]))
            for i in range(len(dataset1[0])):
                idx, dist = find_nearest_index(dataset2[0], test_dataset1[0][i], distance=True)
                if dist < test_dataset1[0].sum()/len(test_dataset1[0]): # if the dist becomes larger than average x point difference in dataset, then dont use
                    N += 1 # add to num of data points used
                    summe += np.abs(test_dataset1[1][i] - dataset2[1][idx]) # contribute to sum of differences
            sums.append(summe/N) # save normalized sum
        return shifts[np.array(sums).argmin()] # return the shift corresponding to the minimum of sum of normalized signal diff
    # now find limits to align over. Note that an extra enter input will be needed from the user
    limits = select_limits_with_plot(*[data1, data2], labels = ["First dataset", "Second dataset"], isolated_plot = isolated_plot)
    # if want to manually set limits
    #limits = [1555.0, 1561.0]
    #limits = [1606.4, 1610.3]
    # now align the plots within the new limits, then renormalize them to the same level
    test_data1 = data1[:, (data1[0] > limits[0]) & (data1[0] < limits[1])]
    test_data2 = data2[:, (data2[0] > limits[0]) & (data2[0] < limits[1])]
    test_data2[1] = normalize(test_data2[1]); test_data1[1] = normalize(test_data1[1])
    # find the best shift of x values, then apply to plot 2
    data2[0] += find_optimal_shift(test_data2, test_data1)
    # recover the original signals
    data1[1] = data1_output; data2[1] = data2_output
    return data1, data2
