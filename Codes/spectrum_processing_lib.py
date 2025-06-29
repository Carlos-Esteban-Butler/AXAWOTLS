# this is a library holding the functions used to extract and process spectra during produce_spectra

import cv2
import os
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt
import json
import matplotlib.ticker as ticker
from copy import deepcopy
from ast import literal_eval

import post_processing_lib as ppl
import general_lib as gl
import data_analysis_lib as dal

# pull the spectrometer parameters from json
specPara = gl.import_event_parameters(get_spectrometer_parameters=True)

# function to pass on parameters from main.py to library
do_crystal_calibration = choose_calibration_event = new_data = DoBinning = numPixelPerBin =  0
def pass_parameters(post_process_p, *spectrum_processing_p):
    global do_crystal_calibration, choose_calibration_event, new_data, DoBinning, numPixelPerBin # make them globally accessible
    for parameter in spectrum_processing_p: # set the int values to bool
        if parameter == 0: parameter = False
        elif parameter == 1: parameter = True
    do_crystal_calibration, choose_calibration_event, new_data, DoBinning, numPixelPerBin = spectrum_processing_p # set the parameters
    ppl.pass_parameters(post_process_p) # pass the parameters onto post_process_lib
    return

# =====================================================================================================================================
# define image processing and box choosing, as well as making useable raw data out of it

# make function to save or read information about boxes from TIFF images used in spectrum extraction. Must choose either save or read
# save is chosen by inputting a label and the box parameters. Default would mean box covers almost whole image
# read will read the corresponding label and return the box parameters
def control_box_data (eventNum, specName, save=["label", [[0, 2048],[0, 1000]]], read="label"):
    if save[0] == "label" and read == "label": # check to make sure the function is being used correctly
        print("Error: Choose either save or read in kwargs of spl.control_box_data().")
        sys.exit()
    datadir = gl.make_datadir(specName, str(eventNum)) # retrieve data directory for event and spectrometer
    boxpath = os.path.join(datadir, 'box_info.txt') # path to box_info.txt
    if not os.path.isfile(boxpath): # create the file if it doesnt exist
        with open(boxpath, 'w') as f:
            f.write("Label corresponds to csv files. Box format is [y1, y2],[x1, x2] \n") # functions as a header line
    if save[0] != "label": # if saving the file
        # pass on the box parameters to the text file, creating lines if neccessary
        # read the existing lines
        with open(boxpath, 'r') as f:
            lines = f.readlines()       
        # replace the information in the corresponding line, creating a line if it doesn't exist
        replaced = 0 # variable to check if a line with the correct label was found
        for i in range(len(lines)):
            if i == 0: continue # don't consider the first line, which is just a header
            line = lines[i]
            label = line[:line.index(" box")] # slice off everything after the substring, giving only the label
            if save[0] == label: # if labels match, replace the line
                lines[i] = label + " box at " + repr(save[1]) + "\n"
                replaced += 1
        if not replaced: # if the line doesn't already exist, create it
            lines.append(save[0] + " box at " + repr(save[1]) + "\n")
        # save box info as text file with new lines
        with open(boxpath, 'w') as f:
            f.writelines(lines)
        return
    elif read != "label": # read in the data
        # read the existing lines
        with open(boxpath, 'r') as f:
            lines = f.readlines()
        # search for the information. If not found, return an error
        found = 0 # to check if a line with the correct label was found
        for i in range(len(lines)):
            if i == 0: continue # don't consider the first line, which is just a header
            line = lines[i]
            label = line[:line.index(" box")] # slice off everything after the substring, giving only the label
            if read == label: # if labels match, read in the box parameters
                box_parameters = literal_eval(line.replace(label + " box at ", "")) # pull the list as str and return it to list type
                found += 1
        if not found: # if the line doesn't already exist, throw an error
            print("Error: the box parameters for "+read+" have not yet been saved.")
            sys.exit()
        return box_parameters

# this function serves as a subfunction of select_box(). adjusts the contrast of the image to make it more visible
def adjust_contrast(image, factor):
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the grayscale image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    # Adjust the contrast by multiplying the enhanced image with the contrast factor
    adjusted_image = np.clip(factor * enhanced_image, 0, 255).astype(np.uint8)
    return adjusted_image

# uses cv2 to show and select the desired box, then saves the dimensions to use for pulling data with Pillow later. Serves as subfunction of fullImageProcessing()
# only_choose_vertical means to only choose the y coordinates, i.e. will make a full length box. If off, then can choose all dimensions of rectangle
def select_box(image, only_choose_vertical = True):
    clone = image.copy()
    roi = []
    cropping = False
    boxCreated = False
    def click_and_crop(event, x, y, flags, param):
        nonlocal roi, cropping
        if event == cv2.EVENT_LBUTTONDOWN:
            roi = [(x, y)]
            cropping = True

        elif event == cv2.EVENT_LBUTTONUP:
            roi.append((x, y))
            cropping = False
            # if only need the horizontal lineout, then automatically use the entire x-range
            if only_choose_vertical: cv2.rectangle(clone, (0, roi[0][1]), (image.shape[1]-1, roi[1][1]), (255, 0, 0), 1)
            else: cv2.rectangle(clone, (roi[0][0], roi[0][1]), (roi[1][0], roi[1][1]), (255, 0, 0), 1)
            cv2.imshow("image", clone)
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    runCycle = True
    while runCycle:
        cv2.imshow("image", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"): # key to save the drawn box
            if len(roi) == 2:
                (x1, y1), (x2, y2) = roi[0], roi[1]
                # if only want the vertical coordinated, then overwrite x1 and x2
                if only_choose_vertical: x1 = 0; x2 = image.shape[1]-1
                box = [[y1, y2], [x1,x2]]
                # sort the y and x values to avoid error if the user draws a box starting from the bottom or right
                box[0] = sorted(box[0]); box[1] = sorted(box[1])
                print ("Box saved at ", box)
                boxCreated = True
        elif key == ord("e"): # exit the opened image
            if (boxCreated):
                print("Box successfully created.")
                break
            else:
                print("Create a box.")
        elif key == ord("r"): # reset the drawn box graphic
            if len(roi) == 2:
                clone = image.copy()
                roi = []
                print("Box reset.")
        elif key == ord("c"): # adjust contrast up
            image = adjust_contrast(image, 1.25)
            clone = image.copy()
        elif key == ord("v"): # adjust contrast down
            image = adjust_contrast(image, 0.75)
            clone = image.copy()
    cv2.destroyAllWindows()
    return box

# define function to do full image processing. type kwarg decides what type of lineout to do. only_choose_vertical only serves to pass to select_box()
# type choices are: "average" (creates horizontal line out), "sum" (sum over y values), "vertical" (vertical line out), "std" (std deviation of all values)
# manual box allows for passing box parameters previously created
def fullImageProcessing(spect, currentEventNum, type = "average", only_choose_vertical = True, manual_box = []):
    if(float(currentEventNum) < 10): # grab the directory of the current event
        extraZeros = "000"
    else: extraZeros = "00"
    currenteventdir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "Events", extraZeros + str(currentEventNum))
    # Read the TIFF image
    TIFFpath = os.path.join(currenteventdir, spect, spect + ".tif")
    image = cv2.imread(TIFFpath, cv2.IMREAD_GRAYSCALE)

    # function to make a data array out of raw arrays.
    def makeData (image, box, is_background=False):
        # create arrays out of images of selected areas
        rawArr = np.array(image)[box[0][0]:box[0][1], box[1][0]:box[1][1]]
        # right now the y axis of the array holds the x pixels, while the x axis holds the y pixels. Transpose to fix this
        np.transpose(rawArr)
        # now find the desired values. average yields a horizontal lineout, sum sums over the y directions for each x
        # and vertical gives a vertical lineout. std gives the std deviation over y values
        finalValues = np.array([])
        if type == "average" or is_background: finalValues = np.mean(rawArr, axis = 0)
        # sum over the y direction if choose to or if the spectrometer is focusing
        if (type == "sum" or specPara[spect]["crystal bend"]=="spherical") and not is_background: finalValues = np.sum(rawArr, axis=0)
        if type == "vertical": finalValues = np.mean(rawArr, axis=1)
        if type == "std": finalValues = np.std(rawArr, axis=0)
        if not finalValues.all(): print("Error: type for spl.makeData() doesn't exist.") ;sys.exit()
        # Finally get array of pixels with corresponding intensity
        Arr = np.vstack((np.arange(len(finalValues)), finalValues))
        return Arr

    # Call the function to select boxes for spectra, giving directions. Then save the box info to a txt file
    # if a manual box hasnt been given, select the boxes then save the info
    if manual_box == []:
        boxS = select_box(image, only_choose_vertical=only_choose_vertical)
        if only_choose_vertical: 
            boxB = select_box(image, only_choose_vertical=only_choose_vertical) # if statement, as knife edge calc doesnt need background
            control_box_data(currentEventNum, spect, save=["spec", boxS]) # in the if statement, so that knife edge doesn't overwrite the spec box data
            control_box_data(currentEventNum, spect, save=["back", boxB])
            # now pull for extra channels if they exist
            if (specPara[spect]["channels"] == "double"): 
                boxS1 = select_box(image, only_choose_vertical=only_choose_vertical)
                control_box_data(currentEventNum, spect, save=["spec_1", boxS1])
            else:
                boxS1 = boxS # essentially a placeholder in case of a single channel
        # if doing source size determination, save the box data accordingly
        elif not only_choose_vertical: control_box_data(currentEventNum, spect, save=["source_size", boxS])
    else: 
        boxS = manual_box

    # Open with Pillow, since it should pull the correct intensity values. For some reason this is not the case for cv2
    im = Image.open(TIFFpath)
    spec = makeData(im, boxS) # spec here means spectrum, while spect is the spectrometer name (sorry for the varied conventions)
    if not only_choose_vertical or manual_box != []: return spec # since dont need background for source size broadening/only need background for stat error calc
    back = makeData(im, boxB, is_background=True) # need to make sure that all boxes have about same height
    spec1 = makeData(im, boxS1)
    return spec, back, spec1

# define function to combine all the steps, returning the raw data as np arrays. 
# temp is if you want to pull temporary data, i.e. the data isn't saved as a csv
# type kwarg is carry over for fullImageProcessing
def pullAndSaveRawData (spectNum, spectName, dataDict, currentEventNum, temp = False, type = "average"):
    dataDict[spectNum]["name"] = spectName # save the actual spectrometer name in data dictionary
    # find datadir
    datadir = gl.make_datadir(spectName, str(currentEventNum))
    # then check if csv of raw data already exists, if it does extract it. Also can overwrite if desired
    cont = "y" # initalize yes so that if csvs dont exist, automatically make them.
    while (True): # while loop so that errors can be fixed
        # prompt the user if parameter new_data from main.py is true and is not temporary
        if (os.path.isfile(os.path.join(datadir, "back_raw.csv")) and new_data and not temp): # check using background csv so works for all types of spectrometer
            cont = input("Raw data for {} previously extracted. Make new data (y or n)?: ".format(spectName)) # give chance to say no
        if (cont == "n" or not new_data and not temp):
            dataDict[spectNum]["spec"] = gl.extract_csv(os.path.join(datadir, "spec_raw.csv"))
            dataDict[spectNum]["back"] = gl.extract_csv(os.path.join(datadir, "back_raw.csv"))
            if (specPara[spectName]["channels"] == "double"):
                dataDict[spectNum]["spec1"] = gl.extract_csv(os.path.join(datadir, "spec_1_raw.csv"))
            return dataDict
        elif (cont == "y" or temp): # do this if temp data is chosen or the user selects yes
            print("Choose boxes for " + spectName)
            spec, back, spec1 = fullImageProcessing(spectName, currentEventNum, type=type)
            # so far have needed to flip the spectra. CHANGE THIS IF PROBLEMS WITH SPECTRUM ORIENTATION
            spec[1] = np.flipud(spec[1])
            back[1] = np.flipud(back[1])
            # load the final spectra into the data dictionary
            dataDict[spectNum]["spec"] = spec
            dataDict[spectNum]["back"] = back
            if (specPara[spectName]["channels"] == "double"): # only save the spec1 if exists, otherwise spec1 is just a dummy variable
                dataDict[spectNum]["spec1"] = spec1
            if os.path.isdir(datadir) is False: # make data directory is doesnt exist
                os.mkdir(datadir)
            if not temp: # dont save the data if temporary
                np.savetxt(os.path.join(datadir, 'spec_raw.csv'), spec, fmt="%1.3f", delimiter=",") # save the csvs
                np.savetxt(os.path.join(datadir, 'back_raw.csv'), back, fmt="%1.3f", delimiter=",")
                if (specPara[spectName]["channels"] == "double"): # only save the spec1 if exists.
                    np.savetxt(os.path.join(datadir,'spec_1_raw.csv'), spec1, fmt="%1.3f", delimiter=",")
            return dataDict
        else:
            print("Error: only use y or n")

# data processing section ============================================================================================================================

# define function to check if energy offset dictionary is already filled with the event info and save. if not initialize the needed keys
def initialize_offset (currentEventNum, spectName, offset_dict):
    if currentEventNum not in offset_dict:
        offset_dict[currentEventNum] = {}
    if spectName not in offset_dict[currentEventNum]:
        offset_dict[currentEventNum][spectName] = {}
    if "E_offset" not in offset_dict[currentEventNum][spectName]:
        offset_dict[currentEventNum][spectName]["E_offset"] = 0 # if theres no E_offset saved by user, then initialize and use 0 [eV]
        if (specPara[spectName]["channels"] == "double"):
            offset_dict[currentEventNum][spectName]["E_offset_1"] = 0 # similar convention to spec and spec1 in pullAndSaveData()
    # save the offset for future use.
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "Parameters", "E_offset.json"), 'w') as f: 
        json.dump(offset_dict, f)
    return offset_dict

# function to carry out first processing given a raw spectrum (count over pixel to background corrected count over photon energy)
def post_process(rawData, sp, eventNum, offset, dispNum, specName, extra=False):
    specKey = "spec"; E_offsetKey = "E_offset" 
    # build in case of looking at second spectrum of double channel spectrometer
    if extra and specPara[specName]["channels"] == "double": 
        specKey = "spec1"; E_offsetKey = "E_offset_1"
    # now do data processing.
    data = ppl.correct_background(rawData[sp][specKey].astype(float), rawData[sp]["back"].astype(float), specName, eventNum) # correct the background out
    data = ppl.calc_E(data, eventNum, specName, dispNum, float(offset[eventNum][specName][E_offsetKey]), extra) # convert px to E using dispersion
    return data

# routine to make the spectra. Outputs results as a dict, which should append onto a complete dict in main code
# Most important function, since here all the post processing steps are called
def make_spectrum(rawData, sp, eventNum, offset, dispNum, setPara):
    specName = rawData[sp]["name"]
    extra = False # initialize extra boolean for use in double channel spectrometers
    # initialize a dictionary to contain the processed spectra
    spectraData = {}
    # process the raw data
    offset = initialize_offset(eventNum, specName, offset) # initialize offset dict in case no current key for spectrometer
    datadir = gl.make_datadir(specName, eventNum) # find data directory path, specFolderConv is dummy variable here
    spectraData["spec"] = post_process(rawData, sp, eventNum, offset, dispNum, specName, extra = extra)
    np.savetxt(os.path.join(datadir, 'spec.csv'), spectraData["spec"], fmt="%1.3f", delimiter=",") # save results
    if (specPara[specName]["channels"] == "double"): # need to repeat for extra spectra of double channel
        spectraData["spec1"] = post_process(rawData, sp, eventNum, offset, dispNum, specName, extra=True)
        np.savetxt(os.path.join(datadir, 'spec_1.csv'), spectraData["spec1"], fmt="%1.3f", delimiter=",") # save the processed data
    return spectraData


# function to calculate the abs coefficient using calibration shots and save the results. Also plots
def absorption_coefficient(spectra, calSpectra, transSpecName, sourceSpecName, aluThickness, eventNum, calEventNum, setPara, isolated_plot = False, final_result = False, ylabel="Counts [-]"):
    # define new variables to make code more readable
    if (specPara[transSpecName]["channels"] == "double"): # if dealing with double channel spectrometer
        trans = spectra[transSpecName]["spec"]
        source = spectra[transSpecName]["spec1"]
        calTrans = calSpectra[transSpecName]["spec"]
        calSource = calSpectra[transSpecName]["spec1"]
    else: # for all other cases
        trans = spectra[transSpecName]["spec"]
        source = spectra[sourceSpecName]["spec"]
        calTrans = calSpectra[transSpecName]["spec"]
        calSource = calSpectra[sourceSpecName]["spec"]
    # if the backlighter targets are the same for both events, line up all the spectra automatically by matching the features. If isolated_plot = False, then will do plt.show(), closing all figures after
    if setPara[calEventNum]["target"] == setPara[eventNum]["target"]: trans, source, calTrans, calSource = ppl.line_up_spectra(trans, source, calTrans, calSource, isolated_plot = isolated_plot)
    # align the spectra data points with one another
    trans_common, source_common = ppl.match_energies(trans, source)
    calTrans_common, calSource_common = ppl.match_energies(calTrans, calSource)
    # divide the trans by the source to get the transmission coefficient for each energy and event, then take the ab coefficient from that.
    TCurrent = trans_common[1,:]/source_common[1,:]
    TCal = calTrans_common[1,:]/calSource_common[1,:]
    TCurrent_common, TCal_common = ppl.match_energies(np.vstack((trans_common[0],TCurrent)), np.vstack((calTrans_common[0],TCal))) # line up energies of cal and current shot transmissions
    T = TCurrent_common[1,:] / TCal_common[1,:]
    T[T < 0] = 0.1 # to avoid error in log with A
    # calculate the absorption, correcting the thickness of the aluminum for the angle
    twod = float(specPara[transSpecName]["lattice spacing"]) # pull the lattice spacing in angstrom
    n_order = float(specPara[transSpecName]["order"]) # pull the order of refraction
    twod = twod/n_order
    deff = aluThickness/np.cos(np.arcsin(12398/(TCurrent_common[0,:]*twod)))
    Aresults = -np.log(T)/deff; Aresults[Aresults<0] = 0; Aresults[Aresults>5] = 5 # if absorption coeff lower than 0 or higher than 5, forcibly adjust it
    A = np.vstack((TCurrent_common[0,:], Aresults)) # add on the energies in first row to get the final np.array
    # now account for uncertainty of the Al thickness
    d_deff_rel = 0.10 # 10% uncertainty
    dA = A[1].max()*d_deff_rel # mathematically works out like this using gaussian propogation. Take max A to make sure to overestimate, rather than under
    # save the results as a csv
    datadir = gl.make_datadir(transSpecName, eventNum) # find data directory path
    np.savetxt(os.path.join(datadir, 'abs.csv'), A, fmt='%1.3f', delimiter=",") 
    # also save the spectra that give the results as csvs in separate folder
    abs_component_spectra_dir = os.path.join(datadir, "Abs Component Spectra")
    if os.path.isdir(abs_component_spectra_dir) is False: # make extra directory for spectra used for the absorption spectrum if doesn't exist
        os.mkdir(abs_component_spectra_dir)
    np.savetxt(os.path.join(abs_component_spectra_dir, 'trans.csv'), trans_common, fmt='%1.3f', delimiter=",")
    np.savetxt(os.path.join(abs_component_spectra_dir, 'source.csv'), source_common, fmt='%1.3f', delimiter=",")
    np.savetxt(os.path.join(abs_component_spectra_dir, 'cal_trans.csv'), calTrans_common, fmt='%1.3f', delimiter=",")
    np.savetxt(os.path.join(abs_component_spectra_dir, 'cal_source.csv'), calSource_common, fmt='%1.3f', delimiter=",")
    # grab literature data from levy 2010
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(scriptdir, "Graphs", "literature_a_spectrum", "Levy2010.csv")
    literature_abs = gl.extract_csv(filepath, abs_lit=True)
    # finally do plotting
    if not final_result: # will show in initial data extraction step, not in data analysis
        # plot the results
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True)
        fig.supxlabel("Photon Energy [eV]")
        # first graph the uncut background corrected spectra
        ax1.plot(source[0, :], source[1, :]/1000, lw=0.7, color = "r", label = "Source")
        ax1.plot(trans[0, :], trans[1, :]/1000, lw=0.7, color = "b", label = "Transmission")
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(15))
        ax1.set_ylabel("Kilocount [-]")
        ax1.set_title("Event " + eventNum + ": Shot using " + transSpecName + " on " + setPara[eventNum]["target"] + " at " + str(setPara[eventNum]["energy"]) + "J")
        # control output of grid
        leftmost = np.ceil(min(source[0, :]))
        rightmost = np.floor(max(source[0, :]))
        major_ticks = np.arange(leftmost, rightmost, 15)
        minor_ticks = np.arange(leftmost, rightmost, 5)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.8)
        ax1.set_xlim(left=leftmost, right=rightmost)
        ax1.legend()
        # now graph the cut spectra. x axis needs to be shortened as well
        ax2.plot(source_common[0, :], source_common[1, :]/1000, lw=0.7, color = "r", label = "Source")
        ax2.plot(trans_common[0, :], trans_common[1, :]/1000, lw=0.7, color = "b", label = "Transmission")
        # ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax2.set_ylabel("Kilocount [-]")
        ax2.set_title("Processed Spectra")
        # control output of grid
        leftmost = np.ceil(min(source_common[0, :]))
        rightmost = np.floor(max(source_common[0, :]))
        major_ticks = np.arange(leftmost, rightmost, 15)
        minor_ticks = np.arange(leftmost, rightmost, 5)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', alpha=0.35)
        ax2.grid(which='major', alpha=0.8)
        #ax2.set_ylim(bottom=min(Trans_common[1,:]))
        ax2.set_xlim(left=leftmost, right=rightmost)
        ax2.legend()
        # Plot the absorption coefficient
        ax3.plot(A[0,:], A[1,:], lw=0.7, label="Absorption")
        ax3.set_ylabel("Absorption Coefficient [μm$^{-1}$]")
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='minor', alpha=0.3)
        ax3.grid(which='major', alpha=0.8)
        ax3.set_title("Absorption through Aluminum Foil")
        ax3.plot(literature_abs[0], literature_abs[1], lw=2, alpha=0.4, color="purple", label="Literature (cold Al)")
        ax3.errorbar(A[0].min()+5, A[1].max(), capsize=4, ecolor="r", yerr=dA, alpha=0.7)
        ax3.text(A[0].min()+7, A[1].max()+dA-0.1, "Al Thickness Error", color="r", alpha=0.7)
        ax3.legend()
        # finally do the calibration spectra
        ax4.plot(calSource_common[0, :], calSource_common[1, :]/1000, lw=0.7, color = "r", label = "Source")
        ax4.plot(calTrans_common[0, :], calTrans_common[1, :]/1000, lw=0.7, color = "b", label = "Transmission")
        ax4.set_title("Processed Calibration Spectra from event " + str(calEventNum)+ " on " + setPara[eventNum]["target"])
        ax4.set_ylabel("Kilocount [-]")
        ax4.legend()
        ax4.grid(which='minor', alpha=0.3)
        ax4.grid(which='major', alpha=0.8)
        fig.set_size_inches(9.5, 11.5)
    elif final_result:
        # plot the results
        fig, (ax2, ax4, ax3) = plt.subplots(3,1, sharex=True)
        # now graph the cut spectra. x axis needs to be shortened as well
        ax2.plot(source_common[0, :], source_common[1, :], lw=2, color = "r", label = "Source from "+sourceSpecName)
        ax2.plot(trans_common[0, :], trans_common[1, :], lw=2, color = "b", label = "Transmitted from "+transSpecName)
        # ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax2.set_ylabel(ylabel)
        ax2.set_title("Event " + eventNum + ": Processed Spectra from "+str(setPara[eventNum]["energy"])+"J Shot on " + setPara[eventNum]["target"])
        # control output of grid
        leftmost = np.ceil(min(source_common[0, :]))
        rightmost = np.floor(max(source_common[0, :]))
        major_ticks = np.arange(leftmost, rightmost, 15)
        minor_ticks = np.arange(leftmost, rightmost, 5)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', alpha=0.35)
        ax2.grid(which='major', alpha=0.8)
        #ax2.set_ylim(bottom=min(Trans_common[1,:]))
        ax2.set_xlim(left=leftmost, right=rightmost)
        ax2.legend()
        # Plot the absorption coefficient
        ax3.plot(A[0,:], A[1,:], lw=2, label="Experiment")
        ax3.plot(literature_abs[0], literature_abs[1], lw=2, alpha=0.4, color="purple", label="Literature (cold Al)")
        ax3.errorbar(A[0].min()+5, A[1].max(), capsize=4, ecolor="r", yerr=dA, alpha=0.7)
        ax3.text(A[0].min()+7, A[1].max()+dA-0.1, "Al Thickness Error", color="r", alpha=0.7)
        ax3.set_ylabel("Absorption Coefficient [μm$^{-1}$]")
        ax3.set_xlabel("Photon Energy [eV]")
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='minor', alpha=0.3)
        ax3.grid(which='major', alpha=0.8)
        ax3.set_title("Absorption through Aluminum Foil of Effective Thickness "+str(np.round(np.mean(deff),2)) + " micron")
        ax3.legend()
        # finally do the calibration spectra
        ax4.plot(calSource_common[0, :], calSource_common[1, :], lw=2, color = "r", label = "Source from "+sourceSpecName+" (calibration)")
        ax4.plot(calTrans_common[0, :], calTrans_common[1, :], lw=2, color = "b", label = "Source from "+transSpecName+" (calibration)")
        ax4.set_title("Calibration Event " + calEventNum + ": Processed Spectra from "+str(setPara[calEventNum]["energy"])+"J Shot on " + setPara[calEventNum]["target"])        
        ax4.set_ylabel(ylabel)
        ax4.legend()
        ax4.grid(which='minor', alpha=0.3)
        ax4.grid(which='major', alpha=0.8)
        fig.set_size_inches(10, 10)
        plt.savefig(os.path.join("Graphs","absorption","absorption_spectrum_of_"+setPara[eventNum]["target"]+"_event_"+eventNum+"_on_"+transSpecName+".pdf"))
    return

# function to save or recall calibration event number
def get_calibration_event_num (eventNum, transSpecName):
    setPara = gl.import_event_parameters() # retrieve general event parameters
    datadir = gl.make_datadir(transSpecName, eventNum) # retrieve data directory for event and spectrometer
    if not os.path.isfile(os.path.join(datadir, "calibration_info.txt")) or choose_calibration_event: # if currently no cal_info file or wish to choose new one
        # Have user input a calEventNum, looping until user puts in a valid event for calibration
        while True:
            print()
            calEventNum = str(input("Input event to use for calibration: ")) # input event for calibration
            if (list(setPara[str(calEventNum)]["spect"].keys()) != list(setPara[str(eventNum)]["spect"].keys())):
                print("Error: calibration setup must be the same as the current setup.")
            else:
                break
        # save calibration info as text file
        with open(os.path.join(datadir, 'calibration_info.txt'), 'w') as f:
            f.write("Used event number " + str(calEventNum) + " for calibration.")
    else: # pull cal event number from txt file
        with open(os.path.join(datadir, 'calibration_info.txt'), 'r') as f:
            line = f.readline().strip('\n')
            calEventNum = [str(s) for s in line.split() if s.isdigit()][0]
    return calEventNum

# function to grab the calibration data. returns it as a dictionary containing the calibration spectra of both event spectrometers
def get_calibration_data (calEventNum, transSpecName, sourceSpecName):
    setPara = gl.import_event_parameters() # retrieve general event parameters
    offset = gl.import_event_parameters(get_E_offset = True) # retrieve E_offset dict
    print("\nNow pulling calibration data: ")
    rawCalData= {"spect1": {}, "spect2": {}} # create dict to store calibration data
    calSpectra = {} # dict to store calibration spectra
    if(specPara[transSpecName]["channels"] == "single"): # if not double channel, since double channel works by itself
        rawCalData = pullAndSaveRawData("spect1", sourceSpecName, rawCalData, calEventNum) # get source raw data
        calSpectra[sourceSpecName] = make_spectrum(rawCalData, "spect1", calEventNum, offset, int(setPara[calEventNum]["spect"][sourceSpecName]["disp"]), setPara)
        print() # for space in between pulls
    rawCalData = pullAndSaveRawData("spect2", transSpecName, rawCalData, calEventNum)
    calSpectra[transSpecName] = make_spectrum(rawCalData, "spect2", calEventNum, offset, int(setPara[calEventNum]["spect"][transSpecName]["disp"]), setPara)
    return calSpectra

# function to calculate absorption spectra, but without crystal calibration shots. return_A=True will return the absorption array, else return nothing
# ylabel controls what y-axis label is used for spectra graphs. defaults to Counts bc that will be used for initial data gathering step.
def absorption_coefficient_wo_calibration(spectra, transSpecName, sourceSpecName, aluThickness, eventNum, return_A=False, plotting = True, final_result = False, ylabel="Counts [-]"):
    setPara = gl.import_event_parameters() # retrieve general event parameters
    if (specPara[transSpecName]["channels"] == "double"): # if dealing with double channel
        trans = spectra[transSpecName]["spec"]
        source = spectra[transSpecName]["spec1"]
    else: # for all other cases
        trans = spectra[transSpecName]["spec"]
        source = spectra[sourceSpecName]["spec"]

    # bring the edge approximately into the right place.
    # theres probably a better way of doing this, but I don't have time at the moment for optimization
    deriv_test_trans = deepcopy(trans)
    temp_num_pix_per_bin = 32
    if DoBinning: temp_num_pix_per_bin -= numPixelPerBin # so that dont overbin
    deriv_test_trans = dal.bin_data(deriv_test_trans, force=True, manual_num_pix_per_bin = temp_num_pix_per_bin) # heavily bin to make sure is pulling from the edge and not a random dip
    deriv_test_trans = deriv_test_trans[:, np.abs(deriv_test_trans[0] - 1558.98) < 10] # for finding the edge using derivative, within 20eV of theoretical edge
    # edge found by finding the location of largest derivative, then identifying the corresponding energy, then getting index for that E in original array
    idx_most_neg_derivative = gl.find_nearest_index(trans[0], deriv_test_trans[0][np.diff(deriv_test_trans[1]).argmin()])
    trans[0] -= trans[0][idx_most_neg_derivative] - 1558.98 # bring the edge approximately into the right place
    
    # align the two spectra with each other through manually choosing
    print("Choose an E range for alignment of plots. ")
    trans, source = gl.plot_alignment(trans, source) # shifts source spectra to align with transmission
    # iron out the differences in E values, creating a common E_range while cutting down the larger E range spectrum
    trans_common, source_common = ppl.match_energies(trans, source)
    # divide the trans by the source to get the transmission, then take the ab coefficient from that.
    T = trans_common[1]/source_common[1]
    T[T < 0] = 0.1 # to avoid error in log with A
    # calculate the absorption, correcting the thickness of the aluminum for the angle
    twod = float(specPara[transSpecName]["lattice spacing"]) # pull the lattice spacing in angstrom
    n_order = float(specPara[transSpecName]["order"]) # pull the order of refraction
    twod = twod/n_order
    deff = aluThickness/np.cos(np.arcsin(12398/(source_common[0]*twod))) # correct for angle
    Aresults = -np.log(T)/deff; Aresults[Aresults<0] = 0; Aresults[Aresults>5] = 5 # if absorption coeff lower than 0 or higher than 5, forcibly adjust it
    A = np.vstack((source_common[0], Aresults)) # add on the energies in first row to get the final np.array
    # now account for uncertainty of the Al thickness
    d_deff_rel = 0.1 # 10% uncertainty
    dA = A[1].max()*d_deff_rel # mathematically works out like this using gaussian propogation. Take max A to make sure to overestimate, rather than under
    # save the results as a csv
    datadir = gl.make_datadir(transSpecName, eventNum) # find data directory path
    np.savetxt(os.path.join(datadir, 'abs_wo_calibration.csv'), A, fmt='%1.3f', delimiter=",") 
    # also save the spectra that give the results as csvs in separate folder
    abs_component_spectra_dir = os.path.join(datadir, "Abs Component Spectra No Calibration")
    if os.path.isdir(abs_component_spectra_dir) is False: # make extra directory for spectra used for the absorption spectrum if doesn't exist
        os.mkdir(abs_component_spectra_dir)
    np.savetxt(os.path.join(abs_component_spectra_dir, 'trans.csv'), trans_common, fmt='%1.3f', delimiter=",")
    np.savetxt(os.path.join(abs_component_spectra_dir, 'source.csv'), source_common, fmt='%1.3f', delimiter=",")
    # grab literature data from levy 2010
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(scriptdir, "Graphs", "literature_a_spectrum", "Levy2010.csv")
    literature_abs = gl.extract_csv(filepath, abs_lit=True)
    # finally plot
    if plotting and not final_result: # for initial data extraction, not data analysis
        # plot the results
        fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
        fig.supxlabel("Photon Energy [eV]")
        # first graph the uncut unaligned spectra
        ax1.plot(source[0], source[1]/1000, lw=0.7, color = "r", label = "Source")
        ax1.plot(trans[0], trans[1]/1000, lw=0.7, color = "b", label = "Transmission")
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(15))
        ax1.set_ylabel("Kilocount [-]")
        ax1.set_title("Event " + eventNum + ": Shot using " + transSpecName + " on " + setPara[eventNum]["target"] + " at " + str(setPara[eventNum]["energy"]) + "J")
        # control output of grid
        leftmost = np.ceil(min(source[0]))
        rightmost = np.floor(max(source[0]))
        major_ticks = np.arange(leftmost, rightmost, 15)
        minor_ticks = np.arange(leftmost, rightmost, 5)
        ax1.set_xticks(major_ticks)
        ax1.set_xticks(minor_ticks, minor=True)
        ax1.grid(which='minor', alpha=0.2)
        ax1.grid(which='major', alpha=0.8)
        ax1.set_xlim(left=leftmost, right=rightmost)
        ax1.legend()
        # now graph the cut and aligned spectra.
        ax2.plot(source_common[0], source_common[1]/1000, lw=0.7, color = "r", label = "Source")
        ax2.plot(trans_common[0], trans_common[1]/1000, lw=0.7, color = "b", label = "Transmission")
        # ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax2.set_ylabel("Kilocount [-]")
        ax2.set_title("Cut and Aligned Spectra")
        # control output of grid
        leftmost = np.ceil(min(source_common[0]))
        rightmost = np.floor(max(source_common[0]))
        major_ticks = np.arange(leftmost, rightmost, 15)
        minor_ticks = np.arange(leftmost, rightmost, 5)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', alpha=0.35)
        ax2.grid(which='major', alpha=0.8)
        #ax2.set_ylim(bottom=min(Trans_common[1,:]))
        ax2.set_xlim(left=leftmost, right=rightmost)
        ax2.legend()
        # Plot the absorption coefficient
        ax3.plot(A[0], A[1], lw=0.7, label="Absorption")
        ax3.set_ylabel("Absorption Coefficient [μm$^{-1}$]")
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='minor', alpha=0.3)
        ax3.grid(which='major', alpha=0.8)
        ax3.set_title("Absorption through Aluminum Foil of Effective Thickness "+str(np.round(np.mean(deff),2)) + " micron")
        ax3.plot(literature_abs[0], literature_abs[1], lw=2, alpha=0.4, color="purple", label="Literature (cold Al)")
        ax3.errorbar(A[0].min()+5, A[1].max(), capsize=4, ecolor="r", yerr=dA, alpha=0.7)
        ax3.text(A[0].min()+7, A[1].max()+dA-0.1, "Al Thickness Error", color="r", alpha=0.7)
        ax3.legend()
        # adjust size of the figure
        fig.set_size_inches(9.5, 11.5)
    if plotting and final_result: # for data analysis, where the plot can be saved
        # plot the results
        fig, (ax2, ax3) = plt.subplots(2,1, sharex=True)
        fig.supxlabel("Photon Energy [eV]")
        # now graph the cut and aligned spectra.
        ax2.plot(source_common[0], source_common[1], lw=2, color = "r", label = "Source")
        ax2.plot(trans_common[0], trans_common[1], lw=2, color = "b", label = "Transmitted")
        # ax2.xaxis.set_major_locator(ticker.MultipleLocator(30))
        ax2.set_ylabel(ylabel)
        ax2.set_title("Event " + eventNum + ": Processed " + transSpecName + " Spectra from "+str(setPara[eventNum]["energy"])+"J Shot on " + setPara[eventNum]["target"])
        #ax2.set_title(transSpecName + " with "+setPara[eventNum]["target"]+" Backlighter")
        # control output of grid
        leftmost = np.ceil(min(source_common[0]))
        rightmost = np.floor(max(source_common[0]))
        major_ticks = np.arange(leftmost, rightmost, 5)
        minor_ticks = np.arange(leftmost, rightmost, 1)
        ax2.set_xticks(major_ticks)
        ax2.set_xticks(minor_ticks, minor=True)
        ax2.grid(which='minor', alpha=0.35)
        ax2.grid(which='major', alpha=0.8)
        #ax2.set_ylim(bottom=min(Trans_common[1,:]))
        ax2.set_xlim(left=leftmost, right=rightmost)
        ax2.legend()
        # Plot the absorption coefficient
        ax3.plot(A[0], A[1], lw=2, label="Experiment")
        ax3.plot(literature_abs[0], literature_abs[1], lw=2, alpha=0.4, color="purple", label="Literature (cold Al)")
        ax3.errorbar(A[0].min()+5, A[1].max(), capsize=4, ecolor="r", yerr=dA, alpha=0.7)
        ax3.text(A[0].min()+7, A[1].max()+dA-0.1, "Al Thickness Error", color="r", alpha=0.7)
        ax3.set_ylabel("Absorption Coefficient [μm$^{-1}$]")
        ax3.set_xticks(major_ticks)
        ax3.set_xticks(minor_ticks, minor=True)
        ax3.grid(which='minor', alpha=0.3)
        ax3.grid(which='major', alpha=0.8)
        ax3.set_title("Absorption through Aluminum Foil of Effective Thickness "+str(np.round(np.mean(deff),2)) + " micron")
        ax3.legend()
        # adjust size of the figure
        fig.set_size_inches(9, 6)
        plt.savefig(os.path.join("Graphs","absorption_no_calibration","absorption_spectrum_of_"+setPara[eventNum]["target"]+"_event_"+eventNum+"_on_"+transSpecName+".pdf"))
    if return_A is True: return A
    return

# define function to change offset in offset.json ========================================================================================
def offset_change (specNum, rawData, offset, eventNum):
    specName = rawData[specNum]["name"]
    additionalString = ""
    if (specPara[specName]["channels"] == "double"): additionalString = " (channel 1)"
    print("Current offset of " + specName + additionalString + " is: " + str(offset[eventNum][specName]["E_offset"]) + "eV.")
    offset[eventNum][specName]["E_offset"] = input("Change offset of " + specName + additionalString +  " to: ")
    if (specPara[specName]["channels"] == "double"):
        additionalString = " (channel 2)"
        print("Current offset of " + specName + additionalString + " is: " + str(offset[eventNum][specName]["E_offset_1"]) + "eV.")
        offset[eventNum][specName]["E_offset_1"] = input("Change offset of " + specName + additionalString +  " to: ")
    return