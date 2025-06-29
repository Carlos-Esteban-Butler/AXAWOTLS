# Welcome to AXAWOTLS!
Author: Carlos Esteban Butler (M. Sc)
Contact: carlosbutler210@gmail.com


## Description

AXAWOTLS (Analysis of X-ray Absorption for WDM Observation and Testing of Locally-made Spectrometers) is a programm designed to process and analyze spectra for experiments on WDM of aluminum at GSI/FAIR using laser-plasma backlighters. Currently, the code can accommodate spectrometer designs with flat or spherical crystals, but should be easy to extend to other cases. The programm is designed to be user friendly and at the surface level should only require the user to be familiar with two codes: main.py and jsonCreator.py. In addition, the user will need to be aquainted with the format of the Events directory, where the TIFF images from the shots, along with other data are stored. The results are also saved as csv files here. 

The first three sections of this README contain the information neccessary to use the programm for an average user. The rest elaborates on more detailed areas of the code and can be used as a reference. Note that .\ always refers to the directory that holds this README and the Events and Codes directories.

If you want to run tests and play around with cases known to work, I have provided 7 events, whose properties can be found in .\Codes\Parameters\jsonCreator.py. When starting a new experiment I strongly advise making an independent copy/branch of AXAWOTLS and deleting all the test events. Make sure to never unintentionally change the base version of AXAWOTLS found here!

## The Events directory and saving data

When conducting an experiment, all raw data is to be saved in the Events directory (a.k.a. dir), conforming to the following format: 
1. Create an event by making an empty dir in the Events dir with a 4 digit designation, i.e. 0001 or 0012. This is known as the "event number".
2. Create an empty directory in the event number dir with the name of a spectrometer. Repeat this for all spectrometers of the event, i.e. 0001\DUCC\ or 0012\SUCC\
3. In the spectrometer dir, save the TIFF file captured by the spectrometer for a shot, using the spectrometer's name, i.e. 0001\DUCC\DUCC.tif or 0012\SUCC\SUCC.tif

This will result in the file structure for a TIFF file of .\Events\0001\DUCC\DUCC.tif, which means the TIFF of the data captured by the DUCC for event number 1. To note is that later on, AXAWOTLS will automatically create a dir in the spectrometer dir to hold the results as csv files and supplementary information. This does not need to be done manually.

And with that you are ready to use the main codes!

## Essential codes and getting started

For the rest of this guide, everything is found in the Codes dir, i.e. .\Codes. AXAWOTLS consist of one main code titled main.py, from which all functions can be controlled, as well as auxillary libraries containing all the functions and methods. I will elaborate on each library later. In addition there is an event_finder.py, with which you can find the events that contain specific parameters, e.g. a shot on aluminum using specific spectrometers within a given laser energy range. Finally, there are a series of subdirectories within .\Codes, each containing specialized data or information.

One key aspect of the spectrum processing that is unique to AXAWOTLS is what I call "dispersion number". It denotes a grouping of events for a given spectrometer that use the same dispersion calibration event, or more specifically have the same base "d-shift", which refers to the linear coordinate shift in the coordinate system used to determine the dispersion equation of the spectrometer. For example, if a control shot on an aluminum target is carried out in event 1, an no further control shots are conducted until event 12, then event 1-11 have the dispersion number 1. If the next control shot is done at event 21, then event 12-19 have the dispersion number 2, and so on. It is important to always add one to the dispersion number when taking a new control shot and to remember that the dispersion number is tied to the spectrometer, i.e. in the previous example if the initial spectrometers for event 1-11 are the SUCC and DUCC, then you have the dispersion number SUCC 1 and DUCC 1. If the spectrometers for event 12-19 are the SUCC and FSSR, then the dispersion numbers become SUCC 2 and FSSR 1, and so on.

Before the fun can really begin, some initial work is required. First, I will outline the steps that need to be done when initializing an experiment (done once before any shots are completed):
1. Navigate to jsonCreator.py in the directory .\Codes\Parameters. Here you can input all of the information required for data processing.
2. In jsonCreator.py, go to the dictionary "specPara". Here you need to input the properties of the spectrometers that remain unchanged. Follow the templates given there and make sure to input all the given properties correctly!
3. Navigate to the "disp" dictionary and copy a line, replacing the spectrometer name with your own.
4. Go to the controls of jsonCreator.py (from line 118) and turn on set_specPara and dispersion switches while turning off event and do_offset switches. Run the code.
5. Turn off the dispersion switch. DO NOT TURN IT BACK ON AGAIN.
6. Navigate to .\Codes\post_processing_lib.py. Go to the function calc_E() and find the dispersion function corresponding to your spectrometers. Check that they are correct.

The following steps are to be taken for the very first event:
1. Navigate to .\Codes\jsonCreator.py again. Go to the "EventInfo" dictionary. Create an entry for your first event, following the structure given in the examples. I would advise commenting out all of the example events from this point on. 
2. Go to the calls of the fill_filters() function. Input the information for the first event, again following the same structure as the examples. Comment out or delete the examples.
3. Open the json file E_offset.json in the Parameters dir. Delete everything in the file. Fill the file with only the characters {}. Save and close the file.
4. Go back to jsonCreator.py. Turn on the switches event and do_offset. Turn off set_specPara and dispersion.
5. Run jsonCreator.py
6. Make sure the switches set_specPara, event, and do_offset stay on from now on. 

Every time you start a new event from this point on, you only have to follow the steps:
1. Add the event information to the "EventInfo" dictionary in jsonCreator.py
2. Add the filter information to the calls of the fill_filters() function, adding new calls as neccessary.
3. Run jsonCreator.py

And that's all! Now you are free to use the capabilities of main.py to their full extent. It is important to note that you will likely have to add filter transmission info in the dir .\Codes\Filters. The details of how to do this is discussed in a future section.

## Using main.py

main.py serves as the main control hub of AXAWOTLS and will become your home away from home. With only the options given as switches and variables in the first section of the code, you will be able to process and analyze any spectra from your spectrometers. Each option is thoroughly commented, so it is advisable to familiarize yourself with every option before beginning (I promise there isn't many). The two most important switches are produce_spectra and data_analysis. Turning on produce_spectra will allow you to extract the data from the TIFF files and perform the universal processing steps by following the prompts in the terminal, producing basic spectra of counts over photon energy. Turning on data_analysis lets you conduct further analysis, e.g. applying corrections to the spectra and extracting final results. Both these main switches are mostly independent, excepting that produce_spectra must always be carried out first for a given event and spectrometer.  The general workflow is as follows, but feel free to mix and match as you choose (but make sure to always produce the spectra first!):
1. Turn on produce_spectra and new_data, while keeping data_analysis off. Follow the prompts in the terminal. 
2. Check the produced basic spectra then close the graphs.
3. Turn off produce_spectra and turn on data_analysis and simple_graphing. Turn off all other main switches of data analysis.
4. Choose the data analysis event settings (line 40).
5. Turn on all basic processing parameters. In the simple graphing parameters, turn on plot_together and turn off in_pixel.
6. Run main.py

This will produce a single graph containing the fully corrected and binned spectra of the chosen cases. If you choose to, you can play around with the various data analysis options to produce the results you want! If you get an error that says something along the lines of "data not found" for a filter csv, then skip to section "Adding filters" in this README to see what to do.

If you've successfully reached this point, you have completed all the prerequisites to make full use of AXAWOTLS! If you're an average user, you can ignore the rest of this README. Have fun and if you have any issues, feel free to contact me at my email carlosbutler210@gmail.com

##========================================================================================================================================================================================##

## Saving of results and data

AXAWOTLS automatically produces and saves results. The numerical data is always stored as csv files in the respective event directories, e.g. .\Events\0001\DUCC\Data. In this directory you will supplementary information as txt files, which should only be of note for a power user.

The graphs produced by the data analysis section of AXAWOTLS are saved in .\Codes\Graphs as pdf files with (hopefully) self explanatory names. It is important to leave the literature_a_spectrum directory here alone!

## Additional directories in .\Codes

\Parameters contains the parameters needed for running AXAWOTLS. The initialization and inputting of these parameters were discussed previously. It should contain 4 json files and one jsonCreator.py file. The json files have the following functions:
E_offset: saves the information about the articifical energy shifts the user can introduce in main.py, which serve to compensate shot-to-shot fluctuations. 
d_shift: contains the info about the d_shifts for each dispersion number of each spectrometer. These are calculated automatically inside of the calc_E() function in post_processing_lib.py and require a control shot on aluminum. 
set: contains the set of parameters for each event, which can change from shot to shot. Created by the "EventInfo" dictionary in jsonCreator.py
specPara: has the properties of the spectrometers that remain constant throughout the experiment. Intialized once in the beginning of the experiment in jsonCreator.py

\Camera has the data of the x-ray camera used on the spectrometers, which are applied in data_analysis_lib.py to correct out the camera influences. The current data is for the GE-VAC 2048 512 series from the company greateyes.

\FSSR contains the results found by a ray tracing simulation conducted by Artem. Should not be relevant for future experiments, but is used for the example events given in the base version of AXAWOTLS.

\Filters has all the filter data. This will be discussed in the next section

\Graphs is discussed in the previous section

## Adding filters

The filter transmission data for every combination of filter material and relative thickness (usually dependent on the spectrometer) needs to be added manually. To do this, you must:
1. Go to the website https://henke.lbl.gov/optical_constants/filter2.html
2. Input the desired material, thickness, and energy range
3. Submit the request and click on "data file here"
4. Save the csv by directly copying the information on the webpage into an empty txt file
5. Save the txt file in .\Codes\Filters using the naming scheme: material_thickness (in micron and with a comma for decimals), e.g. aluminum_5,07 for an aluminum filter with thickness of 5.07 microns

If you are unsure of what filter cases you are missing, you can simply run the code FilterCalc.py. It will output all the unique combinations of filter material and thickness across all the spectrometers and events.

## Libraries and supplementary codes

The heavy lifting of AXAWOTLS is done by the function libraries, which are imported into main.py and sometimes into other libraries. Every module is heavily commentated and mostly self-contained. If you wish for a deeper understanding of the inner workings of AXAWOTLS, feel free to peruse the corresponding library.

Each library has the purpose:

spectrum_processing_lib: This library contains the functions used by the main switch produce_spectra. It assists in extracting the data from the TIFF images and creating basic spectra out of them (uncorrected), as well as performing the calcuations and visualization of the absorption coefficient. 

post_processing_lib: a smaller library that is responsible for the more detailed processing procedures of the spectra and can be seen as being relevant for both produce_spectra and data_analysis. The most important function here is calc_E(), which converts pixel count into photon energy and determines d_shifts for the dispersion numbers. Note that further processing, like filter and camera corrections, are conducted by data_analysis_lib.

general_lib: an even smaller library that defines general functions used throughout AXAWOTLS. There is no overarching theme here, just useful supplementary functions.

data_analysis: here the juiciest stuff happens. It falls under the shadow of data_analysis switch of main.py and carries out the meat of the processing to reach the most important results. The finer details of the analysis can be controlled by the switches and variables found in the beginning of the library. These are heavily commentated and hopefully understandable. 


As of writing this, AXAWOTLS has a single supplementary code called event_finder.py. It can be used to find the event numbers fulfilling certain requirements and is very useful once the number of events becomes large. Note that it may have to be adjusted when adding in new spectrometers.

## Closing=============================================================================================================================================================================

That's all the information I can think of writing down for now. I did my best to keep the code readable and well-commented, so please check the comments and code itself if you ever run into bugs or issues. If anything especially perplexing should come up, don't hesitate to contact me at carlosbutler210@gmail.com

I hope you enjoy using AXAWOTLS and happy experimenting!

