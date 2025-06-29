import json
import os
import numpy as np


EventInfo ={
  1 : {
    "target" : "Gd",
    "energy" : 87.4,
    "spect" : {
        "SUCC" : {
            "disp" : 1, # dispersion number
            "type" : "source" # type of data (source or abs=absorption). Essentially with or without a sample
        },
        "DUCC" : {
            "disp" : 1,
            "type" : "abs",
            "thickness" : 0.8 # thickness of the sample. Required if type = abs
        }
    },
    "absType" : "cold" # whether or not the sample experienced preheat from the source (cold or hot)
  },
  2: {
    "target" : "PTFE",
    "energy" : 110.3,
    "spect" : {
        "SUCC" : {
            "disp" : 2,
            "type" : "source"
        },
        "OSUCC" : {
            "disp" : 1,
            "type" : "source",
            "oldCrystal" : "no" # can leave this out in future experiments. Was relevent for the May 2023 experiment
        }
    }
  },
  3: {
    "target" : "PTFE",
    "energy" : 88.9,
    "spect" : {
        "SUCC" : {
            "disp" : 2,
            "type" : "source"
        },
        "OSUCC" : {
            "disp" : 1,
            "type" : "abs",
            "thickness" : 0.8,
            "oldCrystal" : "no"
        }
    },
    "absType" : "cold"
  },
  4: {
    "target" : "PTFE",
    "energy" : 30.8,
    "spect" : {
        "SUCC" : {
            "disp" : 2,
            "type" : "source"
        },
        "FSSR" : {
            "disp" : 1,
            "type" : "abs",
            "thickness" : 0.8,
            "focus" : "no" # only required for spherically bent crystal spectrometers. Whether or not the rays were fully focused
        }
    },
    "absType" : "hot"
  },
  5: {
    "target" : "PTFE",
    "energy" : 30.4,
    "spect" : {
        "SUCC" : {
            "disp" : 2,
            "type" : "source"
        },
        "FSSR" : {
            "disp" : 1,
            "type" : "source",
            "focus" : "no"
        }
    }
  },
  6 : {
    "target" : "Al",
    "energy" : 27,
    "spect" : {
        "SUCC" : {
            "disp" : 3,
            "type" : "source"
        },
        "DUCC" : {
            "disp" : 2,
            "type" : "source"
        }
    }
  },
  7 : {
    "target" : "Al",
    "energy" : 24,
    "spect" : {
        "SUCC" : {
            "disp" : 4,
            "type" : "source"
        },
        "FSSR" : {
            "disp" : 2,
            "type" : "source",
            "focus" : "no"
        }
    }
  },8 : {
    "target" : "Al",
    "energy" : 24,
    "spect" : {
        "SUCC" : {
            "disp" : 5,
            "type" : "source"
        },
        "FSSR" : {
            "disp" : 3,
            "type" : "source",
            "focus" : "no"
        }
    }
  },
} 

# control which jsons to make
event = 1 # write set parameter json file
set_specPara = 1 # write the setPara (spectrometer parameters) info into json file
do_offset =  1 # only for initializing E offset

#!!!! BE VERY CAREFUL WITH THE FOLLOWING OPTIONS, WILL ERASE PROGRESS!!#
dispersion = 0 # intialize d_shift info into file
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

# Define the parameters of the spectrometers themselves that don't vary between shots
specPara = {
  "DUCC" : {
    "channels" : "double", # how many channels it has (single or double)
    "crystal bend" : "flat", # type of crystal bending (flat or spherical)
    "knife edge" : "no", # is there a knife edge used for the image
    "length" : 117.584*2, # length from source to detector [mm]
    "order" : 1, # order of refraction of crystal in integer
    "lattice spacing" : 10.64, # lattice spacing of crystal [angstrom]
    "E-central" : 1578.42, # central energy of the spectrometer [eV]
    "R_int" : 40 # expected integrated reflectivity [micro rad] (gilfrich 1975)
  },
  "SUCC" : {
    "channels" : "single", # how many channels it has (single or double)
    "crystal bend" : "flat", # type of crystal bending (flat or spherical)
    "knife edge" : "yes", # is there a knife edge used for the image
    "length" : 325.2, # length from source to detector [mm]
    "order" : 1, # order of refraction of crystal in integer
    "lattice spacing" : 26.632, # lattice spacing of crystal [angstrom]
    "E-central" : 1572, # central energy of the spectrometer [eV]
    "R_int" : 80 # expected integrated reflectivity [micro rad] (loisel2016, fig 3)
  },
  "OSUCC" : {
    "channels" : "single", # how many channels it has (single or double)
    "crystal bend" : "flat", # type of crystal bending (flat or spherical)
    "knife edge" : "no", # is there a knife edge used for the image
    "length" : 600, # length from source to detector [mm]
    "order" : 1, # order of refraction of crystal in integer
    "lattice spacing" : 26.632, # lattice spacing of crystal [angstrom]
    "E-central" : 1576.07, # central energy of the spectrometer [eV]
    "R_int" : 80 # expected integrated reflectivity [micro rad] (loisel2016, fig 3)
  },
  "FSSR" : {
    "channels" : "single", # how many channels it has (single or double)
    "crystal bend" : "spherical", # type of crystal bending (flat or spherical)
    "knife edge" : "no", # is there a knife edge used for the image
    "radius of curvature" : 155.04, # radius of curvature of crystal [mm]
    "a0" : 549.71, # source to center of crystal [mm]
    "order" : 2, # order of refraction of crystal in integer
    "lattice spacing" : 19.84, # lattice spacing of crystal [angstrom]
    "E-central" : 1600.08, # central energy of the spectrometer [eV]
    "R_int" : 2.752, # expected integrated reflectivity [micro rad] (from artem's simulation)
    "crystal width" : 10 # width of the crystal [mm]
  },
}

# convert areal density in microgram/cm2 to micron
def convert_ad_to_micron(mat, thick):
    carbon_density = 2.266*10**6 # is [g/cm3] --> [mu g/cm3]. Uses density of pure carbon
    gold_density = 19.3*10**6 # is [g/cm3] --> [mu g/cm3]. Uses density of pure gold
    for i in range(len(mat)):
        if (mat[i] == "carbon"):
            thick[i] *= 10000/carbon_density # divide by density then convert to micron
            thick[i] = round(thick[i], 3) # round to 3 decimal places
        if (mat[i] == "gold"):
            thick[i] *= 10000/gold_density # divide by density then convert to micron
    return thick

# convert base thickness to one that takes the angle of the rays into account
def calc_thick(thick, spec):
    n = specPara[spec]["order"]; twod = specPara[spec]["lattice spacing"]; Ecentral = specPara[spec]["E-central"]
    C = 12398 # Umrechnungsfaktor
    theta0 = np.arcsin(n*C/(twod*Ecentral)) # central Bragg angle
    if specPara[spec]["crystal bend"] == "spherical": theta0 = 0 # as rays are approx perpendicular to filters for FSSR
    adjustedThick = [round(i/np.cos(theta0), 2) for i in thick] # calc adjustment and round to 2 decimal places
    return adjustedThick

# fill the dictionary with filter info. extraMat and extraThicc ;) refer to spec1 for eKAP and ADP
def fill_filters(startEvent, endEvent, mat1, thick1, mat2, thick2, extraMat=[], extraThick=[]):
    spectro = list(EventInfo[startEvent]["spect"].keys())
    # add on mylar filter for all as present in each spectrometer
    mat1.append("mylar"); thick1.append(2); mat2.append("mylar"); thick2.append(2)
    if extraMat != [] and extraThick != []: extraMat.append("mylar"); extraThick.append(2) 
    # convert areal density to micron
    thick1 = convert_ad_to_micron(mat1, thick1)
    thick2 = convert_ad_to_micron(mat2, thick2)
    # adjust the thickness to account for ray angles
    thick1 = calc_thick(thick1, spectro[0])
    thick2 = calc_thick(thick2, spectro[1])
    if(extraMat != [] and extraThick != []): 
        extraThick = convert_ad_to_micron(extraMat, extraThick)
        extraThick = calc_thick(extraThick, spectro[1])
    for i in range(startEvent, endEvent+1):
        EventInfo[i]["spect"][spectro[0]]["filterType"] = mat1
        EventInfo[i]["spect"][spectro[0]]["filterThickness"] = thick1
        EventInfo[i]["spect"][spectro[1]]["filterType"] = mat2
        EventInfo[i]["spect"][spectro[1]]["filterThickness"] = thick2
        if(extraMat != [] and extraThick != []):
            EventInfo[i]["spect"][spectro[1]]["filterType1"] = extraMat
            EventInfo[i]["spect"][spectro[1]]["filterThickness1"] = extraThick
    return

# carbon and gold numbers in microgram/cm2, while all others in micron. After fill_filters, all numbers in micron
fill_filters(1, 1, ["carbon", "gold", "pokalon"], [915, 950*2, 15], ["carbon", "pokalon"], [976.4, 15], ["carbon", "pokalon"], [815.8, 15])
fill_filters(2, 2, ["carbon", "pokalon"], [915, 15], ["carbon", "pokalon"], [533, 15])
fill_filters(3, 3, ["carbon", "pokalon"], [915, 15], ["carbon", "pokalon"], [533, 15])
fill_filters(4, 5, ["carbon", "pokalon"], [915, 15], ["carbon", "pokalon"], [871, 15])
fill_filters(6, 6, ["carbon", "gold", "pokalon"], [915, 950*2, 15], ["carbon", "pokalon"], [976.4, 15], ["carbon", "pokalon"], [815.8, 15])
fill_filters(7, 7, ["carbon", "gold", "pokalon"], [915, 950*2, 15], ["carbon", "gold"], [871, 950*2])

# write the dict into a json file
if event:
    file = os.path.dirname(os.path.realpath(__file__)) + "\set.json"
    with open(file, 'w') as f: 
        json.dump(EventInfo, f)

# write the spectrometer parameter dict into json file
if set_specPara:
    file = os.path.dirname(os.path.realpath(__file__)) + "\specPara.json"
    with open(file, 'w') as f: 
        json.dump(specPara, f)

# only for initializing offset
if do_offset:
    scriptdir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(scriptdir, "E_offset.json"), 'r') as f:
        offset = json.load(f)
    for event_num in EventInfo:
        event_num = str(event_num)
        if event_num not in offset:
            offset[event_num] = {}
        for spectName in EventInfo[int(event_num)]["spect"]:
            if spectName not in offset[event_num]:
                offset[event_num][spectName] = {}
            if "E_offset" not in offset[event_num][spectName]:
                offset[event_num][spectName]["E_offset"] = 0 # if theres no E_offset saved by user, then initialize and use 0 [eV]
                if (specPara[spectName]["channels"] == "double"):
                    offset[event_num][spectName]["E_offset_1"] = 0 # similar convention to spec and spec1 in pullAndSaveData()
    offset = dict(sorted(offset.items(), key=lambda item: int(item[0])))
    # save the initialized offset. If an offset value already exists for an event, it will not be overwritten.
    with open(os.path.join(scriptdir, "E_offset.json"), 'w') as f: 
        json.dump(offset, f)

# initialize dispersion shift json file. should only need to do once at beginning of beamtime
disp = {"SUCC": [0],
        "OSUCC": [0],
        "DUCC": [0],
        "DUCC extra": [0],
        "FSSR": [0]
}

if dispersion:
    file = os.path.dirname(os.path.realpath(__file__)) + "\d_shift.json"
    with open(file, 'w') as f: 
        json.dump(disp, f)