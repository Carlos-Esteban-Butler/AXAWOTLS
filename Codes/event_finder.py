import sys
import general_lib as gl

# choices
do_laser_E  =  0 # search for laser energy
do_spec    =   1 # search according to spectrometer
do_target   =  1 # according to backlighter target
energy_range = [1,35]
target     =   ["PTFE"]
spectro    =   [0] # 0 is SUCC, 1 is DUCC, 2 is FSSR, 3 is OSUCC, 4 is ESUCC

# search
event_info = gl.import_event_parameters()
for i in range(len(spectro)):
    if spectro[i] == 0: spectro[i] = "SUCC"
    elif spectro[i] == 1: spectro[i] = "DUCC"
    elif spectro[i] == 2: spectro[i] = "FSSR"
    elif spectro[i] == 3: spectro[i] = "OSUCC"
    elif spectro[i] == 4: spectro[i] = "ESUCC"
    else: print("Error: "+str(i)+" is not a valid option for spectro."); sys.exit()
results = []
for eventNum in event_info:
    energy = float(event_info[eventNum]["energy"])
    add_to_result = 0
    if event_info[eventNum]["target"] in target and do_target:
        add_to_result += 1
    for event_spec in event_info[eventNum]["spect"]:
        if event_spec in spectro and do_spec:
            add_to_result += 1
            break
    if energy > energy_range[0] and energy < energy_range[1] and do_laser_E:
        add_to_result += 1
    if add_to_result == do_laser_E + do_spec + do_target: results.append(eventNum)
print(results)