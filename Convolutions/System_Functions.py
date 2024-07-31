################ HEADER ################
# Author: Patrick Sandoval             #
# Date: 2024-07-31                     #
########################################    
# Description: The following scripts   #
# contains the discretized system      #
# system functions often used in       #
# signal processing tasks.             #
########################################
import numpy as np

def boxcar(t,T):
    '''
    ######################################################
    Description: Returns a boxcar filter for range [0,T]
    ######################################################
    Inputs:
    t: time array of signal
    T: upper bound of boxcar   
    ######################################################
    Outputs:
    w: boxcar array of size len(t)
    ######################################################
    '''
    box = []
    for x in t:
        if x>=0 and x<=T:
            box.append(1)
        else:
            box.append(0)
    return(np.array(box))

def Hann(t,T):
    '''
    ######################################################
    Description: Retruns a hann window filter for range
    [0,T].
    ######################################################
    Inputs:
    t: time array of signal
    T: upper bound of window
    ######################################################
    Outputs:
    w: hann window array of size len(t)
    ######################################################
    '''
    H = []
    for x in t:
        if x>=0 and x<=T:
            H.append(0.5*(1-np.cos((2*np.pi*x)/T)))
        else:
            H.append(0)
    return(np.array(H))