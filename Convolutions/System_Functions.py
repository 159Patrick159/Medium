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
from scipy.signal import butter, cheby1, cheby2, bessel, freqresp
from scipy.special import jv

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

def MovingAverage(boxsize):
    '''
    ######################################################
    Description: Return boxcar filter where amplitude
    is defined as 1/N where N is the length of the 
    boxcar
    ######################################################
    Inputs: 
    boxsize: number of elements included in boxcar
    ######################################################
    Outputs:
    w: Boxcar array where each value is 1/N where N is
    the length of the boxcar   
    ######################################################
    '''
    return np.ones(boxsize)/boxsize
    

def sinc_function(f,k):
    '''
    ######################################################
    Description: Returns normalized sinc function. Which 
    is ideal for low pass filters with a flat frequency 
    response in its passband.
    ######################################################
    Inputs:
    f: frequency array
    k: horizontal compression:
    ######################################################
    Outputs:
    w: normalized sinc function of len f
    ######################################################
    '''
    return np.sinc(f*k) # Numpy returns a normalized sinc funciton

def ideal_boxcar(f,flow,fhigh):
    '''
    ######################################################
    Description: Returns boxcar filter for specified
    frequency range. This could be an idealized low-pass,
    high-pass or band-pass filters.
    ######################################################
    Inputs:
    f: frequency array
    flow: lower bound for frequency filter
    fhigh: upper bound for frequency filter
    ######################################################
    w: Boxcar filter for specified frequency range
    ######################################################
    '''
    return np.where(np.logical_and(f <= fhigh, f>= flow),1,0)

def bessel_function(f, n=0, omega=1):
    '''
    ######################################################
    Description: Return bessel function for specified 
    frequency array. Ideal for maximally flat group delay
    ######################################################
    Inputs: 
    f: frequency array
    n: order of bessel function set to 0
    w: scale of frequency array
    '''
    return jv(n, omega * f)

# 10. Chebyshev Polynomials (Transfer Function)
def chebyshev1_filter(order, ripple, Wn):
    return cheby1(order, ripple, Wn, output='sos')

def chebyshev2_filter(order, attenuation, Wn):
    return cheby2(order, attenuation, Wn, output='sos')

# 11. Butterworth Response (Transfer Function)
def butterworth_filter(order, Wn):
    return butter(order, Wn, output='sos')