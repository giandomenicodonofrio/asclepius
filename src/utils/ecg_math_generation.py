from math import pi, cos
import matplotlib.pyplot as plt
import numpy as np 

""" 
REFS
https://pastebin.com/vS0bKxX3
https://www.quora.com/What-is-the-equation-for-the-heartbeat-waveform
"""

digacc=1e-1000
 
def PTUwave(x,wavamp,wavdur,wavint,hrate):
    ecg_ptu=0    
    i=1
    a=wavamp
    b=2*hrate/wavdur
    x=x+wavint
    a0=2/(b*pi)
    anOld=0.0
    diff=1000
    #digacc=1
    #ecg=0
    anNew=0
    while(diff>digacc):
        coeff=4*b*cos(i*pi/b)/((b-2*i)*(b+2*i)*pi)
        anNew=anOld+coeff*cos(i*pi*x/hrate)
        diff=abs(anNew-anOld)
        anOld=anNew
        i+=1
    
    ecg_ptu+=a*(a0+anNew)
    return ecg_ptu
    
def qrs_segment(x,aqrswav,dqrswav,hrate):
    ecg_qrs=0    
    a=aqrswav
    b=2*hrate/dqrswav
    a0=0.5*a/b
    anOld=0.0
    i=1
    #digacc=1
    diff=1000
    #ecg=0
    anr=0   
    while(diff>digacc):
        coeff=2*a*b*(1-cos(i*pi/b))/(i*pi)**2
        anr=anOld+coeff*cos(i*pi*x/hrate)
        diff=abs(anr-anOld)
        anOld=anr
        i+=1
 
    ecg_qrs=a0+anr
    return ecg_qrs
    
def qswave(x,ASwav,DSwav,TQwav,hrate):
    ecg_qs=0   
    a=ASwav
    b=2*hrate/DSwav
    x=x+TQwav
    a0=0.5*a/b
    anOld=0.0
    i=1
    #digacc=1
    diff=1000
    #ecg=0
    while(diff>digacc):
        coeff=2*a*b*(1-cos(i*pi/b))/(i*pi)**2
        anNew=anOld+coeff*cos(i*pi*x/hrate)
        diff=abs(anNew-anOld)
        anOld=anNew
        i+=1
    
    ecg_qs+=-1*(a0+anNew)
    return ecg_qs
 
#PTUx3
        #1: APwav=0.25,DPwav=0.12,TPwav=0.12,Hrate
        #2: ATwav=0.35, DTwav=0.066, -(TTwav=0.2+0.045),Hrate
        #3: AUwav=0.025, DUwav=0.048, -TUwav=0.433, Hrate
    #QSx2
        #1: AQwav=0.2, DQwav=0.04, TQwav=0.166, Hrate
        #2: ASwav=0.025, DSwav=0.066, -TSwav=0.08, Hrate
    #QRSx1
        #1: AQRSwav=1.6, DQRSwav=0.06, Hrate
 
def ecg_wav_generator(hrate,
            APwav,DPwav,TPwav,
            ATwav,DTwav,TTwav,
            AUwav,DUwav,TUwav,
            AQwav,DQwav,TQwav,
            ASwav,DSwav,TSwav,
            AQRSwav,DQRSwav):
    
    ecgvals=[]
    xvals=[]
    ecg=0
    x=0.45
 
    
    for i in range(-3000,3000):                  
        ecg=PTUwave(x,APwav,DPwav,TPwav,hrate)+ \
            PTUwave(x,ATwav,DTwav,-(TTwav+0.045),hrate)+ \
            PTUwave(x,AUwav,DUwav,-TUwav,hrate)+ \
            qswave(x,AQwav,DQwav,TQwav,hrate)+ \
            qswave(x,ASwav,DSwav,-TSwav,hrate)+ \
            qrs_segment(x,AQRSwav,DQRSwav,hrate)
        
        xvals.append(x)
        x+=0.01
        ecgvals.append(ecg+1)
        
    return [ecgvals,xvals]    

 
#hrate=30/72,
#            APwav=0.25,DPwav=0.12,TPwav=0.12,
#            ATwav=0.35, DTwav=0.066,TTwav=0.2,
#            AUwav=0.025, DUwav=0.048, TUwav=0.433,
#            AQwav=0.2, DQwav=0.04,TQwav=0.166,
#            ASwav=0.025, DSwav=0.066,TSwav=0.08,
#            AQRSwav=1.6, DQRSwav=0.06
#
#hrate=1.5,
#            APwav=0.25,DPwav=0.09,TPwav=0.16,
#            ATwav=0.35, DTwav=0.142,TTwav=0.2,
#            AUwav=0.035, DUwav=0.0476, TUwav=0.433,
#            AQwav=0.025, DQwav=0.066,TQwav=0.166,
#            ASwav=0.25, DSwav=0.066,TSwav=0.08,
#            AQRSwav=1.6, DQRSwav=0.11