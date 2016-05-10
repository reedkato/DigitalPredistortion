#Python2.7
#DPD data calculation code.
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

#Load amplifier input and output data.
#L1: Input pwr, L2: Output pwr, L3: Output phase.
data = np.loadtxt('ampdata.txt',comments='#',delimiter='\t')
Inpwr_1  = data[:,0]	#Input power when phase data measurement.
pha_deg = data[:,1]	#Phase data.
Inpwr_2 = data[:,2]	#Input power when phase data measurement.
Outpwr = data[:,3]	#Putput power data.

#AMAM data normalization.
Outamp = np.sqrt(np.power(10,Outpwr/10))
Outamp_n = Outamp/np.nanmax(Outamp)
Inamp_AMAM = np.sqrt(np.power(10,Inpwr_2/10))
Inamp_AMAM_n = Inamp_AMAM/Inamp_AMAM[np.nanargmax(Outamp)]


#Linear function fitting in normalized input & output data.
#Define linear fitting area manually!!
myX = Inamp_AMAM_n[0:120]
myY = Outamp_n[0:120]
myA = np.array([myX,np.ones(len(myX))])
myA = myA.T
slope,intercept = np.linalg.lstsq(myA,myY)[0]

#Normalized linear output data.
Lin_n = slope*Inamp_AMAM_n
Lin_n = Lin_n.clip(0.0,1.0)

#Data extraction except output saturation area.
Inamp_n0 = Inamp_AMAM_n[0:np.nanargmax(Outamp_n)]
Outamp_n0 = Outamp_n[0:np.nanargmax(Outamp_n)]
Lin_n0 = Lin_n[0:np.nanargmax(Outamp_n)]

#Discritisized input.
mybin = 2**4#<-Discritisization bit.
In_discrit = np.linspace(np.min(Inamp_AMAM_n), 1.0/slope, mybin)
In_discrit1 = np.linspace(0, 1.0/slope, mybin)
Out_lin_discrit = slope*In_discrit

#Find nearest output value in AMAM output data.
#If there is no AMAM data, assume no distortion.
Pd_in = Inamp_AMAM_n[np.searchsorted(Outamp_n0, Out_lin_discrit)]
Pd_out = Outamp_n[np.searchsorted(Outamp_n0, Out_lin_discrit)]

#Interpolation.
"""
In_discrit_ip = [x for x in In_discrit if x <=np.min(Inamp_AMAM_n)] 
Pd_in = [x for x in Pd_in if x >  np.min(Inamp_AMAM_n)] 

Out_discrit_ip = [x for x in Out_lin_discrit if x <=np.min(Outamp_n)] 
Pd_out = [x for x in Pd_out if x >  np.min(Outamp_n)]

#Data concatenation.
Pd_in = np.r_[In_discrit_ip,Pd_in]
Pd_out = np.r_[Out_discrit_ip,Pd_out]
"""
#Calculate phase rotation effect.
Inamp_AMPM = np.sqrt(np.power(10,Inpwr_1/10))
#Scaling factor shuld be the same in AMAM &AMPM.
Inamp_AMPM_n = Inamp_AMPM/Inamp_AMAM[np.nanargmax(Outamp)]
#Find nearest output from AMPM output data.
#If there is no AMAM data, assume no distortion.
pha_rad = pha_deg/180.0*np.pi
pha_rad_discrit = pha_rad[np.searchsorted(Inamp_AMAM_n,Pd_in)]

#Calcurate predistortion data.
#To avoid 0/0.
#In_discrit0 = [x for x in In_discrit if x !=0] 
In_discrit0 = In_discrit 
#In_discrit0 =np.r_[1, In_discrit0]
#Pd_in0 = [x for x in Pd_in if x !=0] 
Pd_in0 = Pd_in 
#Pd_in0 =np.r_[1, Pd_in0]

Ipd = Pd_in0 / In_discrit0 * np.cos(- pha_rad_discrit)
Qpd = Pd_in0 / In_discrit0 * np.sin(- pha_rad_discrit)

#Discritisized input lower limit & upper limit.
l_lim = In_discrit - 1.0/slope/mybin/2.0
#l_lim = [x if x > 0 else 0 for x in l_lim]
#u_lim = [x for x in l_lim if x !=0]
u_lim = In_discrit + 1.0/slope/mybin/2.0
#u_lim = np.r_[u_lim, 1.0]

#Below codes are used for visualization.
#Output saturation area data cut.
Inpha_n0 = Inamp_AMPM_n[0:np.nanargmax(Outamp_n)]
Outpha_n0 = pha_rad[0:np.nanargmax(Outamp_n)]

#Data output.
outdarr1=np.array([l_lim, u_lim, Pd_out, pha_rad_discrit, Ipd, Qpd])
outdarr1=outdarr1.T
header = '{0}i\t{1}\t{2}\t{3}\t{4}\t{5}'.format("Input_llim", "Input_ulim", "Pd_amplitude", "Pd_phase", "Ipd", "Qpd")
np.savetxt('pd.out', outdarr1, fmt='%1.6e', delimiter='\t', header=header)

###Data plot
#plt, ((ax1L, ax1R),(ax2L,ax2R)) = plt.subplots(ncols=2, nrows=2, figsize=(12,8), sharex=True)
plt, ((ax1L, ax1R),(ax2L,ax2R)) = plt.subplots(ncols=2, nrows=2, figsize=(14,10))

ax1L.plot(Inamp_n0, Outamp_n0, ',', label = "AMAM data")
ax1L.plot(In_discrit, Out_lin_discrit,'.', label = "Discritisized linear out")
ax1L.plot(Pd_in, Pd_out, '.' , label = "PD out")
ax1L.plot(l_lim, Pd_out, '>k', label = "Discritisized lower limit input")
ax1L.plot(u_lim, Pd_out, '<k', label = "Discritisized upper limit input")
ax1L.set_title('AMAM')
ax1L.set_xlabel('Normalized input amplitude[A. U.]')
ax1L.set_ylabel('Normalized output amplitude[A. U.]')
ax1L.grid(True)
ax1L.legend(loc='best', numpoints=1, fontsize=10)
ax1L.set_xlim([0,1])
ax1L.set_ylim([0,1])

ax1R.plot(Inpha_n0, Outpha_n0,',', label = "AMPM data")
ax1R.plot(Pd_in, pha_rad_discrit,'.r', label = "PD out")
ax1R.set_title('AMPM')
ax1R.set_xlabel('Normalized input amplitude[A. U.]')
ax1R.set_ylabel('Output phase[rad]')
ax1R.grid(True)
ax1R.legend(loc='best', numpoints=1, fontsize=10)
ax1R.set_xlim([0,1])

ax2L.plot(In_discrit, Ipd,'.r')
ax2L.set_title('Ipd')
ax2L.set_xlabel('Normalized input amplitude[A. U.]')
ax2L.set_ylabel('In-phase predistortion [A.U.]')
ax2L.grid(True)
ax2L.set_xlim([0,1])

ax2R.plot(In_discrit, Qpd,'.r')
ax2R.set_title('Qpd')
ax2R.set_xlabel('Normalized input amplitude[A. U.]')
ax2R.set_ylabel('Quadrature-phase distoriton [A.U.]')
ax2R.grid(True)
ax2R.set_xlim([0,1])

plt.show()
plt.savefig("test.png")

