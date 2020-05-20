
import numpy as np
import matplotlib.pyplot as grafik
from scipy import signal
#---------------------------------------------------------------------------------------------------
P1=np.load(r"C:\Users\milja\OneDrive\Desktop\SIFT\2.npz")['arr_0']
P2=np.load(r"C:\Users\milja\OneDrive\Desktop\SIFT\new.npz")['arr_0']
#---------------------------------------------------------------------------------------------------
grafik.figure(0)
grafik.subplot(2,1,1)
grafik.plot(P1,color='blue',linewidth=1.2,label='Original Video')
grafik.plot(P2,color='red',linewidth=1.2,label='DeepFake Video')
grafik.grid(True)
grafik.ylabel('Frame Matching [%]',fontsize=12,weight='bold')
grafik.xlim(1,np.amin([len(P1),len(P2)]))
grafik.legend(loc='best')
grafik.title('Original Frame Matching',fontsize=12,weight='bold')
#---------------------------------------------------------------------------------------------------
N=5
B=np.ones(N)/N
A=1
P1f=signal.filtfilt(B,A,P1,padlen=10)
P2f=signal.filtfilt(B,A,P2,padlen=10)
#---------------------------------------------------------------------------------------------------
grafik.subplot(2,1,2)
grafik.plot(P1f,color='blue',linewidth=1.2,label='Original Video')
grafik.plot(P2f,color='red',linewidth=1.2,label='DeepFake Video')
grafik.grid(True)
grafik.xlabel('Frame index',fontsize=12,weight='bold')
grafik.ylabel('Frame Matching [%]',fontsize=12,weight='bold')
grafik.xlim(1,np.amin([len(P1f),len(P2f)]))
grafik.legend(loc='best')
grafik.title('Moving Average N=5',fontsize=12,weight='bold')
#---------------------------------------------------------------------------------------------------
grafik.tight_layout()
grafik.show()
#---------------------------------------------------------------------------------------------------
