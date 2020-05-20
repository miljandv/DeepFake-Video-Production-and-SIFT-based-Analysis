
import numpy as np
import matplotlib.pyplot as grafik

P=np.load(r"C:\Users\milja\OneDrive\Desktop\SIFT\4.npz")['arr_0']

grafik.figure(0)
grafik.plot(P,color='blue',linewidth=2)
grafik.grid(True)
grafik.xlabel('Frame index',fontsize=12,weight='bold')
grafik.ylabel('Frame Matching [%]',fontsize=12,weight='bold')
grafik.xlim(1,len(P))

grafik.show()
