from cv2 import *
import numpy as np
import sys
#citanje video frejm po frejm-----------------------------------------------------------------------
video_fajl=r"C:\Users\milja\OneDrive\Desktop\Videos\new.avi"
cap = VideoCapture(video_fajl)
broj_frejmova=int(cap.get(CAP_PROP_FRAME_COUNT))
i=0
frejmovi=[]
while(i<broj_frejmova):
    ret,frame=cap.read()
    frejmovi.append(frame)
    i+=1
cap.release()
#parametri SIFT algoritma---------------------------------------------------------------------------
sift = xfeatures2d.SIFT_create()
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = FlannBasedMatcher(index_params, search_params)
#poredimo po dva susedna frejma---------------------------------------------------------------------
procenti=[]
for j in np.arange(broj_frejmova-1):
    frejm_1=frejmovi[j]
    frejm_2=frejmovi[j+1]
    #kljucne tacke i deskriptori
    kt1, deskriptor_1 = sift.detectAndCompute(frejm_1, None)
    kt2, deskriptor_2 = sift.detectAndCompute(frejm_2, None)
    #poredjenje
    matches = flann.knnMatch(deskriptor_1, deskriptor_2, k=2)
    uracunate_tacke = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            uracunate_tacke.append(m)
    #-----------------------------------------------------------------------------------------------
    uracunaj = np.amin([len(kt1),len(kt2)])
    procenti.append(round(len(uracunate_tacke)/uracunaj*100,2))
    print(j/broj_frejmova*100)
#---------------------------------------------------------------------------------------------------
procenti=np.asarray(procenti)
np.savez(r"C:\Users\milja\OneDrive\Desktop\SIFT\new.npz",procenti)
