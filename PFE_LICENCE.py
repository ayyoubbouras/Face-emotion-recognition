#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#LA PHASE D'APPRENTISSAGE
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image 
import glob 
#Importation des images 
img_dir = "Desktop/Baesdedonnees/Apprentissage"  
data_path = os.path.join(img_dir,'*g') 
files = glob.glob(data_path) 
dataset = [] 
for f1 in files:
    img = cv2.imread(f1)
    img = PIL.Image.open(f1)
    img = img.resize((58,49), PIL.Image.ANTIALIAS)
    img2 = np.array(img).flatten()
    dataset.append(img2) 
face_vector = np.asarray(dataset)  # ETAPE 1:Creation de la matrice 
des donnees
face_vector = face_vector.transpose()    
#ETAPE 2:Normalisation de la matrice des donnees 
avg_face_vector = face_vector.mean(axis=1)
avg_face_vector = avg_face_vector.reshape(face_vector.shape[0], 1)
normalized_face_vector = face_vector - avg_face_vector
#ETAPE 3 :Creation de la matrice de correlation
covariance_matrix = np.cov(np.transpose(normalized_face_vector))
#ETAPE 4 :Determination des valeurs et des vecteurs propres
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
#ETAPE 5 :Determination de l’espace propre
eigen_vectors = eigen_vectors.argsort()
k_eigen_vectors = eigen_vectors[0:16, :]
#ETAPE 6 :La projection des visages d’apprentissage sur l’espace propre
eigen_faces = k_eigen_vectors.dot(normalized_face_vector.T)
weights = (normalized_face_vector.T).dot(eigen_faces.T)
\end{lstlisting}
\end{appendices}
\newpage
\begin{appendices}
\chapter*{Appendix B}
\label{appendix:b}
\begin{lstlisting}[language=Python,caption=INTERFACE GRAPHIQUE ET PHASE DE TEST]
#INTERFACE GRAPHIQUE
from tkinter import *
import tkinter as tk  
from tkinter import ttk
from tkinter import filedialog, Text
import os
import PIL.Image
import PIL.ImageTk
from resizeimage import resizeimage

def ChoisirImage():
    global label1
    for widget in frame1.winfo_children():  
        widget.destroy()
    label1 = Label(frame1, text='' ,bg="#d6eaf2" )
    label1.pack(pady=32)
    if(label1.winfo_exists()==1):
        label1.destroy()
    global filename
    for widget in frame.winfo_children():  
        widget.destroy()
    filename = filedialog.askopenfilename(initialdir="/home/ubuntu/Desktop",
    title= "Choisissez Une Image :")
    
    if filename != "":
        with open(filename, 'rb') as file:
            image = PIL.Image.open(file)
            resized_image = resizeimage.resize_cover(image, [270, 260],
            validate=False)
            resized_image.save('resized_image.png', image.format)
        photo = PIL.ImageTk.PhotoImage(file='resized_image.png')
        Artwork = Label(frame, image=photo)
        Artwork.photo = photo
        Artwork.pack()
    
def AppliquerAcp():
    #La PHASE DE TEST
    img = PIL.Image.open(filename)
    img = img.resize((2842, 1), PIL.Image.ANTIALIAS)
    test_normalized_face_vector = avg_face_vector.T - img
    test_normalized_face_vector=test_normalized_face_vector.T
    test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)
    index =  np.argmin(np.linalg.norm(test_weight - weights, axis=1))
    if(index>=0 and index <=44):
        x='La colere' 
    if(index>=45 and index<=89):
        x='Neutre'
    if(index>=90 and index<=134):
        x='Le degout'  
    if(index>=135 and index<=179):
        x='La peur'  
    if(index>=180 and index<=224):
        x='La joie'  
    if(index>=225 and index<=269):
        x='La tristesse'
    if(index>=270 and index<=315):
        x='Le surpris'
    label1 = Label(frame1, text=x ,bg="#d6eaf2" ,font='Helvetica 66 bold' )
    label1.pack(pady=32) 

root = tk.Tk()
root.title("Reconnaissance D'Emotion En Utilisant l'ACP")

canvas = tk.Canvas(root, height=700, width=1000, bg="#20576e")  
canvas.pack(expand=YES, fill=BOTH)  # to make canvas effective

frame = tk.Frame(canvas, bg="#d6eaf2")
frame.place(relwidth=0.3, relheight=0.4, relx=0.05, rely=0.25)
frame.config(highlightthickness=10)

frame1 = LabelFrame(canvas,text="L'Emotion Correspondante",
font='times 25 bold', bg="#d6eaf2")
frame1.place(relwidth=0.5, relheight=0.4,relx=0.45, rely=0.25)
frame1.config(highlightthickness=10,highlightcolor='blue')

choisirImg = ttk.Button(root,command=ChoisirImage)
choisirImg.pack(fill=X)
photo = tk.PhotoImage(file="Choisir1.png", master=root)
choisirImg.config(image=photo )

ACPbutton= tk.Button(root,command=AppliquerAcp)
ACPbutton.pack(fill=X)
photo1 = tk.PhotoImage(file="acp1.png", master=root)
ACPbutton.config(image=photo1)

ImageMoulay= tk.Button(canvas) 
ImageMoulay.pack()
photo2 = tk.PhotoImage(file="index.png", master=root)
ImageMoulay.place(x=250, y=0)
ImageMoulay.config(image=photo2)

Created= tk.Button(canvas,borderwidth=0)
Created.pack()
photo3 = tk.PhotoImage(file="par1.png", master=root)
Created.place(x=0, y=551)
Created.config(image=photo3)

root.mainloop()

