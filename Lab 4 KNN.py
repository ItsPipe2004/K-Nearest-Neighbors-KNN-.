import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageTk

# =====================================================
# CARGAR BASE DE DATOS
# =====================================================

df = pd.read_csv("sensores_dataset_800.csv")

FEATURES = ["distancia_us","reflectancia_ir","temperatura_c"]

X = df[FEATURES].values
y = df["clase"].values

# =====================================================
# ENTRENAMIENTO
# =====================================================

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.25,random_state=42
)

scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# =====================================================
# MÉTRICAS DE DISTANCIA
# =====================================================

def euclidiana(a,b):
    return np.sqrt(np.sum((a-b)**2))

def manhattan(a,b):
    return np.sum(np.abs(a-b))

def minkowski(a,b,p=3):
    return np.sum(np.abs(a-b)**p)**(1/p)

VI = np.linalg.pinv(np.cov(X_train_s.T))

def mahalanobis(a,b):
    diff = a-b
    return np.sqrt(diff @ VI @ diff)

# =====================================================
# ALGORITMO KNN
# =====================================================

def knn_predict(X_tr,y_tr,X_te,dist_fn,k):

    preds = []

    for punto in X_te:

        distancias = [dist_fn(punto,xt) for xt in X_tr]

        idx = np.argsort(distancias)[:k]

        votos = Counter(y_tr[idx])

        preds.append(votos.most_common(1)[0][0])

    return np.array(preds)

# =====================================================
# CARGAR IMÁGENES
# =====================================================

imagenes = {
    "Metal frio":"metal.png",
    "Plástico caliente":"plastico.png",
    "Madera":"madera.png",
    "Bateria":"bateria.png"
}

def mostrar_imagen(clase):

    img = Image.open(imagenes[clase])
    img = img.resize((150,150))

    img_tk = ImageTk.PhotoImage(img)

    panel_imagen.config(image=img_tk)
    panel_imagen.image = img_tk

# =====================================================
# FUNCIÓN DE CLASIFICACIÓN
# =====================================================

def clasificar():

    try:

        dist = float(entry_dist.get())
        ir = float(entry_ir.get())
        temp = float(entry_temp.get())

        k = int(combo_k.get())
        metrica = combo_metric.get()

        nuevo = np.array([[dist,ir,temp]])
        nuevo_s = scaler.transform(nuevo)

        if metrica == "Euclidiana":
            dist_fn = euclidiana

        elif metrica == "Manhattan":
            dist_fn = manhattan

        elif metrica == "Minkowski":
            dist_fn = lambda a,b: minkowski(a,b,3)

        elif metrica == "Mahalanobis":
            dist_fn = mahalanobis

        pred = knn_predict(X_train_s,y_train,nuevo_s,dist_fn,k)[0]

        if pred.lower() == "bateria":
            estado = "RECOGIDO"
        else:
            estado = "NO RECOGIDO"

        resultado_label.config(
            text=f"Objeto detectado: {pred}\nEstado: {estado}",
            fg="green"
        )

        mostrar_imagen(pred)

    except ValueError:
        messagebox.showerror("Error","Ingrese valores numéricos válidos")

# =====================================================
# INTERFAZ
# =====================================================

root = tk.Tk()

root.title("Clasificador de Material - Robot Recolector")
root.geometry("500x520")
root.configure(bg="#F4F6F8")

titulo = tk.Label(
    root,
    text="Clasificación de Material\nRobot Recolector",
    font=("Arial",16,"bold"),
    bg="#F4F6F8"
)

titulo.pack(pady=10)

frame = tk.Frame(root,bg="#F4F6F8")
frame.pack(pady=10)

# ENTRADAS

tk.Label(frame,text="Distancia US",bg="#F4F6F8").grid(row=0,column=0,pady=5)
entry_dist = ttk.Entry(frame)
entry_dist.grid(row=0,column=1)

tk.Label(frame,text="Reflectancia IR",bg="#F4F6F8").grid(row=1,column=0,pady=5)
entry_ir = ttk.Entry(frame)
entry_ir.grid(row=1,column=1)

tk.Label(frame,text="Temperatura °C",bg="#F4F6F8").grid(row=2,column=0,pady=5)
entry_temp = ttk.Entry(frame)
entry_temp.grid(row=2,column=1)

# VECINOS

tk.Label(frame,text="Número de vecinos (k)",bg="#F4F6F8").grid(row=3,column=0,pady=5)

combo_k = ttk.Combobox(frame,values=[1,3,5,7],state="readonly")
combo_k.current(1)
combo_k.grid(row=3,column=1)

# MÉTRICA

tk.Label(frame,text="Métrica de distancia",bg="#F4F6F8").grid(row=4,column=0,pady=5)

combo_metric = ttk.Combobox(
    frame,
    values=["Euclidiana","Manhattan","Minkowski","Mahalanobis"],
    state="readonly"
)

combo_metric.current(0)
combo_metric.grid(row=4,column=1)

# BOTÓN

btn = ttk.Button(
    root,
    text="Clasificar objeto",
    command=clasificar
)

btn.pack(pady=15)

# RESULTADO

resultado_label = tk.Label(
    root,
    text="Ingrese datos de sensores",
    font=("Arial",12),
    bg="#F4F6F8"
)

resultado_label.pack(pady=10)

# IMAGEN

panel_imagen = tk.Label(root,bg="#F4F6F8")
panel_imagen.pack(pady=15)

root.mainloop()