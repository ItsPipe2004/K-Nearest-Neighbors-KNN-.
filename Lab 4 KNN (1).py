"""
╔══════════════════════════════════════════════════════════════╗
║         ANÁLISIS COMPLETO k-NN — SENSORES ROBÓTICOS          ║
║  Dataset  : 200 muestras × 4 clases = 800 total             ║
║  Sensores : distancia_us (cm) | reflectancia_ir (ADC 0-255) ║
║             | temperatura_c (°C)                            ║
║  Clases   : Metal frío | Plástico caliente                  ║
║             | Madera   | Bateria                       ║
║  SALIDAS                                                     
║   sensores_dataset_800.csv   → Dataset completo             ║
║   knn_resultados.csv         → Tabla de métricas            ║
║   01_analisis_eda.png        → 6 gráficos exploratorios     ║
║   02_analisis_knn.png        → Exactitud vs k por métrica   ║
║   03_matrices_confusion.png  → 5 matrices de confusión      ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import time
import warnings
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")
np.random.seed(42)

# ══════════════════════════════════════════════════════════════
# SECCIÓN 1 — GENERACIÓN DE DATASET
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  [1/3]  GENERANDO DATASET")
print("=" * 62)

N = 200

clases_cfg = {
    "Metal frío": {
        "color": "#4A90D9",
        "dist_m": 15,  "dist_s": 3.5,
        "ir_m":  208,  "ir_s":  18,
        "temp_m": 19,  "temp_s": 4,
    },
    "Plástico caliente": {
        "color": "#E84040",
        "dist_m": 22,  "dist_s": 4.5,
        "ir_m":  135,  "ir_s":  24,
        "temp_m": 63,  "temp_s": 9,
    },
    "Madera": {
        "color": "#8B5E3C",
        "dist_m": 33,  "dist_s": 5.5,
        "ir_m":   88,  "ir_s":  22,
        "temp_m": 26,  "temp_s": 5,
    },
    "Bateria": {
        "color": "#F5A623",
        "dist_m": 11,  "dist_s": 3.5,
        "ir_m":  162,  "ir_s":  18,
        "temp_m": 36,  "temp_s": 3,
    },
}

registros = []
for nombre, p in clases_cfg.items():
    dist = np.clip(np.random.normal(p["dist_m"], p["dist_s"], N), 2,   400)
    ir   = np.clip(np.random.normal(p["ir_m"],   p["ir_s"],   N), 0,   255).astype(int)
    temp = np.clip(np.random.normal(p["temp_m"], p["temp_s"], N), -5,  120)
    for i in range(N):
        registros.append({
            "distancia_us":    round(dist[i], 2),
            "reflectancia_ir": ir[i],
            "temperatura_c":   round(temp[i], 2),
            "clase":           nombre,
        })

df = pd.DataFrame(registros).sample(frac=1, random_state=42).reset_index(drop=True)

# Ruido del 5% para variación realista de accuracy
otras = list(clases_cfg.keys())
noise_idx = df.sample(frac=0.05, random_state=7).index
for idx in noise_idx:
    actual   = df.at[idx, "clase"]
    opciones = [c for c in otras if c != actual]
    df.at[idx, "clase"] = np.random.choice(opciones)

df.to_csv("sensores_dataset_800.csv", index=False)

CLASES   = list(clases_cfg.keys())
COLORES  = {k: v["color"] for k, v in clases_cfg.items()}
FEATURES = ["distancia_us", "reflectancia_ir", "temperatura_c"]
patches_leg = [mpatches.Patch(color=COLORES[c], label=c) for c in CLASES]

for cl in CLASES:
    g = df[df["clase"] == cl]
    print(f"  {cl:20s}: {len(g):3d} muestras  "
          f"dist={g.distancia_us.mean():.1f}  "
          f"ir={g.reflectancia_ir.mean():.0f}  "
          f"temp={g.temperatura_c.mean():.1f} °C")
print(f"\n  Total: {len(df)} muestras\n")

# ── Preparación train/test (compartida entre análisis 2 y 3) ─
X = df[FEATURES].values
y = df["clase"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Funciones de distancia ────────────────────────────────────
def euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def manhattan(a, b):
    return np.sum(np.abs(a - b))

def minkowski(a, b, p=3):
    return np.sum(np.abs(a - b) ** p) ** (1.0 / p)

VI = np.linalg.pinv(np.cov(X_train_s.T))
def mahalanobis(a, b):
    diff = a - b
    return np.sqrt(np.maximum(diff @ VI @ diff, 0))

METRICAS = {
    "Euclidiana":     euclidiana,
    "Manhattan":      manhattan,
    "Minkowski":      lambda a, b: minkowski(a, b, p=3),
    "Mahalanobis":    mahalanobis,
}

def knn_predict(X_tr, y_tr, X_te, dist_fn, k):
    preds = []
    for punto in X_te:
        dists = [dist_fn(punto, xt) for xt in X_tr]
        idx_k = np.argsort(dists)[:k]
        votos = Counter(y_tr[idx_k])
        preds.append(votos.most_common(1)[0][0])
    return np.array(preds)

# ══════════════════════════════════════════════════════════════
# SECCIÓN 2 — ANÁLISIS EXPLORATORIO (6 GRÁFICOS EDA)
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  [2/3]  GENERANDO ANÁLISIS EDA (01_analisis_eda.png)")
print("=" * 62)

fig1 = plt.figure(figsize=(20, 12))
fig1.patch.set_facecolor("#F4F6F8")
fig1.suptitle(
    "Análisis Exploratorio — Sensores k-NN\n"
    "Metal frío | Plástico caliente | Madera | Bateria (800 muestras)",
    fontsize=13, fontweight="bold", y=0.99
)

# ── 2.1 Matriz de Correlación ─────────────────────────────────
ax1 = fig1.add_subplot(2, 3, 1)
corr = df[FEATURES].corr()
im1  = ax1.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
ax1.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
ax1.set_yticklabels(FEATURES, fontsize=8)
for i in range(3):
    for j in range(3):
        val = corr.iloc[i, j]
        ax1.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=10, fontweight="bold",
                 color="white" if abs(val) > 0.5 else "black")
ax1.set_title("Matriz de Correlación entre Sensores", fontsize=10, fontweight="bold", pad=8)
ax1.set_facecolor("#EAEEF2")

# ── 2.2 Espacio 3D de Características ────────────────────────
ax2 = fig1.add_subplot(2, 3, 2, projection="3d")
ax2.set_facecolor("#EAEEF2")
for cl in CLASES:
    sub = df[df["clase"] == cl]
    ax2.scatter(sub["distancia_us"], sub["reflectancia_ir"], sub["temperatura_c"],
                c=COLORES[cl], s=15, alpha=0.65, edgecolors="none", label=cl)
ax2.set_xlabel("Distancia US (cm)", fontsize=7, labelpad=3)
ax2.set_ylabel("Reflectancia IR",   fontsize=7, labelpad=3)
ax2.set_zlabel("Temperatura (°C)",  fontsize=7, labelpad=3)
ax2.set_title("Espacio de Características 3D", fontsize=10, fontweight="bold", pad=8)
ax2.legend(fontsize=7, loc="upper left", framealpha=0.7)
ax2.tick_params(labelsize=6)

# ── 2.3 Coordenadas Paralelas ─────────────────────────────────
ax3 = fig1.add_subplot(2, 3, 3)
ax3.set_facecolor("#EAEEF2")
df_norm = df.copy()
df_norm[FEATURES] = MinMaxScaler().fit_transform(df[FEATURES]) * 250
for cl in CLASES:
    sub = df_norm[df_norm["clase"] == cl].sample(50, random_state=42)
    for _, row in sub.iterrows():
        ax3.plot(range(3), [row[f] for f in FEATURES],
                 color=COLORES[cl], alpha=0.35, linewidth=0.9)
ax3.set_xticks(range(3))
ax3.set_xticklabels(FEATURES, fontsize=8)
ax3.set_ylabel("Valor (normalizado 0–250)", fontsize=8)
ax3.set_title("Coordenadas Paralelas", fontsize=10, fontweight="bold", pad=8)
ax3.legend(handles=patches_leg, fontsize=7, loc="upper right", framealpha=0.8)
ax3.set_xlim(-0.05, 2.05); ax3.set_ylim(-10, 270)
for xv in range(3):
    ax3.axvline(xv, color="gray", linewidth=1.0, alpha=0.6)

# ── 2.4 Distribución de Temperatura por Clase ─────────────────
ax4 = fig1.add_subplot(2, 3, 4)
ax4.set_facecolor("#EAEEF2")
for cl in CLASES:
    ax4.hist(df[df["clase"] == cl]["temperatura_c"], bins=25,
             alpha=0.55, color=COLORES[cl], label=cl,
             edgecolor="white", linewidth=0.3)
ax4.set_xlabel("Temperatura (°C)", fontsize=9)
ax4.set_ylabel("Frecuencia",       fontsize=9)
ax4.set_title("Distribución de Temperatura por Clase", fontsize=10, fontweight="bold", pad=8)
ax4.legend(handles=patches_leg, fontsize=8, framealpha=0.8)
ax4.grid(True, alpha=0.3)

# ── 2.5 Boxplots de Sensores ──────────────────────────────────
ax5 = fig1.add_subplot(2, 3, 5)
ax5.set_facecolor("#EAEEF2")
feat_box   = ["distancia_us", "reflectancia_ir"]
box_labels = ["Distancia US", "Reflectancia IR"]
n_cls   = len(CLASES)
width   = 0.18
offsets = np.linspace(-(n_cls - 1) * width / 2, (n_cls - 1) * width / 2, n_cls)
for ci, cl in enumerate(CLASES):
    sub = df[df["clase"] == cl]
    positions = [v + offsets[ci] for v in range(2)]
    ax5.boxplot([sub[f].values for f in feat_box],
                positions=positions, widths=width * 0.85,
                patch_artist=True, showfliers=True,
                flierprops=dict(marker="o", markersize=3, alpha=0.4,
                                markerfacecolor=COLORES[cl],
                                markeredgecolor=COLORES[cl]),
                medianprops=dict(color="white", linewidth=2),
                whiskerprops=dict(color=COLORES[cl], linewidth=1.2),
                capprops=dict(color=COLORES[cl], linewidth=1.2),
                boxprops=dict(facecolor=COLORES[cl], alpha=0.75, linewidth=0))
ax5.set_xticks(range(2))
ax5.set_xticklabels(box_labels, fontsize=9)
ax5.set_ylabel("Valor", fontsize=9)
ax5.set_title("Boxplots de Sensores", fontsize=10, fontweight="bold", pad=8)
ax5.legend(handles=patches_leg, fontsize=8, loc="upper left", framealpha=0.8)
ax5.grid(True, alpha=0.3, axis="y")

# ── 2.6 Perfil Promedio (Heatmap) ─────────────────────────────
ax6 = fig1.add_subplot(2, 3, 6)
perfil = df.groupby("clase")[FEATURES].mean().reindex(CLASES)
im6    = ax6.imshow(perfil.values, cmap="YlOrRd", aspect="auto")
plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04, label="Valor medio")
ax6.set_xticks(range(3)); ax6.set_yticks(range(4))
ax6.set_xticklabels(FEATURES, rotation=45, ha="right", fontsize=8)
ax6.set_yticklabels(CLASES, fontsize=8)
thresh6 = (perfil.values.max() + perfil.values.min()) / 2
for i in range(4):
    for j in range(3):
        val = perfil.values[i, j]
        txt = f"{val:.0f}" if val < 100 else f"{val/100:.1g}e+02"
        ax6.text(j, i, txt, ha="center", va="center",
                 fontsize=9, fontweight="bold",
                 color="white" if val > thresh6 else "black")
ax6.set_title("Perfil Promedio de Sensores por Clase", fontsize=10, fontweight="bold", pad=8)
ax6.set_xlabel("nombre_clase", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("01_analisis_eda.png",
            dpi=150, bbox_inches="tight", facecolor=fig1.get_facecolor())
plt.close()
print("  ✓ 01_analisis_eda.png generado\n")

# ══════════════════════════════════════════════════════════════
# SECCIÓN 3 — EXACTITUD VS k POR MÉTRICA + TIEMPOS
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  [3/3a] EVALUANDO MÉTRICAS (02_analisis_knn.png)")
print("=" * 62)

K_VALUES  = [1, 3, 5, 7, 9]
resultados = {}
filas_csv  = []

for nombre, fn in METRICAS.items():
    accs, tiempos = [], []
    for k in K_VALUES:
        t0     = time.perf_counter()
        y_pred = knn_predict(X_train_s, y_train, X_test_s, fn, k=k)
        t1     = time.perf_counter()
        ms     = (t1 - t0) * 1000
        acc    = accuracy_score(y_test, y_pred)
        prec   = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec    = recall_score(y_test, y_pred,    average="macro", zero_division=0)
        f1     = f1_score(y_test, y_pred,        average="macro", zero_division=0)
        accs.append(acc); tiempos.append(ms)
        filas_csv.append({"Metrica": nombre, "k": k,
                          "Accuracy": round(acc, 4), "Precision": round(prec, 4),
                          "Recall":   round(rec, 4), "F1_Score":  round(f1, 4),
                          "Tiempo_ms": round(ms, 2)})
        print(f"  {nombre:14s}  k={k}  Acc={acc:.4f}  t={ms:.1f}ms")
    resultados[nombre] = {"acc": accs, "tiempo_ms": tiempos}

pd.DataFrame(filas_csv).to_csv("knn_resultados.csv", index=False)
print("  ✓ knn_resultados.csv guardado\n")

# ── Figura 2: 2×2 subplots ────────────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.patch.set_facecolor("#E8EAF0")
BG   = "#E8EAF0"
BLUE = "#1A3A9C"

for ax, (nombre, datos) in zip(axes2.flatten(), resultados.items()):
    accs    = datos["acc"]
    tiempos = datos["tiempo_ms"]

    ax.set_facecolor(BG)
    ax.plot(K_VALUES, accs, color=BLUE, linewidth=2.2,
            marker="o", markersize=7,
            markerfacecolor=BLUE, markeredgecolor=BLUE)

    acc_mid = (max(accs) + min(accs)) / 2
    acc_rng = max(accs) - min(accs) if max(accs) != min(accs) else 0.005
    for k, acc, t in zip(K_VALUES, accs, tiempos):
        dy = acc_rng * 0.35 if acc >= acc_mid else -acc_rng * 0.55
        ax.annotate(f"{t:.2f}ms",
                    xy=(k, acc), xytext=(k, acc + dy),
                    ha="center", va="center",
                    fontsize=8.5, color="#222222")

    ax.set_title(f"Métrica: {nombre}", fontsize=12, fontweight="bold",
                 color="#111111", pad=10)
    ax.set_xlabel("k (número de vecinos)", fontsize=9, color="#444444")
    ax.set_ylabel("Exactitud",             fontsize=9, color="#444444")
    ax.set_xticks(K_VALUES)
    ax.tick_params(labelsize=8.5)
    pad = acc_rng * 3 if acc_rng > 0 else 0.01
    ax.set_ylim(min(accs) - pad, max(accs) + pad)
    ax.grid(True, linestyle="-", linewidth=0.5, color="#C8CADE", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.tight_layout(pad=3.0)
plt.savefig("02_analisis_knn.png",
            dpi=150, bbox_inches="tight", facecolor=fig2.get_facecolor())
plt.close()
print("  ✓ 02_analisis_knn.png generado\n")

# ══════════════════════════════════════════════════════════════
# SECCIÓN 4 — MATRICES DE CONFUSIÓN (5 CONFIGURACIONES)
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  [3/3b] MATRICES DE CONFUSIÓN (03_matrices_confusion.png)")
print("=" * 62)

COMBINACIONES = [
    {"k": 3, "nombre": "Euclidiana",     "fn": euclidiana},
    {"k": 5, "nombre": "Euclidiana",     "fn": euclidiana},
    {"k": 3, "nombre": "Manhattan",      "fn": manhattan},
    {"k": 3, "nombre": "Mahalanobis",    "fn": mahalanobis},
    {"k": 7, "nombre": "Minkowski(p=3)", "fn": lambda a, b: minkowski(a, b, p=3)},
]

for cfg in COMBINACIONES:
    y_pred    = knn_predict(X_train_s, y_train, X_test_s, cfg["fn"], k=cfg["k"])
    cfg["acc"] = accuracy_score(y_test, y_pred)
    cfg["cm"]  = confusion_matrix(y_test, y_pred, labels=CLASES)
    print(f"  k={cfg['k']:2d}, {cfg['nombre']:15s}  Accuracy={cfg['acc']:.3f}")

fig3 = plt.figure(figsize=(20, 13))
fig3.patch.set_facecolor("#F0F2F8")

gs   = gridspec.GridSpec(2, 6, figure=fig3, hspace=0.45, wspace=0.35)
pos  = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
        gs[1, 1:3], gs[1, 3:5]]
etiq = ["M.frío", "Plást.cal.", "Madera", "Bateria"]

for ax_pos, cfg in zip(pos, COMBINACIONES):
    ax  = fig3.add_subplot(ax_pos)
    cm  = cfg["cm"]

    im = ax.imshow(cm, cmap="Blues", interpolation="nearest",
                   vmin=0, vmax=cm.max())
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    n = len(CLASES)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(etiq, fontsize=7.5, rotation=20, ha="right")
    ax.set_yticklabels(etiq, fontsize=7.5)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "#1A2E6C")

    ax.set_xlabel("Predicho", fontsize=8.5, labelpad=4)
    ax.set_ylabel("Real",     fontsize=8.5, labelpad=4)
    ax.set_title(f"k={cfg['k']}, {cfg['nombre']}\nAccuracy={cfg['acc']:.3f}",
                 fontsize=10, fontweight="bold", color="#111111", pad=8)
    ax.set_facecolor("#EEF0F8")
    for spine in ax.spines.values():
        spine.set_visible(False)

plt.savefig("03_matrices_confusion.png",
            dpi=150, bbox_inches="tight", facecolor=fig3.get_facecolor())
plt.close()
print("  ✓ 03_matrices_confusion.png generado\n")

# ══════════════════════════════════════════════════════════════
# RESUMEN FINAL
# ══════════════════════════════════════════════════════════════
print("=" * 62)
print("  RESUMEN — MEJOR k POR MÉTRICA")
print("=" * 62)
print(f"  {'Métrica':14s}  {'k':>3}  {'Acc':>7}  {'t/muestra':>10}")
print("  " + "-" * 42)
for nombre, datos in resultados.items():
    best_i = int(np.argmax(datos["acc"]))
    print(f"  {nombre:14s}  "
          f"k={K_VALUES[best_i]}  "
          f"{datos['acc'][best_i]:.4f}  "
          f"{datos['tiempo_ms'][best_i]:>8.1f} ms")

print("\n  Archivos generados en :")
for f in ["sensores_dataset_800.csv", "knn_resultados.csv",
          "01_analisis_eda.png", "02_analisis_knn.png",
          "03_matrices_confusion.png"]:
    print(f"    ✓ {f}")
print("=" * 62)

