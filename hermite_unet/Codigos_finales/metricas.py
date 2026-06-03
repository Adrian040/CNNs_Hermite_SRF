# PROYECTO FINAL - APRENDIZAJE DE MÁQUINA PARA VC
# Comparación de mpetricas entre U-Net y U-Net con Hermite

# Autores:
 # Oscar Eduardo Morales Toledo
 # Katya Verónica Fuentes Sánchez
 # Adrián Jesús Maldonado Oclica

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Cargar csv de métricas
# ============================================================

hist_unet = pd.read_csv(r"C:\Users\katya\OneDrive\Documentos\PCIC\2semestre\A.MAQUINA\Proyecto final\segmentacion_unet_resultados\logs\history.csv")
hist_hermite = pd.read_csv(r"C:\Users\katya\OneDrive\Documentos\PCIC\2semestre\A.MAQUINA\Proyecto final\segmentacion_hermite_preprocesada_dice_resultados_3\logs\history.csv")

# ============================================================
# Crear DataFrame para Seaborn
# ============================================================

df_dice = pd.DataFrame({
    "epoch": pd.concat([
        hist_unet["epoch"],
        hist_unet["epoch"],
        hist_hermite["epoch"],
        hist_hermite["epoch"]
    ], ignore_index=True),

    "valor": pd.concat([
        hist_unet["train_dice"],
        hist_unet["val_dice"],
        hist_hermite["train_dice"],
        hist_hermite["val_dice"]
    ], ignore_index=True),

    "curva": (
        ["U-Net Train"] * len(hist_unet) +
        ["U-Net Val"] * len(hist_unet) +
        ["U-Net Hermite Train"] * len(hist_hermite) +
        ["U-Net Hermite Val"] * len(hist_hermite)
    )
})

df_loss = pd.DataFrame({
    "epoch": pd.concat([
        hist_unet["epoch"],
        hist_unet["epoch"],
        hist_hermite["epoch"],
        hist_hermite["epoch"]
    ], ignore_index=True),

    "valor": pd.concat([
        hist_unet["train_loss"],
        hist_unet["val_loss"],
        hist_hermite["train_loss"],
        hist_hermite["val_loss"]
    ], ignore_index=True),

    "curva": (
        ["U-Net Train"] * len(hist_unet) +
        ["U-Net Val"] * len(hist_unet) +
        ["U-Net Hermite Train"] * len(hist_hermite) +
        ["U-Net Hermite Val"] * len(hist_hermite)
    )
})

# ============================================================
# Configuración gráfica y paletas de colores
# ============================================================

sns.set_theme(style="whitegrid", context="talk")

palette = {
    "U-Net Train": "#1f77b4",
    "U-Net Val": "#6baed6",
    "U-Net Hermite Train": "#d95f02",
    "U-Net Hermite Val": "#fdae6b"
}

linestyles = {
    "U-Net Train": "-",
    "U-Net Val": "--",
    "U-Net Hermite Train": "-",
    "U-Net Hermite Val": "--"
}

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# ============================================================
# Gráfica Dice
# ============================================================

for curva in df_dice["curva"].unique():
    data = df_dice[df_dice["curva"] == curva]

    sns.lineplot(
        data=data,
        x="epoch",
        y="valor",
        ax=axes[0],
        label=curva,
        color=palette[curva],
        linestyle=linestyles[curva],
        linewidth=2.5
    )

#axes[0].set_title("Dice durante el entrenamiento")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("Dice")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=15, title_fontsize=11)

# ===========================================================
# GráficaLoss
# ============================================================

for curva in df_loss["curva"].unique():
    data = df_loss[df_loss["curva"] == curva]

    sns.lineplot(
        data=data,
        x="epoch",
        y="valor",
        ax=axes[1],
        label=curva,
        color=palette[curva],
        linestyle=linestyles[curva],
        linewidth=2.5
    )

#axes[1].set_title("Loss durante el entrenamiento")
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=15, title_fontsize=11)


plt.tight_layout()

plt.savefig(
    "comparacion_unet_vs_hermite_seaborn_leyenda_clara.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()