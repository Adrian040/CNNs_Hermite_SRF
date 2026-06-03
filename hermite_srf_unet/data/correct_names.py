import os
import re
import shutil

# 📂 Ruta de la carpeta de entrada (cámbiala según necesites)
f_name = "val_imgs_unlabeled"
input_folder = "all_data_original/"+f_name
output_folder =  "all_data/"+f_name

# Crear carpeta de salida si no existe
os.makedirs(output_folder, exist_ok=True)

# Expresión regular para extraer el número
pattern = re.compile(r"(\d+)\.tif$")

for filename in os.listdir(input_folder):
    if filename.endswith(".tif"):
        match = pattern.search(filename)
        if match:
            number = match.group(1)
            new_name = f"{number}.tif"
            src = os.path.join(input_folder, filename)
            dst = os.path.join(output_folder, new_name)
            shutil.copy2(src, dst)  # copia manteniendo metadata
            print(f"{filename} → {new_name}")
