# Hermite U-Net para segmentación pulmonar multiclase

Este directorio contiene una versión experimental en scripts monolíticos para comparar una **U-Net convencional** contra una **U-Net con capas Hermite/SRF** en una tarea de segmentación multiclase. La implementación trabaja con imágenes monocanal, máscaras codificadas por clase y métricas de segmentación como Dice, IoU, precisión y recall.

La idea general del experimento es comparar dos modelos bajo un flujo de entrenamiento equivalente:

```text
imagen -> preprocesamiento por percentiles -> U-Net / Hermite-SRF U-Net -> máscara multiclase
```

En la variante Hermite/SRF, algunas convoluciones del encoder se reemplazan por capas basadas en un banco fijo de filtros Hermite/derivadas gaussianas, seguido de una mezcla aprendible `1x1`.

## Estructura

```text
hermite_unet/
└── Codigos_finales/
    ├── segmentacion_unet.py                         # entrenamiento/evaluación de U-Net convencional
    ├── segmentacion_hermite.py                      # entrenamiento/evaluación de U-Net con Hermite/SRF
    ├── calcular_matriz_confusion_normal.py          # matriz de confusión para U-Net convencional
    ├── calcular_matriz_confusion_hermite.py         # matriz de confusión para Hermite/SRF U-Net
    ├── metricas.py                                  # comparación gráfica entre curvas de ambos modelos
    ├── segmentacion_unet_resultados/                # resultados guardados para U-Net convencional
    └── segmentacion_hermite_preprocesada_dice_resultados_4/
        ├── checkpoints/                             # pesos entrenados
        ├── logs/                                    # métricas CSV
        ├── graficas/                                # curvas de entrenamiento
        └── visualizaciones/                         # predicciones visuales por época
```

## Instalación

Desde la raíz del repositorio:

```bash
git clone https://github.com/Adrian040/CNNs_Hermite_SRF.git
cd CNNs_Hermite_SRF/hermite_unet/Codigos_finales
```

Se recomienda crear un ambiente virtual:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

Instala las dependencias principales:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib pillow tqdm seaborn
```

Si tienes GPU NVIDIA, instala la versión de PyTorch compatible con tu versión de CUDA siguiendo las instrucciones oficiales de PyTorch.

## Preparar datos

Los scripts esperan que las imágenes y máscaras estén separadas en `train`, `val` y `test`. La estructura recomendada es:

```text
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

Cada máscara debe tener el mismo nombre base que su imagen correspondiente. Por ejemplo:

```text
data/train/images/0001.png
data/train/masks/0001.png
```

Formatos soportados por los scripts:

```text
.png, .jpg, .jpeg, .tif, .tiff, .bmp
```

Las máscaras pueden estar ya codificadas como clases enteras:

```text
0, 1, 2, 3
```

También se contempla el caso de máscaras con valores tipo:

```text
0, 85, 170, 255
```

En ese caso, el script remapea automáticamente esos valores a clases consecutivas `0, 1, 2, 3`, siempre que el número de valores únicos no exceda `NUM_CLASSES`.

### División desde `all_data`

Este directorio no incluye un script propio para dividir datos. Si estás usando la estructura del directorio hermano `hermite_srf_unet` (que se encuentra en una carpeta arriba en este repositorio, de otro experimento), puedes preparar la división desde ahí:

```bash
cd ../../hermite_srf_unet
python scripts/prepare_data.py --train 0.70 --val 0.15 --test 0.15 --seed 42 --convert-to-png --overwrite
cd ../hermite_unet/Codigos_finales
```

Esto genera una estructura como:

```text
hermite_srf_unet/data/train/images/
hermite_srf_unet/data/train/masks/
hermite_srf_unet/data/val/images/
hermite_srf_unet/data/val/masks/
hermite_srf_unet/data/test/images/
hermite_srf_unet/data/test/masks/
```

## Ajustar rutas antes de correr

Los scripts actuales usan rutas absolutas de Windows. Antes de ejecutar, abre cada archivo y modifica las constantes de la sección `RUTAS`.

Por ejemplo, en `segmentacion_unet.py` y `segmentacion_hermite.py` cambia:

```python
TRAIN_IMAGES = r"C:\...\data\train\images"
TRAIN_MASKS  = r"C:\...\data\train\masks"
VAL_IMAGES   = r"C:\...\data\val\images"
VAL_MASKS    = r"C:\...\data\val\masks"
TEST_IMAGES  = r"C:\...\data\test\images"
TEST_MASKS   = r"C:\...\data\test\masks"
```

por rutas locales. Una opción más portable es usar rutas relativas:

```python
from pathlib import Path

DATA_ROOT = Path("../../hermite_srf_unet/data")

TRAIN_IMAGES = DATA_ROOT / "train" / "images"
TRAIN_MASKS  = DATA_ROOT / "train" / "masks"
VAL_IMAGES   = DATA_ROOT / "val" / "images"
VAL_MASKS    = DATA_ROOT / "val" / "masks"
TEST_IMAGES  = DATA_ROOT / "test" / "images"
TEST_MASKS   = DATA_ROOT / "test" / "masks"
```

También ajusta `OUTPUT_DIR`. Por ejemplo:

```python
OUTPUT_DIR = Path("segmentacion_unet_resultados")
```

o, para la variante Hermite/SRF:

```python
OUTPUT_DIR = Path("segmentacion_hermite_preprocesada_dice_resultados_4")
```

## Configuración experimental principal

Los hiperparámetros principales se definen directamente dentro de cada script.

### U-Net convencional

Archivo:

```text
segmentacion_unet.py
```

Configuración base:

```python
NUM_CLASSES = 4
IMAGE_SIZE = 512
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 24
DEPTH = 4
DROPOUT = 0.05
```

### Hermite/SRF U-Net

Archivo:

```text
segmentacion_hermite.py
```

Configuración base:

```python
NUM_CLASSES = 4
IMAGE_SIZE = 512
EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 7e-4
WEIGHT_DECAY = 1e-4
BASE_CHANNELS = 24
DEPTH = 4
DROPOUT = 0.05
```

Configuración Hermite/SRF:

```python
HERMITE_KERNEL_SIZE = 7
HERMITE_MAX_ORDER = 5
HERMITE_SCALES = (1.5,)
SRF_STAGES = [0, 1, 2]
```

Esto aplica capas Hermite/SRF en las primeras tres etapas del encoder.

## Preprocesamiento

Ambos scripts usan el mismo preprocesamiento general:

```text
1. Lectura de imagen como monocanal.
2. Si la imagen viene RGB, se promedia a un solo canal.
3. Recorte de intensidades por percentiles.
4. Escalamiento a [0, 1].
5. Redimensionamiento a 512 x 512.
6. Máscaras redimensionadas con interpolación nearest-neighbor.
```

Parámetros de preprocesamiento:

```python
PERCENTILE_LOW = 1.0
PERCENTILE_HIGH = 99.0
USE_ZSCORE_AFTER_PERCENTILE = False
```

## Data augmentation

Durante entrenamiento se aplican augmentations sincronizadas entre imagen y máscara:

```python
AUG_HORIZONTAL_FLIP_P = 0.5
AUG_VERTICAL_FLIP_P = 0.10
AUG_ROTATE_P = 0.55
AUG_ROTATE_DEGREES = 12
AUG_AFFINE_P = 0.45
AUG_TRANSLATE_PIXELS = 10
AUG_SCALE_MIN = 0.90
AUG_SCALE_MAX = 1.10
AUG_SHEAR_DEGREES = 4
AUG_GAMMA_P = 0.25
AUG_GAMMA_RANGE = (0.90, 1.10)
AUG_BLUR_P = 0.10
AUG_NOISE_P = 0.20
AUG_NOISE_STD = 0.015
```

## Función de pérdida

La pérdida combina Cross Entropy ponderada y Dice Loss:

```text
LOSS = CE_WEIGHT * CrossEntropy ponderada + DICE_WEIGHT * Dice Loss
```

Configuración usada:

```python
CE_WEIGHT = 0.35
DICE_WEIGHT = 0.65
BACKGROUND_WEIGHT = 0.20
MAX_CLASS_WEIGHT = 10.0
```

El fondo se pondera menos para reducir su dominancia frente a clases pequeñas.

## Entrenar U-Net convencional

Desde `hermite_unet/Codigos_finales`:

```bash
python segmentacion_unet.py
```

Salidas esperadas:

```text
segmentacion_unet_resultados/
├── checkpoints/
│   ├── model_best_dice.pth
│   └── model_final.pth
├── logs/
│   ├── history.csv
│   └── test_metrics.csv
├── graficas/
└── visualizaciones/
```

## Entrenar Hermite/SRF U-Net

Desde `hermite_unet/Codigos_finales`:

```bash
python segmentacion_hermite.py
```

Salidas esperadas:

```text
segmentacion_hermite_preprocesada_dice_resultados_4/
├── checkpoints/
│   ├── model_best_dice.pth
│   └── model_final.pth
├── logs/
│   ├── history.csv
│   └── test_metrics.csv
├── graficas/
│   ├── loss_curve.png
│   ├── dice_curve.png
│   ├── iou_curve.png
│   ├── dice_clase_1.png
│   ├── dice_clase_2.png
│   └── dice_clase_3.png
└── visualizaciones/
    ├── epoch_000_before_training/
    ├── epoch_001/
    ├── epoch_020/
    ├── epoch_040/
    ├── epoch_060/
    ├── epoch_080/
    ├── epoch_100/
    └── test_best_model/
```

## Evaluación final

Los scripts de entrenamiento evalúan el mejor modelo y guardan métricas en:

```text
logs/test_metrics.csv
```

Las métricas principales son:

```text
Dice, IoU, Precision, Recall, Loss
```

En la variante Hermite/SRF también se guardan métricas por clase:

```text
dice_c1, iou_c1, precision_c1, recall_c1
dice_c2, iou_c2, precision_c2, recall_c2
dice_c3, iou_c3, precision_c3, recall_c3
```

## Resultados obtenidos

Los resultados ya guardados en esta carpeta reportan:

| Modelo | Dice | IoU | Precision | Recall | Loss |
|---|---:|---:|---:|---:|---:|
| U-Net convencional | 0.3113 | 0.1980 | 0.2620 | 0.4195 | 0.7333 |
| Hermite/SRF U-Net | 0.5192 | 0.3715 | 0.5563 | 0.5244 | 0.5568 |

Métricas por clase para Hermite/SRF U-Net:

| Clase | Dice | IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Clase 1 | 0.5747 | 0.4038 | 0.5405 | 0.6175 |
| Clase 2 | 0.6589 | 0.5075 | 0.6104 | 0.7201 |
| Clase 3 | 0.3239 | 0.2030 | 0.5179 | 0.2356 |

Estos valores pueden variar ligeramente al reentrenar debido a inicialización, GPU, orden de carga de datos y operaciones no completamente deterministas.

## Matriz de confusión

Para generar la matriz de confusión de la U-Net convencional:

```bash
python calcular_matriz_confusion_normal.py
```

Antes de correrlo, ajusta estas variables dentro del script:

```python
CHECKPOINT_PATH = r"ruta/al/model_best_dice.pth"
EVAL_SPLIT = "test"  # o "val"
OUTPUT_DIR = Path("matriz_confusion_unet")
```

Para generar la matriz de confusión de Hermite/SRF U-Net:

```bash
python calcular_matriz_confusion_hermite.py
```

Ajusta igualmente:

```python
CHECKPOINT_PATH = r"ruta/al/model_best_dice.pth"
EVAL_SPLIT = "test"  # o "val"
OUTPUT_DIR = Path("matriz_confusion_hermite")
```

Salidas esperadas:

```text
confusion_matrix_raw.csv
confusion_matrix_normalized_by_gt.csv
confusion_matrix_raw.png
confusion_matrix_normalized.png
metrics_from_confusion.csv
```

Dependiendo del script, algunos nombres pueden incluir el split, por ejemplo:

```text
confusion_matrix_raw_test.csv
confusion_matrix_normalized_by_gt_test.csv
metrics_from_confusion_test.csv
```

## Comparar curvas de ambos modelos

El archivo:

```text
metricas.py
```

genera gráficas comparativas usando los `history.csv` de ambos entrenamientos.

Antes de ejecutarlo, actualiza las rutas:

```python
hist_unet = pd.read_csv("segmentacion_unet_resultados/logs/history.csv")
hist_hermite = pd.read_csv("segmentacion_hermite_preprocesada_dice_resultados_4/logs/history.csv")
```

Luego ejecuta:

```bash
python metricas.py
```

## Reproducir el experimento completo

Flujo recomendado:

```bash
# 1. Entrar a la carpeta de scripts
cd CNNs_Hermite_SRF/hermite_unet/Codigos_finales

# 2. Editar rutas en segmentacion_unet.py y segmentacion_hermite.py
#    TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS, TEST_IMAGES, TEST_MASKS, OUTPUT_DIR

# 3. Entrenar baseline U-Net
python segmentacion_unet.py

# 4. Entrenar Hermite/SRF U-Net
python segmentacion_hermite.py

# 5. Editar CHECKPOINT_PATH en los scripts de matriz de confusión
python calcular_matriz_confusion_normal.py
python calcular_matriz_confusion_hermite.py

# 6. Editar rutas en metricas.py y comparar curvas
python metricas.py
```

## Notas importantes

- Los scripts actuales son autocontenidos y no usan archivos `.yaml` ni argumentos por consola.
- Para cambiar hiperparámetros se deben editar directamente las constantes al inicio de cada script.
- Si aparece error de memoria en GPU, reduce `BATCH_SIZE` o `IMAGE_SIZE`.
- Si las máscaras tienen más valores únicos que `NUM_CLASSES`, revisa si están en RGB o si necesitan conversión previa a escala de grises/clases enteras.
- Para experimentos más limpios y extensibles, la carpeta `hermite_srf_unet` contiene una versión más modular con `configs`, `scripts` y `src`.

## Autores

Proyecto final de Aprendizaje de Máquina para Visión Computacional.

- Oscar Eduardo Morales Toledo
- Katya Verónica Fuentes Sánchez
- Adrián Jesús Maldonado Oclica
