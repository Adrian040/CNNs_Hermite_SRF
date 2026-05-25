# Hermite-SRF U-Net para segmentación pulmonar COVID

Proyecto base para segmentación binaria o multiclase con una U-Net cuyo encoder puede reemplazarse por bloques **Structured Receptive Field (SRF)** inspirados en Jacobsen et al. En estos bloques, cada convolución se implementa como:

```text
entrada -> banco fijo Hermite/Gaussian derivatives -> mezcla aprendible 1x1
```

Por linealidad, esto equivale a aprender kernels efectivos como combinaciones lineales de una base fija de derivadas gaussianas/Hermite, sin construir explícitamente el kernel en cada iteración.

## Estructura

```text
configs/default.yaml              # hiperparámetros
scripts/prepare_data.py           # división train/val/test
scripts/generate_basis.py         # banco de filtros Hermite
scripts/train.py                  # entrenamiento
scripts/test.py                   # evaluación final
scripts/predict_folder.py         # predicción de carpeta completa
scripts/visualize_predictions.py  # figuras imagen/GT/predicción/overlay
scripts/inspect_dataset.py        # revisión básica del dataset
src/models/hermite_basis.py       # generación del banco Hermite
src/models/srf_layers.py          # capa HermiteBasisConv2d tipo Jacobsen
src/models/unet_srf.py            # U-Net con encoder SRF
src/data/dataset.py               # dataset PyTorch
src/utils/metrics.py              # Dice, IoU, Precision, Recall, Hausdorff
```

## Instalación

```bash
pip install -r requirements.txt
```

## Preparar datos

Coloca tus datos así:

```text
data/all_data/images/
data/all_data/masks/
```

Cada máscara debe tener el mismo nombre o el mismo `stem` que su imagen correspondiente.

División ejemplo 70/15/15 preservando formatos originales:

```bash
python scripts/prepare_data.py --train 0.70 --val 0.15 --test 0.15 --seed 42 --overwrite
```

División convirtiendo todo a PNG:

```bash
python scripts/prepare_data.py --train 0.70 --val 0.15 --test 0.15 --seed 42 --convert-to-png --overwrite
```

## Generar banco Hermite

El entrenamiento lo genera automáticamente si no existe, pero puedes crearlo y visualizarlo manualmente:

```bash
python scripts/generate_basis.py --kernel-size 7 --max-order 3 --scales 1.0 2.0 --preview --overwrite
```

Esto crea:

```text
assets/hermite_basis/hermite_order3_k7_scales_1.0_2.0.pt
assets/hermite_basis/hermite_order3_k7_scales_1.0_2.0.json
assets/hermite_basis/hermite_order3_k7_scales_1.0_2.0.png
```

## Entrenar

```bash
python scripts/train.py --config configs/default.yaml
```

Salidas principales:

```text
outputs/exp01_srf_unet/checkpoints/model_best_dice.pth
outputs/exp01_srf_unet/checkpoints/model_final.pth
outputs/exp01_srf_unet/logs/history.csv
outputs/exp01_srf_unet/figures/training_curves.png
```

## Evaluar en test

```bash
python scripts/test.py \
  --config configs/default.yaml \
  --checkpoint outputs/exp01_srf_unet/checkpoints/model_best_dice.pth \
  --save-preds
```

Guarda:

```text
outputs/exp01_srf_unet/test_results/per_image_metrics.csv
outputs/exp01_srf_unet/test_results/summary_metrics.csv
outputs/exp01_srf_unet/test_results/predicted_masks/
```

## Visualizar predicciones

```bash
python scripts/visualize_predictions.py \
  --config configs/default.yaml \
  --checkpoint outputs/exp01_srf_unet/checkpoints/model_best_dice.pth \
  --split test \
  --num-samples 12
```

## Predecir carpeta externa

```bash
python scripts/predict_folder.py \
  --config configs/default.yaml \
  --checkpoint outputs/exp01_srf_unet/checkpoints/model_best_dice.pth \
  --input-dir data/test/images \
  --output-dir predicted_images
```

## Modos binario y multiclase

En `configs/default.yaml`:

```yaml
segmentation_mode: binary
num_classes: 2
```

Para multiclase:

```yaml
segmentation_mode: multiclass
num_classes: 4
```

En multiclase, las máscaras deben estar codificadas como enteros `0, 1, ..., C-1` en escala de grises.

## Cambiar entre U-Net normal y Hermite-SRF

En el YAML:

```yaml
model:
  encoder_block: srf   # srf | conv
  decoder_block: conv  # conv | srf
```

Para baseline U-Net normal:

```yaml
model:
  encoder_block: conv
  decoder_block: conv
```

Para versión SRF en encoder:

```yaml
model:
  encoder_block: srf
  srf_stages: [0, 1, 2, 3]
```

## Nota metodológica

La implementación SRF sigue la idea de Jacobsen et al.: la red no aprende directamente cada peso espacial del kernel, sino pesos de combinación sobre una base fija de derivadas gaussianas/Hermite. Esto introduce un prior local de bordes, líneas y curvaturas, útil para escenarios médicos con pocos datos.

## Configuraciones incluidas

- `configs/baseline_unet.yaml`: U-Net convencional.
- `configs/default.yaml`: versión práctica con SRF en los primeros niveles del encoder.
- `configs/full_srf_encoder.yaml`: versión más fiel/ambiciosa con SRF en todos los niveles del encoder.
- `configs/debug.yaml`: configuración pequeña para verificar instalación y flujo.

Recomendación experimental inicial:

```bash
python scripts/train.py --config configs/baseline_unet.yaml
python scripts/train.py --config configs/default.yaml
```

Después, si el tiempo/GPU lo permite:

```bash
python scripts/train.py --config configs/full_srf_encoder.yaml
```

La versión completa es más cercana a reemplazar todo el encoder por bloques SRF, pero también es más costosa. Para un avance de tesis/materia suele ser más seguro reportar primero baseline vs SRF práctico y dejar `full_srf_encoder` como experimento adicional.
