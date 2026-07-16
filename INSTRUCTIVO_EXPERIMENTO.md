# Instructivo paso a paso — Pipeline completo MvTec AD

Convención usada en todo el instructivo (evita mezclar resultados de categorías/semillas):
cada corrida usa el sufijo `<categoria>_s<semilla>`, p. ej. `bottle_s42`.
Los flags de línea de comandos **siempre tienen prioridad** sobre el YAML, así que
no necesitas editar los YAML para cambiar categoría o semilla — solo para los
parámetros estructurales que se indican.

---

## FASE 0 — Preparar el entorno (una sola vez)

**En el PC:**
```
pip install pynvml scipy pandas matplotlib pyyaml pillow
```
`pynvml` es obligatorio para que la energía se mida por NVML (sin él, el
objetivo energía queda inoperativo y el script lo reportará como tal).

**En el Jetson Orin Nano:** JetPack 5.1.2 ya incluye TensorRT y tegrastats.
Instalar PyTorch para Jetson (wheel de NVIDIA), `pyyaml`, `pillow`, `opencv-python`.

---

## FASE 1 — Preparación de datos (PC, una sola vez, ~5–15 min)

No requiere configuración. El archivo `mvtec_anomaly_detection.tar.xz` debe estar
en la raíz del proyecto (ya está).

```
python main_prepare.py
```

Verificar al final: `results/dataset_summary.json` debe decir `"valid": true` y
contener `archive_sha256`. Esto genera los splits de las 15 categorías de una vez;
no se repite por categoría ni por semilla.

---

## FASE 2 — Búsqueda NSGA-II (PC, ~34 h por categoría/semilla)

**Configurar una sola vez** en `config/search.yaml`:
- `fitness:` → `energy_backend: nvml` (ya está así; en Jetson sería `auto`)
- `population_size: 40` y `n_generations: 50` (ya están)

**Ejecutar** (la categoría y semilla van por CLI, no editar el YAML):
```
python main_search.py --category bottle --seed 42 --results-dir results/search/bottle_s42
```

- Si se interrumpe, reanudar con:
  `python main_search.py --category bottle --seed 42 --results-dir results/search/bottle_s42 --resume-from results/search/bottle_s42/latest_checkpoint`
- Al terminar verificar en `results/search/bottle_s42/search_summary.json` que
  `inoperative_objectives` esté vacío `[]`. Si aparece `energy_mj`, la medición
  de energía no funcionó (revisar pynvml) y hay que repetir.

---

## FASE 3 — Reentrenamiento a presupuesto completo (PC, ~2.5 h por categoría/semilla)

```
python main_retrain.py --category bottle --seed 42 ^
    --search-results-dir results/search/bottle_s42 ^
    --results-dir results/retrain/bottle_s42 ^
    --checkpoints-dir checkpoints/final_models/bottle_s42
```
(En Linux/Mac usar `\` en lugar de `^` para continuar línea, o todo en una línea.)

- Para repetir desde cero (ignorando checkpoints previos) añadir `--no-skip-existing`.
- Verificar: `results/retrain/bottle_s42/model_ranked.csv` con 5 filas `status=ok`.

---

## FASE 4 — Copiar al Jetson (una vez por categoría)

Copiar al Jetson, conservando la misma estructura de carpetas del proyecto:

1. Todo el código: `main_*.py`, `src/`, `config/`
2. `data/splits/` (completo)
3. `data/raw/mvtec_ad/<categoria>/` — **obligatorio**: la calibración INT8 lee
   imágenes del split train desde esta ruta relativa
4. `checkpoints/final_models/bottle_s42/`
5. `results/retrain/bottle_s42/`

---

## FASE 5 — Despliegue TensorRT + benchmark (JETSON, ~1.5–3 h por categoría)

```
python3 main_deploy.py --category bottle ^
    --retrain-results-dir results/retrain/bottle_s42 ^
    --checkpoints-dir checkpoints/final_models/bottle_s42 ^
    --results-dir results/deploy/bottle_s42 ^
    --deploy-dir deployment/models/bottle_s42
```

- La calibración INT8 es automática (200 imágenes del train split, cache en
  `deployment/models/bottle_s42/<id>_int8.calib`; la primera compilación INT8
  tarda 5–10 min por modelo, las siguientes usan el cache).
- Verificar en `results/deploy/bottle_s42/runtime_metrics.csv`: filas `status=ok`
  para fp16 **e** int8. Si un INT8 aparece como `failed_engine` con mensaje
  "missing calibration", revisar el punto 3 de la FASE 4.
- Antes de medir, fijar el modo de potencia para resultados reproducibles:
  `sudo nvpmodel -m 0 && sudo jetson_clocks`

---

## FASE 6 — Validación de tracking (JETSON, ~30–45 min por engine)

Opción A — con cámara USB conectada:
```
python3 main_tracking.py --source 0 ^
    --deploy-results-dir results/deploy/bottle_s42 ^
    --results-dir results/tracking/bottle_s42
```

Opción B — con un video pregrabado (recomendado para reproducibilidad):
```
python3 main_tracking.py --source ruta/al/video.mp4 ^
    --deploy-results-dir results/deploy/bottle_s42 ^
    --results-dir results/tracking/bottle_s42
```

- Usa automáticamente el engine mejor rankeado de la FASE 5; para forzar otro:
  `--candidate-id rank000 --precision int8`.
- Corre 5 escenarios × 3 repeticiones = 15 sesiones. Los escenarios y las
  ganancias PID se ajustan en `config/tracking.yaml` (secciones `control: pid:`
  y la lista `scenarios` si se define plana).
- Verificar: `results/tracking/bottle_s42/tracking_metrics.csv` con 15 filas.

---

## FASE 7 — Copiar resultados de vuelta al PC

Copiar del Jetson al PC: `results/deploy/bottle_s42/` y `results/tracking/bottle_s42/`.

---

## FASE 8 — Reporte y figuras (PC, <1 min)

**Editar** `config/report.yaml` (líneas 10–13) para apuntar a la corrida:
```yaml
search_dir:   "search/bottle_s42"
retrain_dir:  "retrain/bottle_s42"
deploy_dir:   "deploy/bottle_s42"
tracking_dir: "tracking/bottle_s42"
```
y la línea 7: `report_dir: "results/report/bottle_s42"`.

**Ejecutar:**
```
python main_report.py
```

Salidas en `results/report/bottle_s42/`: `pareto_front.png`, `convergence.png`,
`boxplots.png`, `latex_tables.tex` (tablas listas para la tesis),
`final_summary.pdf` y `report_summary.json` (incluye los tests Mann-Whitney
fp16 vs int8).

---

## FASE 9 — Semillas y categorías adicionales

Repetir FASES 2→8 cambiando `--seed` y el sufijo de carpetas:

```
python main_search.py --category bottle --seed 43 --results-dir results/search/bottle_s43
python main_search.py --category bottle --seed 44 --results-dir results/search/bottle_s44
```
y lo mismo con `--category carpet` / `--category screw` (sufijos `carpet_s42`, etc.).

Presupuesto recomendado para las semillas adicionales (reduce 34 h → ~12 h):
```
python main_search.py --category bottle --seed 43 --population-size 24 --n-generations 30 --results-dir results/search/bottle_s43
```

Plan mínimo publicable: 3 categorías (bottle, carpet, screw) × 3 semillas.
FASE 1 no se repite nunca; FASES 5–6 solo se repiten por categoría/semilla si
se quieren barras de error también en hardware (recomendado: al menos las
3 semillas de la categoría principal).

---

## Resumen de tiempos

| Fase | Dónde | Duración | Se repite por |
|---|---|---|---|
| 1 Prepare | PC | 5–15 min | nunca |
| 2 Search | PC | ~34 h (12 h reducida) | categoría × semilla |
| 3 Retrain | PC | ~2.5 h | categoría × semilla |
| 5 Deploy | Jetson | 1.5–3 h | categoría (× semilla opcional) |
| 6 Tracking | Jetson | 30–45 min | categoría (× semilla opcional) |
| 8 Report | PC | <1 min | corrida |

Plan mínimo (3 cat × 3 semillas, semillas reducidas): **~10–12 días de GPU** en PC
+ ~2 días de Jetson, paralelizable por categoría si hay más de una máquina.
