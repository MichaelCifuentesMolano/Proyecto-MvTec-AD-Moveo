# config/legacy/

Archivos de configuración **obsoletos**, conservados solo como referencia.
Ninguno de estos archivos es leído por los scripts del pipeline.

| Archivo | Motivo |
|---|---|
| `search_OLD_no_compatible.yaml` | Esquema anidado que nunca coincidió con `SearchConfig` (main_search.py). Describía además un espacio de búsqueda (resnet/efficientnet) distinto al implementado en `src/nas/search_space.py`. Sustituido por `config/search.yaml` plano. |
| `tracking_OLD_no_compatible.yaml` | Esquema anidado (camera/detector/control) incompatible con `TrackingConfig`. Sustituido por `config/tracking.yaml` plano. |
| `train_no_usado_por_ningun_script.yaml` | Ningún script lo carga; los hiperparámetros de entrenamiento reales viven en `config/retrain.yaml` (`RetrainConfig`). |
| `device_referencia_jetson.yaml` | Ningún script lo carga. Útil como documentación de la Jetson, pero la versión de JetPack que menciona (5.1.2) está desactualizada: verificar con `cat /etc/nv_tegra_release` tras flashear. |

Historia: hasta junio de 2026 los seis `main_*.py` buscaban la carpeta
`configs/` (con s) que no existía, por lo que **ninguna ejecución cargó
estos YAML**; todo corrió con los defaults internos de los dataclasses.
Los scripts ahora apuntan a `config/` y avisan con WARNING si el archivo
de configuración no existe.
