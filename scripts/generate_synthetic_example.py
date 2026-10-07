"""Generate toy feature inputs. These are not patient or validation data."""
from pathlib import Path
import numpy as np

path = Path(__file__).resolve().parents[1] / 'examples' / 'synthetic_path_features.npz'
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    raise FileExistsError(path)
rng = np.random.default_rng(41300)
np.savez(path, ls_features=rng.normal(size=(2,256)).astype(np.float32))
print(path)
