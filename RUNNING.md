# Running LeWorldModel

This repo expects the official HDF5 datasets and checkpoints to live under
`STABLEWM_HOME`. The default examples below use a cache inside this workspace.

## 1. Create the environment

Use Python 3.10, as recommended by the upstream README.

PowerShell:

```powershell
cd C:\Users\ADMIN\Downloads\JEPA\le-wm
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `py -3.10` is not available, install Python 3.10 first. The active Python on
this machine is MSYS2 Python 3.12 without `pip`, so it is not a good runtime for
this project.

Linux/macOS:

```bash
cd /path/to/le-wm
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2. Point the code at your data cache

PowerShell:

```powershell
$env:STABLEWM_HOME = "C:\Users\ADMIN\Downloads\JEPA\.stable-wm"
New-Item -ItemType Directory -Force $env:STABLEWM_HOME
```

Linux/macOS:

```bash
export STABLEWM_HOME=/path/to/.stable-wm
mkdir -p "$STABLEWM_HOME"
```

## 3. Prepare datasets later

When you download a dataset from the Hugging Face collection, decompress it so
the final `.h5` file name matches the YAML config:

- `config/train/data/pusht.yaml` expects `$STABLEWM_HOME/pusht_expert_train.h5`
- `config/train/data/tworoom.yaml` expects `$STABLEWM_HOME/tworoom.h5`
- `config/train/data/dmc.yaml` expects `$STABLEWM_HOME/reacher.h5`
- `config/train/data/ogb.yaml` expects the OGB dataset name used by
  `stable_worldmodel`

## 4. Smoke-test the model code

This does not require any dataset or `stable_worldmodel`. It only checks the
local JEPA modules with random tensors.

```powershell
python scripts\smoke_test.py
```

Expected output:

```text
smoke test passed
```

## 5. Train

PushT is the default config:

```powershell
python train.py data=pusht
```

Useful local overrides:

```powershell
python train.py data=tworoom trainer.max_epochs=1 loader.batch_size=16
python train.py data=pusht wandb.enabled=true wandb.config.entity=YOUR_ENTITY wandb.config.project=YOUR_PROJECT
```

Checkpoints are written under `$STABLEWM_HOME/<hydra job id>/`.

## 6. Evaluate or plan

`policy` is the checkpoint path relative to `$STABLEWM_HOME`, without the
`_object.ckpt` suffix:

```powershell
python eval.py --config-name=pusht.yaml policy=pusht/lewm
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm
```

The solver defaults to CUDA. To force CPU for a small sanity run:

```powershell
python eval.py --config-name=tworoom.yaml policy=tworoom/lewm solver.device=cpu
```
