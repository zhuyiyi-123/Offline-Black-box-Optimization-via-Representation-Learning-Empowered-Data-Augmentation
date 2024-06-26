# Offline-Black-box-Optimization-via-Representation-Learning-Empowered-Data-Augmentation

We propose the Cyclic Predictive Encoder method (CPE) to augment the non-ideal dataset.

## Installation
The environment of CPE can be installed as:
```bash
conda create --name CPE --file requirements.txt
conda activate CPE
```
## Reproducing Performance
For continuous tasks, we can run CPE as:
```bash
python main_con.py --task Superconductor-RandomForest-v0 --sample 1 --ratio 0.2
```
For discrete tasks, we can run CPE as:
```bash
python main_dis.py --task TFBind8-Exact-v0 --sample 1 --ratio 0.2
```
