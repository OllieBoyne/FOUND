<p align="center">
  <a href="http://ollieboyne.github.io/FOUND">
        <img width=70% src="https://ollieboyne.github.io/FOUND/images/logos/found_v1.png">
  </a>
</p>

This repository contains the code for 3D shape fitting to predicted surface normals, as shown in our paper:

> **FOUND: <ins>F</ins>oot <ins>O</ins>ptimisation with <ins>U</ins>ncertain <ins>N</ins>ormals for Surface <ins>D</ins>eformation using Synthetic Data**  \
> Winter Conference on Applications of Computer Vision 2024 \
> [Oliver Boyne](https://ollieboyne.github.io), [Gwangbin Bae](https://www.baegwangbin.com/), [James Charles](http://www.jjcvision.com), and [Roberto Cipolla](https://mi.eng.cam.ac.uk/~cipolla/) \
> [[arXiv]](https://arxiv.org/abs/2310.18279) [[project page]](https://ollieboyne.github.io/FOUND/)

<p align="center">
  <a href="http://ollieboyne.github.io/FOUND">
        <img width=90% src="https://ollieboyne.github.io/FOUND/images/itw/sliders.gif">
  </a>
</p>

## Quickstart

1) `git clone --recurse-submodules http://github.com/OllieBoyne/FOUND`
2) Install dependencies: `pip install -r requirements.txt`
3) Download the [pretrained FIND model](https://drive.google.com/drive/folders/1XWmEVo3AdnhJU2fs6igls-emp93beQpm?usp=share_link) to `data/find_nfap`
4) Download our [benchmark foot dataset](https://github.com/OllieBoyne/Foot3D) to `data/scans`
5) Fit a single scan:

```
python FOUND/fit.py --exp_name <exp_name> --data_folder <data_folder>
```

You can use `--cfg <file>.yaml` to use a config file to set parameters. See [`args.py`](utils/args.py) for all arguments, and [`example-cfg.yaml`](example-cfg.yaml) for an example config file.

6) Evaluate all of our reconstruction dataset:

```
python FOUND/eval.py --data_folder <data_folder> --gpus <gpu_indices>
```

gpu_indices is a space separated list, e.g. `--gpus 0 1 2 3`


## Data

We provide our synthetic foot dataset, [SynFoot](https://github.com/OllieBoyne/SynFoot), which contains 50K synthetic foot scans, with RGB, normals, and masks.

We also provide a benchmark multiview evaluative dataset, [Foot3D](https://github.com/OllieBoyne/Foot3D).


### Related work

Please check out all of our projects that built into this work!

- [FIND - Generative foot model](https://ollieboyne.github.io/FIND)
- [Surface Normal Estimation w/ Uncertainty](https://github.com/baegwangbin/surface_normal_uncertainty)
- [BlenderSynth - Synthetic data generation](https://ollieboyne.github.io/BlenderSynth)


### Citation

If you use our work, please cite:

```
@inproceedings{boyne2024found,
            title={FOUND: {F}oot {O}ptimisation with {U}ncertain {N}ormals for Surface {D}eformation using Synthetic Data},
            author={Boyne, Oliver and Bae, Gwangbin and Charles, James and Cipolla, Roberto},
            booktitle={Winter Conference on Applications of Computer Vision (WACV)},
            year={2024}
}
```


### Troubleshooting

If you have any issues with `trimesh` and `shapely`, see [misc/shapely.md](misc/shapely.md).
