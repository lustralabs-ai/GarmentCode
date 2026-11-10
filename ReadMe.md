# GarmentCode: Programming Parametric Sewing Patterns

> [!WARNING]
> **This is a locally modified, experimental copy of GarmentCode — not the official release.**
> It has been patched to run its cloth simulation on an AMD Radeon 8060S (ROCm/HIP) using a port of the
> GarmentCode Warp fork to Warp 1.13 (branch `garmentcode-rocm` of our Warp fork) instead of
> [NvidiaWarp-GarmentCode](https://github.com/maria-korosteleva/NvidiaWarp-GarmentCode). Changes relative to upstream
> [maria-korosteleva/GarmentCode](https://github.com/maria-korosteleva/GarmentCode) `d449629`:
>
> - **NumPy 2 support**: 2-D `np.cross` calls replaced by a `cross_2d` helper (`pygarment/pattern/{utils,core}.py`,
>   `pygarment/garmentcode/{utils,operators}.py`); `np.infty` alias restored for pyrender
>   (`pygarment/meshgen/render/pythonrender.py`); `numpy<2` pin dropped in `setup.cfg`.
> - **Warp 1.13**: `pygarment/data_config.py` no longer touches `warp.context.runtime`.
>
> The installation instructions below (NVIDIA GPU, NvidiaWarp-GarmentCode) do **not** describe this setup.
> Use [`port-docs/SETUP.md`](port-docs/SETUP.md) and `port-docs/scripts/setup_amd.sh` instead.
> [`port-docs/PLAN.md`](port-docs/PLAN.md) gives the overview of the port and its status; `gpu_validation.md` and
> `batch_validation.md` there hold the validation results, and `port-docs/reference/` the comparison harness.
> Known differences in behaviour: results differ from NVIDIA runs at
> the level of the simulation's own run-to-run chaos; box-mesh vertex counts can differ slightly between NumPy 1 and 2;
> the GUI and `pattern_fitter.py` are untested in this configuration.

![Examples of garments sampled from GarmentCode configurator](https://github.com/maria-korosteleva/GarmentCode/raw/main/assets/img/header.png)

Official Implementation of [GarmentCode: Programming Parametric Sewing Patterns](https://igl.ethz.ch/projects/garmentcode/) and [GarmentCodeData: A Dataset of 3D Made-to-Measure Garments With Sewing Patterns](https://igl.ethz.ch/projects/GarmentCodeData/).

> You can find the body measurements part of the project here: https://github.com/mbotsch/GarmentMeasurements 

## News

**[June 3, 2025]** GarmentCode online configurator demo is back online 🥳

**[April, 2025]** GarmentCode online configurator demo is temporarily unavailable. We are working on bringing it back online!

**[Nov 20, 2024]** GarmentCode configurator demo is now ONLINE 🥳 Check it out: https://garmentcode.ethz.ch/ (not for mobile) 

**[Sept 4, 2024]** We release a new version of the dataset with a few fixes and updates: [GarmentCodeData (v2)](https://doi.org/10.3929/ethz-b-000690432). See documentation for new data version for more details.

**[Aug 30, 2024]** Major release -- implementation of [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/) in `pygarment v2.0.0`, new GUI, and other updates and improvements. Basic JSON representation classes are now part of PyGarment library! See CHANGELOG for more details

**[July 1, 2024]** [GarmentCodeData](https://igl.ethz.ch/projects/GarmentCodeData/) is accepted to ECCV!

**[May 29, 2024]** First release of [GarmentCodeData](https://doi.org/10.3929/ethz-b-000673889) dataset!

**[Oct 18, 2023]** First release of GarmentCode!

## Documents

1. [Installation](https://github.com/maria-korosteleva/GarmentCode/blob/main/docs/Installation.md)
2. [Running Configurator](https://github.com/maria-korosteleva/GarmentCode/blob/main/docs/Running_garmentcode.md) 
3. [Running Data Generation (warp)](https://github.com/maria-korosteleva/GarmentCode/blob/main/docs/Running_data_generation.md) 
3. [Body measurements](https://github.com/maria-korosteleva/GarmentCode/blob/main/docs/Body%20Measurements%20GarmentCode.pdf)
4. [Dataset documentation](https://www.research-collection.ethz.ch/handle/20.500.11850/673889)
3. [Running Old Maya+Qualoth tools](https://github.com/maria-korosteleva/GarmentCode/blob/main/docs/Running_Maya_Qualoth.md) 

## Navigation

### Library

[PyGarment](https://github.com/maria-korosteleva/GarmentCode/tree/main/pygarment) is the core library described in the GarmentCode paper. It contains the base types (Edge, Panel, Component, Interface, etc.), as well as edge factory and various helpers and operators that help you design sewing patterns.  

See [Installation instructions](https://github.com/maria-korosteleva/GarmentCode/tree/main/docs/Installation.md) before use.

### Examples

* [assets/garment_programs/](https://github.com/maria-korosteleva/GarmentCode/tree/main/assets/garment_programs/) contains the code of garment components designed using PyGarment. 
* [assets/design_params/](https://github.com/maria-korosteleva/GarmentCode/tree/main/assets/design_params/), [assets/bodies/](https://github.com/maria-korosteleva/GarmentCode/tree/main/assets/bodies/) contain examples of design and body measurements presets. They can be used in both GarmentCode GUI and `test_garmentcode.py` script.

> NOTE: [assets/design_params/default.yaml](https://github.com/maria-korosteleva/GarmentCode/blob/main/assets/design_params/default.yaml) is the setup used by GUI on load. Changing this file results in changes in the GUI initial state =) 


## Citation

If you are using our system in your research, please cite our papers:

```bibtex
@inproceedings{GarmentCodeData:2024,
  author = {Korosteleva, Maria and Kesdogan, Timur Levent and Kemper, Fabian and Wenninger, Stephan and Koller, Jasmin and Zhang, Yuhan and Botsch, Mario and Sorkine-Hornung, Olga},
  title = {{GarmentCodeData}: A Dataset of 3{D} Made-to-Measure Garments With Sewing Patterns},
  booktitle={Computer Vision -- ECCV 2024},
  year = {2024},
  keywords = {sewing patterns, garment reconstruction, dataset},
}
```

```bibtex
@article{GarmentCode2023,
  author = {Korosteleva, Maria and Sorkine-Hornung, Olga},
  title = {{GarmentCode}: Programming Parametric Sewing Patterns},
  year = {2023},
  issue_date = {December 2023},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {42},
  number = {6},
  doi = {10.1145/3618351},
  journal = {ACM Transaction on Graphics},
  note = {SIGGRAPH ASIA 2023 issue},
  numpages = {16},
  keywords = {sewing patterns, garment modeling}
}
```

## Issues, questions, suggestions

Please post your issues and questions to [GitHub Issues](https://github.com/maria-korosteleva/GarmentCode/issues).

For other requests you can find my info on https://korosteleva.com/.  

## Contributors

This project is brought to life by these people:

* [Maria Korosteleva](https://github.com/maria-korosteleva)
* [Jasmin Koller](https://github.com/JasminKoller)
* [Yuhan Zhang](https://github.com/yuhan-zh)
* [Yuhan Liu](https://github.com/yuhanliu-tech)
* [Ami Beuret](https://github.com/amibeuret)
* [Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/index.php)

The body measurements team developed [GarmentMeasurements](https://github.com/mbotsch/GarmentMeasurements): 
* [Fabian Kemper](https://github.com/fabiankemper)
* [Stephan Wenninger](https://github.com/stephan-wenninger)
* [Mario Botsch](https://github.com/mbotsch)
