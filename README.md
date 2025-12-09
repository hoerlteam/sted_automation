# A flexible framework for automated STED super-resolution microscopy

**autoSTED** allows you to automate an Abberior Instruments STED microscope in a flexible and generic way.
We have sucessfully used it for multiple super-resolution studies dealing mainly with chromatin organization:
* [Brandstetter et al. (Biophysical Journal, 2022)](https://linkinghub.elsevier.com/retrieve/pii/S0006349522001096)
* [Palikyras et al. (Aging Cell, 2024)](https://onlinelibrary.wiley.com/doi/10.1111/acel.14083)
* [Steinek et al. (Cell Reports Methods, 2024)](https://doi.org/10.1016/j.crmeth.2024.100840)
* [Stumberger et al. (Nucleic Acids Research, 2025)](https://academic.oup.com/nar/article/doi/10.1093/nar/gkaf1255/8373955)

The technical manuscript describing the framework (**D. Hörl: *A flexible framework for automated STED super-resolution microscopy*, 2025**) is available as a preprint at [https://www.biorxiv.org/content/10.1101/2025.05.05.652196v1](https://www.biorxiv.org/content/10.1101/2025.05.05.652196v1).

## Warning

**autoSTED is designed to automate operation of expensive scientific equipment. While we don't do anything that is not also possible manually via Imspector or SpecPy, be careful to avoid damage to the microscope (e.g. through collisions of stage and objective) or danger to yourself or others (e.g. from lasers).**

- **SpecPy and therefore autoSTED uses SI units, so a movement or image size of ```1.0``` would correspond to 1 meter. Stages and scanners should typically stop when they reach their limits, but nonetheless, pay attention to specify, e.g. distances in the appropriate fractions of meters (in Python you can use scientific notation like ```5e-6``` to indicate $5*10^{-6}$ (m) = 5 micron).**
- **Never use autoSTED to operate a microscope when laser safety measures are disabled!**

## Installation

### Requirements

autoSTED is designed to operate microscopes controlled through Imspector. Thus, for full functionality, it needs to be installed on the control workstation of a microscope with Imspector installed. Imspector and the Python interface package SpecPy are Windows-exclusive. Currently, the software is developed and tested with Imspector 16.3.19720 (2024.08), SpecPy 1.2.3 and Python 3.11.

Imspector ships with Python installation(s) that you can use, but to not clutter those up, we recommend that you use your own [Anaconda/Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main) installation. 

The autoSTED package itself only depends on common Python data science / image processing packages and our own freely available collection of helper functions [CalmUtils](https://github.com/CALM-LMU/CalmUtils) which are automatically downloaded and installed when you install autoSTED via pip. Depending on your connection speed the installation should not take more than a few minutes.

To **test the functionality of the framework on any system**, you can skip installation of SpecPy and run the simulated demo (see below).

### SpecPy and compatible NumPy

Our framework talks to the microscope control software Imspector via the SpecPy interface.

SpecPy ```.whl``` files can be found in the Imspector installation folder (e.g. ```C:\Imspector\Versions\{VersionNr}\python\specpy\{SpecPy_Version_Nr}```). SpecPy is built for specific Python and NumPy versions, indicated in the file/folder name, so it is a good idea to create a conda environment with the corresponding versions (e.g. for Python 3.11.4, NumPy 1.24.3):

```bash
conda create -n autosted-env python=3.11.4 numpy=1.24.3
```

Then, in you new environment, you can install SpecPy:

```bash
conda activate autosted-env
pip install C:/Imspector/path/to/specpy.whl
```

### autoSTED

Once you have an environment set, you can install autoSTED (this repository):

```bash
git clone https://github.com/hoerlteam/sted_automation.git
cd sted_automation
pip install .
```

Alternatively, install directly without cloning:

```bash
pip install git+https://github.com/hoerlteam/sted_automation.git
```

### Aviod NumPy updates

**Warning:** Installing additional third-party packages may sometimes update NumPy, causing SpecPy to stop working. Consider using the ```--dry-run``` options of ```pip/conda``` to check if an installation might cause problems.

Sometimes, it helps to explicity re-state the NumPy version during install:

```bash
pip install package-of-choice numpy==1.24.3
conda install package-of-choice numpy=1.24.3
```

Pinning the NumPy version in ```conda``` might also be worth a look: [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#preventing-packages-from-updating-pinning)

## Usage

You can find Jupyter notebooks showcasing the use of autoSTED under ```examples```. To run the notebooks, you should also install Jupyter into your environment via conda/pip.

We assume that the microscope and Imspector are running and typically that the sample is (roughly) in focus when an automation pipeline is started. If you have a newer Imspector version with the LIGHTBOX interface, you need to switch to the classical interface via ```Ctrl-Alt-Shift-F11```.

### Simulated Demo

**For demonstration purposes, we provide the ```examples/demo_overview_detail.ipynb``` notebook that performs virtual microscopy in a pre-recorded dataset, allowing testing of autoSTED without a microscope.** 

This notebook showcases overview-detail imaging in the existing data. With the default settings it should not run longer than a few minutes. You can use the result of any autoSTED run saved to a combined HDF5 file as the basis for the virtual run, but we also provide an example dataset at [https://osf.io/et4br/?view_only=23d060ed9a8f48b6863852ed65ac2c45](https://osf.io/et4br/?view_only=23d060ed9a8f48b6863852ed65ac2c45) (~250MB download). 

If you want to use the [Cellpose](https://github.com/MouseLand/cellpose)-based cell detection in the demo notebook, you need to install that package via ```pip install cellpose```.