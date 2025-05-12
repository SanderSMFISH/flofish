# Description
A light wrapper around the Big-FISH smRNA FISH spot detection module:

https://big-fish.readthedocs.io/

Goal is to automate processing and record processing parameters.

# Expected inputs
The module expects a directory containing the following files:
- Fluorescent channels: .vsi microscopy files and the corresponding vsi directories
- Cell outlines: 2D .tif files (e.g. DIC images)
- Configuration: a json file containing channel and processing parameters (see `workflow.ipynb` to download an example)


# Installation
You can install `flofish` locally after cloning the repo:

    pip install -e . 

# Dependencies
Dependencies are listed in `pyproject.toml`

Mostly:
- `big-fish`: for smFISH image processing.
- `omnipose`: a working Omnipose with GPU support. GPU is not a prerequisite as such but I haven't tested without a GPU so the code might need tweaking to run without a GPU.
- `bioio_bioformats`: to open VSI images. This requires a working Java environment.
- `pytests`: to perform test on the library.
- `natsort` : to sort file names. Used in sorting image lists. 

Details:
 - Omnipose

   On MacOS: install from git source or git clone on MacOS in a conda env with python<=3.10.11 (see [this issue](https://github.com/kevinjohncutler/omnipose/issues/14))

   First add `jupyterlab` to `omnipose_mac_environment.yml` from [Omnipose](https://omnipose.readthedocs.io/installation.html) repository, then:
   ```
   conda env create --name myenv --file /Volumes/DataDrive/omnipose_mac_environment.yml
   conda activate myenv
   pip install git+https://github.com/kevinjohncutler/omnipose.git
   pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
   ```

- Omnipose GUI (nice to have, not required)

   Run `omnipose` from the command line.

Omnipose has to be installed with GPU support (https://omnipose.readthedocs.io/installation.html#gpu-support). Alternatively, the variable use_GPU can be set to False in experiment.py and the test python scripts.  

- Other
   ```
   conda install -c conda-forge scyjava
   pip install "napari[pyqt6]"
   pip install bioio bioio_bioformats big-fish jsonpickle pathlib  
   ```
- Might be needed:
   - `export JAVA_HOME=/Users/adele/miniconda3/envs/myenv/lib/jvm`
   - deactivate and reactivate myenv for `scyjava` to work
   - edit `peakdetect.py` fft import line to: `from scipy.fft import fft, ifft`
# Test data
Available at:
 
https://zenodo.org/records/15363842

# Typical workflow
1. Batch process inputs to TIF files: `flofish/bulk_processing.ipynb`
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures: `flofish/bulk_processing.ipynb`
4. Find good spot detection parameters using napari-flofish plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots: `flofish/bulk_processing.ipynb`