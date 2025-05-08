import pytest, logging, time
from pathlib import Path

from flofish.experiment import Experiment
from flofish.image import Image

import numpy as np
from skimage import io

#from omnipose.gpu import use_gpu
from cellpose_omni import models


@pytest.fixture
def exp16():
    cfg_file = "flofish/tests/data/exp16/config.json"
    return Experiment.from_cfg_file(cfg_file)


@pytest.fixture
def my_image(exp16):
    my_params = {
        'vsi_file': "MG1655_GLU_OD_0.3_left_CY5, CY3.5 NAR, CY3, DAPI_02.vsi",
        'cell_file': "MG1655_GLU_OD_0.3_left_DIC_02.tif"
    }
    return Image.from_dict(my_params, exp16)


@pytest.fixture
def my_grgb():
    file = "flofish/tests/data/exp16/input/grgb.npy"
    return np.load(file)


def test_from_params(my_image):
    assert isinstance(my_image, Image) == True
    assert isinstance(my_image.experiment, Experiment) == True
    pass


def test_omnipose(my_grgb):

    params = {
        'channels': [0, 0],  # always define this with the model
        'rescale': None,  # upscale or downscale your images, None = no rescaling
        'mask_threshold': 0,  # erode or dilate masks with higher or lower values between -5 and 5
        'flow_threshold': 0,
        'min_size': 200,
        'diameter': 0,
        'invert': False,
        'transparency': True,  # transparency in flow output
        'omni': True,  # we can turn off Omnipose mask reconstruction, not advised
        'cluster': True,  # use DBSCAN clustering
        'resample': True,  # whether or not to run dynamics on rescaled grid or original grid
        'verbose': False,  # turn on if you want to see more output
        'tile': False,  # average the outputs from flipped (augmented) images; slower, usually not needed
        'niter': None,
        # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
        'augment': False,  # Can optionally rotate the image and average network outputs, usually not needed
        'affinity_seg': False,  # new feature, stay tuned...
    }

    # This checks to see if you have set up your GPU properly.
    # CPU performance is a lot slower, but not a problem if you
    # are only processing a few images.
    #use_GPU = use_gpu()
    use_GPU = False
    # model = models.CellposeModel(gpu=use_GPU, model_type="cyto2")
    # params['channels'] = [2, 0]

    model = models.CellposeModel(gpu=use_GPU, model_type="cyto2_omni")
    params['channels'] = [1, 2]

    mask, flow, style = model.eval(my_grgb, **params)
    logging.info(f'Found {np.max(mask)} cells')
    savepath = Path('flofish/tests/data/exp16/output/MG1655_GLU_OD_0.3_left_02')
    savepath.mkdir(exist_ok=True)
    io.imsave(savepath / 'masks.tif', mask)

    assert np.all(mask == io.imread('flofish/tests/data/exp16/output/MG1655_GLU_OD_0.3_left_02/masks.tif'))


def test_pipeline(my_image):
    # load image (~ 01-configure)
    tic = time.time()
    my_image.read_image()
    my_image.read_cells()
    my_image.align()
    my_image.create_grgb()

    # crop image
    my_image.crop()

    # save image (write to dir)
    my_image.save_input_layers()
    my_image.time['01-configure'] = time.time() - tic
    my_image.save_metadata('configure')

    # segment image (~ 02-segment)
    tic = time.time()
    my_image.experiment.init_omnipose()
    my_image.segment_cells()
    my_image.time['02-segment-cells'] = time.time() - tic
    my_image.save_metadata('segment-cells')

    tic = time.time()
    my_image.segment_dapi()
    my_image.time['02-segment-dapi'] = time.time() - tic
    my_image.save_metadata('segment-dapi')

    # postprocess masks
    tic = time.time()
    my_image.postprocess_masks()
    my_image.time['02-segment-pp'] = time.time() - tic
    my_image.save_metadata('segment-pp')

    # cropping: better process the whole picture and then crop
    # (i.e. select cells from good area)
    # duh...

    # detect spots (~ 03-detect-spots)
    tic = time.time()
    my_image.find_focus()
    my_image.filter()
    my_image.detect_spots()
    my_image.time['03-detect-spots'] = time.time() - tic
    my_image.save_metadata('detect-spots')

    # decompose spots (~ 04-decompose-spots)
    tic = time.time()
    my_image.decompose_spots()
    my_image.time['04-decompose-spots'] = time.time() - tic
    my_image.save_metadata('decompose-spots')

    # assign spots (05-assign-spots)
    my_image.assign_spots()
    my_image.save_metadata('assign-spots')

    # save image (json pickle)
    tic = time.time()
    my_image.save("05")
    my_image.time['05-save'] = time.time() - tic
    my_image.save_metadata('save')

    pass


@pytest.fixture
def my_image_from_json(exp16):
    img_json = "flofish/tests/data/exp16/output/MG1655_GLU_OD_0.3_left_02/img.json"
    return Image.from_json(img_json, exp16)


def test_from_json(my_image_from_json):
    assert isinstance(my_image_from_json, Image) == True
    assert isinstance(my_image_from_json.experiment, Experiment) == True
    pass
