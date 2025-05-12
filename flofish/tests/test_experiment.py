import pytest, logging, time

from flofish.experiment import Experiment
from flofish.image import Image


@pytest.fixture
def exp():
    return Experiment.from_cfg_file("flofish/tests/data/config.json")

# @pytest.fixture
# def exp24():
#     return Experiment.from_cfg_file("/Volumes/KINGSTON/Florence/smFISH/zenodo/smfish-analysis/tests/data/exp24/config.json")


def test_experiment_exp(exp):
    assert len(exp.channels) == 4
    exp.create_image_list()
    pass

# def test_experiment_exp24(exp24):
#     assert len(exp24.channels) == 4
#     exp24.create_image_list()
#     pass


def test_experiment_from_jsons_exp(exp):
    assert len(exp.channels) == 4
    exp.read_image_list_from_jsons()
    pass


# def test_experiment_from_jsons_exp24(exp24):
#     assert len(exp24.channels) == 4
#     exp24.read_image_list_from_jsons()
#     pass


def test_configure_exp(exp):
    exp.create_image_list()

    for params in exp.images.values():
        logging.info(params)
        my_image = Image.from_dict(params, exp)

        tic = time.time()
        my_image.read_image()
        my_image.read_cells()
        my_image.align()
        my_image.create_grgb()

        # save image (write to dir)
        my_image.save_input_layers()
        my_image.time['01-configure'] = time.time() - tic
        my_image.save_metadata("configure")
    pass

def test_segment_exp(exp):
    exp.read_image_list_from_jsons()
    exp.init_omnipose()

    for f in exp.json_files:
        logging.info(f'image: {f}')
        my_image = Image.from_json(f, exp)

        # segment image (~ 02-segment)
        tic = time.time()
        my_image.segment_cells()
        my_image.time['02-segment-cells'] = time.time() - tic

        tic = time.time()
        my_image.segment_dapi()
        my_image.time['02-segment-dapi'] = time.time() - tic

        # postprocess masks
        tic = time.time()
        my_image.postprocess_masks()
        my_image.time['02-segment-pp'] = time.time() - tic

        # indicate in metadata which stage was run last
        my_image.save_metadata("segment")

def test_spots_exp(exp):
    exp.read_image_list_from_jsons()

    for f in exp.json_files:
        logging.info(f'image: {f}')
        my_image = Image.from_json(f, exp)

        # detect spots (~ 03-detect-spots)
        tic = time.time()
        my_image.find_focus()
        my_image.filter()
        my_image.detect_spots()
        my_image.time['03-detect-spots'] = time.time() - tic

        # decompose spots (~ 04-decompose-spots)
        tic = time.time()
        my_image.decompose_spots()
        my_image.time['04-decompose-spots'] = time.time() - tic

        # assign spots (05-assign-spots)
        my_image.assign_spots()

        # save image (json pickle)
        tic = time.time()
        my_image.save("05")
        my_image.time['05-save'] = time.time() - tic

        my_image.save_metadata("spots")
