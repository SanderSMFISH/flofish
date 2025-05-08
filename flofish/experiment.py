import logging
import json
from pathlib import Path
import re
from flofish.image import Image

#from omnipose.gpu import use_gpu
from cellpose_omni import models


class Experiment:
    """
    An experiment is a list of images in the same directory,
    that share processing parameters
    """
    def __init__(self, cfg):
        for key, value in cfg.items():
            setattr(self, key, value)
        self.filter2mRNA = { v['filter']: v['mrna'] for v in cfg['channels'].values() }
        # deal with threshold 'None'
        for key in cfg['channels'].keys():
            if 'threshold' not in cfg['channels'][key]:
                self.channels[key]['threshold'] = None


    @classmethod
    def from_cfg_file(cls, cfg_file):
        try:
            with open(cfg_file, 'r') as file:
                logging.info(f'found cfg file {cfg_file}')
                cfg = json.load(file)
                cfg['cfg_file'] = cfg_file
                return cls(cfg)
        except FileNotFoundError:
            logging.warning(f'failed to find cfg file {cfg_file}')


    def create_image_list(self):
        self.images = {}

        # get the vsi files:
        vsi_files = list(Path(self.inputdir).glob('*.vsi'))
        logging.info(f'getting .vsi files from {self.inputdir}')

        for f in vsi_files:
            logging.info(f'examining image {f}')
            stem = re.sub(r'_CY[A-Z0-9\s,._]+DAPI', '', f.stem)
            params = { 'vsi_file': f.parts[-1], 'cell_file': None, 'valid': True }

            # we need a VSI directory
            if (Path(self.inputdir) / f'_{f.stem}_').is_dir():
                logging.info(f'..found VSI directory _{f.stem}_')
            else:
                logging.warning(f'..failed to find VSI directory _{f.stem}_')
                params['valid'] = False

            # we need a DIC file
            if (Path(self.inputdir) / f'{stem}_DIC.tif').is_file():
                # sequence number precedes "DIC"
                params['cell_file'] = (Path(self.inputdir) / f'{stem}_DIC.tif').parts[-1]
                logging.info(f'..found DIC file {params["cell_file"]}')
            else:
                # sequence number follows "DIC":
                dicfilename = re.sub('(_\d\d)$', r'_DIC\1', stem)
                if (Path(self.inputdir) / f'{dicfilename}.tif').is_file():
                    params['cell_file'] = (Path(self.inputdir) / f'{dicfilename}.tif').parts[-1]
                    logging.info(f'..found DIC file {params["cell_file"]}')
                else:
                    logging.warning(f'..failed to find a DIC file for {Path(self.inputdir) / stem}')
                    params['valid'] = False

            if params['valid'] == True:
                logging.info(f'..all good, adding image to image list')
                self.images[stem] = params
                # self.images[stem] = Image.from_dict(self, params)


    def read_image_list_from_jsons(self):
        # get the img.json files:
        self.json_files = sorted(Path(self.outputdir).glob('*/img.json'))

        for f in self.json_files:
            logging.info(f'reading json file {f}')

        # for f in json_files:
        #     self.images[f.parts[-1]] = Image.from_json(f, self)


    def init_omnipose(self):

        # This checks to see if you have set up your GPU properly.
        # CPU performance is a lot slower, but not a problem if you
        # are only processing a few images.
        #self.use_GPU = use_gpu()
        self.use_GPU = False

        #default_model = models.CellposeModel(gpu=use_GPU, model_type="cyto2_omni")
        # default channels options: [1,2]

        model_list = {
            "cyto2_omni": True,
        }

        self.read_image_list_from_jsons()
        for f in self.json_files:
            logging.info(f'reading json file {f}')
            my_image = Image.from_json(f, self)
            if hasattr(my_image, 'segmentation'):
                model_list[my_image.segmentation['cells']['model_type']] = True

        for model_type in model_list.keys():
            model_list[model_type] = models.CellposeModel(gpu=self.use_GPU, model_type=model_type)

        self.model_list = model_list


def make_counts(self):
    # collate all CSV files
    return pd.DataFrame.from_dict({})

