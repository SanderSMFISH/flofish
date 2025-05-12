import logging
from pathlib import Path
from bioio import BioImage
import bioio_bioformats
from bioio.writers import OmeTiffWriter
from skimage import io
import numpy as np
import re
import json, jsonpickle

#from omnipose.gpu import use_gpu
# from cellpose_omni import io, transforms
from cellpose_omni import models

from bigfish.detection import detect_spots, decompose_dense
from bigfish.stack import remove_background_gaussian, compute_focus
from scipy.signal import savgol_filter

from flofish.utils import translate_image, remove_cell_masks_with_no_dapi, filter_cells_by_area, filter_cells_by_shape, get_regionprops, expand_masks
from flofish.utils import find_in_focus_indices, find_high_density_patch
from flofish.utils import spot_assignment



jsonpickle.handlers.register('pandas')

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s ', datefmt='%m/%d/%Y %I:%M:%S%p',
                    level=logging.INFO)


def access():
    logging.info("OK")


class Image:

    def __init__(self, exp, img):
        """
        Class for a FISH image.

        Parameters
        ----------
        - exp : an Experiment object containing at least
            - channel configuration
            - spot detection parameters
            - input and output directories

        - img : a dict containing at least two keys
            - vsi_file : the VSI file with the fluorescent channels
            - cell_file : the file with the cell outline data, e.g. a 2D DIC picture

            Optional keys:
            - segmentation: segmentation parameters
            - crop_by: cropping coordinates
            - translate_by: cell file translation vector


        Returns
        -------
        a FISH image object.
        """

        setattr(self, "experiment", exp)
        setattr(self, "parameters", exp.parameters)
        setattr(self, "time", {})
        setattr(self, "results", {})
        for key, value in img.items():
            setattr(self, key, value)


        if 'savepath' not in img:
            try:
                stem = re.sub(r'_CY[A-Z0-9\s,._]+DAPI', '', Path(img['vsi_file']).stem)
                self.stem = stem
                self.savepath = str(Path(exp.outputdir) / stem)
                Path(self.savepath).mkdir(parents=True, exist_ok=True)
                logging.info(f"created output dir {self.savepath}")
            except:
                logging.warning(f"failed to create output dir {Path(exp.outputdir) / stem}")
        else:
            try:
                if Path(img['savepath']).is_dir():
                    logging.info(f"found output dir {img['savepath']}")
            except FileNotFoundError:
                logging.warning(f"could not find output dir {img['savepath']}")


    @classmethod
    def from_json(cls, filename, exp):
        img = json.load(open(filename))
        return cls(exp, img)


    @classmethod
    def from_dict(cls, params, exp):
        return cls(exp, params)


    def read_image(self):
        img = {}

        logging.info(f'reading fluorescence image: {self.vsi_file}')
        image = BioImage(Path(self.experiment.inputdir) / self.vsi_file, reader=bioio_bioformats.Reader)
        logging.info(f'..found scenes: {image.scenes}')
        for s in image.scenes:
            if s == 'macro image':
                logging.info(f"....ignoring scene '{s}'")
            else:
                image.set_scene(s)
                logging.info(f"....reading scene '{s}' (shape {image.shape})")
                logging.info(f"....found channels {image.channel_names}")

                for ch in enumerate(s.replace(r'001 ', '').split(", ")):
                    if ch[1] in self.experiment.filter2mRNA.keys():
                        filter = ch[1]
                        mrna = self.experiment.filter2mRNA[ch[1]]
                        img[mrna] = {}
                        logging.info(f"......reading in channel: C={ch[0]} filter={filter} mrna={mrna}")
                        img[mrna]['data'] = image.get_image_data("ZYX", C=ch[0])
                        self.mrna = img
                        if mrna != 'DAPI':
                            self.results[mrna] = {}

                    else:
                        logging.warning(f"......ignoring unknown channel {filter}")


    def load_data(self, layer):
        if layer == 'grgb':
            self.grgb = np.load(Path(self.savepath) / f'grgb.npy')
        elif layer == 'cells':
            self.cell_masks = io.imread(Path(self.savepath) / f'DIC_masks_pp.tif')
            self.cell_masks_expanded = io.imread(Path(self.savepath) / f'DIC_masks_pp_expanded.tif')
            self.dapi_masks = io.imread(Path(self.savepath) / f'DAPI_masks.tif')
        elif layer == 'mrna':
            setattr(self, 'mrna', {})
            for ch in self.experiment.channels.keys():
                self.mrna[ch] = {}
                self.mrna[ch]['aligned'] = io.imread(Path(self.savepath) / f'{ch}.tif')


    def read_cells(self):
        logging.info(f'reading cell image: {self.cell_file}')
        cells = io.imread(Path(self.experiment.inputdir) / self.cell_file)
        self.cells = {}
        self.cells['data'] = cells


    def crop(self):
        if hasattr(self, 'crop_by'):
            ymin, xmin, ymax, xmax = self.crop_by
            self.cells['aligned'] = self.cells['aligned'][ymin:ymax, xmin:xmax]
            for ch in self.mrna.keys():
                self.mrna[ch]['aligned'] = self.mrna[ch]['aligned'][ymin:ymax, xmin:xmax]


    def align(self):
        logging.info(f"aligning DIC image by: {self.experiment.parameters['translate_dic_by']}")
        dy = self.experiment.parameters['translate_dic_by'][0]
        dx = self.experiment.parameters['translate_dic_by'][1]

        # translate and crop DIC
        img_translated = translate_image(self.cells['data'], dy, dx)
        self.cells['aligned'] = img_translated[1]

        # crop fluorescent channels
        for ch in self.mrna.keys():
            data = self.mrna[ch]['data']
            Y, X = data.shape[1], data.shape[2]
            self.mrna[ch]['aligned'] = data[:, max(dy, 0):Y + min(dy, 0), max(dx, 0):X + min(dx, 0)]


    def create_grgb(self):
        logging.info(f'creating GRGB composite image')
        cells = self.cells['aligned']
        dapimaxproj = np.max(self.mrna['DAPI']['aligned'], axis=0)
        self.grgb = np.stack([np.zeros(cells.shape), cells, dapimaxproj, np.zeros(cells.shape)], axis=0)


    def save_input_layers(self):
        logging.info(f'saving layers to: {self.savepath}')
        # fluorescent channels
        for ch in self.mrna.keys():
            data = self.mrna[ch]['aligned']
            savepath = Path(self.savepath) / f'{ch}.tif'
            OmeTiffWriter.save(data, savepath, dim_order="ZYX", channel_names=[ch])
            logging.info(f'..saving {ch} channel to {savepath}')

        # cell channel
        data = self.cells['aligned']
        savepath = Path(self.savepath) / 'DIC.tif'
        # io.imwrite(savepath, data)
        io.imsave(savepath, data)
        logging.info(f'..saving DIC channel to {savepath}')

        # grgb file for segmentation
        np.save(Path(self.savepath) / 'grgb.npy', self.grgb)


    def save_metadata(self, stage):
        logging.info(f'saving metadata to file: {Path(self.savepath) / "img.json"}')

        keys = [
            'vsi_file',
            'cell_file',
            'stem',
            'savepath',
            'parameters',
            'segmentation',
            'results',
            'time'
        ]
        metadata = { i: getattr(self, i) for i in keys if hasattr(self, i)}
        metadata['last_run'] = stage

        with open(Path(self.savepath) / "img.json", "w") as f:
            json.dump(metadata, f, default=vars, indent=4)

        return metadata


    def save(self, suffix):
        logging.info(f'saving image to: {self.savepath}/img.{suffix}.pkl')

        with open(Path(self.savepath) / f'img.{suffix}.pkl', "w") as f:
            f.write(jsonpickle.encode(self))


    def unpickle(pickle_file):
        with open(pickle_file, "r+") as f:
            return jsonpickle.decode(f.read())


    # redo for list of images to avoid loading the models for each image
    def segment_cells(self):
        logging.info(f'segmenting DIC image')

        # check if the object contains the grgb data,
        # otherwise load it
        self.load_data('grgb')

        if self.experiment.parameters["organism"] == 'E.coli': 

            params = {
                'channels': [1, 2],  # always define this with the model
                'rescale': None,  # upscale or downscale your images, None = no rescaling
                'mask_threshold': 0,  # erode or dilate masks with higher or lower values between -5 and 5
                'flow_threshold': 0,
                'min_size': 20,
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
        

        elif self.experiment.parameters["organism"] == 'S.cerevisiae': 

            params = {
                'channels': [1, 2],  # always define this with the model
                'rescale': None,  # upscale or downscale your images, None = no rescaling
                'mask_threshold': -3,  # erode or dilate masks with higher or lower values between -5 and 5
                'flow_threshold': 1,
                'min_size': 10,
                'diameter': 50,
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

        # overwrite default segmentation parameters with img.json values
        if hasattr(self, 'segmentation'):
            for param, value in self.segmentation['cells'].items():
                params[param] = value
            if 'model_type' in params:
                del params['model_type']

            model_type = self.segmentation['cells']['model_type']
            logging.info(f"model_type from config-selected: {self.segmentation['cells']['model_type']}, channels: {params['channels']}")
        else:
            self.segmentation = { 'cells': {}, 'dapi': {} }
            model_type = 'cyto2_omni'
            logging.info(f"default model_type: {model_type}, channels: {params['channels']}")

        model = self.experiment.model_list[model_type]
        mask, flow, style = model.eval(self.grgb, **params)
        logging.info(f'Found {np.max(mask)} cells')

        self.cell_masks = mask
        self.segmentation['cells'] = params
        self.segmentation['cells']['model_type'] = model_type
        maskfile_latest = Path(self.savepath) / f'DIC_masks.model={model_type}_chan={str(params["channels"]).replace(" ", "")}_diameter={params["diameter"]}_minsize={params["min_size"]}_mask={params["mask_threshold"]}_flow={params["flow_threshold"]}.tif'
        io.imsave(maskfile_latest, mask)
        cellmaskfile = Path(self.savepath) / 'DIC_masks.tif'
        #cellmaskfile.unlink(missing_ok=True)
        #cellmaskfile.symlink_to(maskfile_latest.parts[-1])
        io.imsave(cellmaskfile, mask)
        logging.info(f"writing cell mask to {maskfile_latest}")



    def segment_dapi(self):
        logging.info(f'segmenting DAPI image')

        # check if the object contains the grgb data,
        # otherwise load it
        self.load_data('grgb')

        # This checks to see if you have set up your GPU properly.
        # CPU performance is a lot slower, but not a problem if you
        # are only processing a few images.
        #use_GPU = use_gpu()
        use_GPU = False
        # model for nuclei
        model_type = "nuclei"
        model = models.CellposeModel(gpu=use_GPU, model_type=model_type)

        # pass parameters to model
        params = {
            'channels': [0, 0], # always define this with the model
            'rescale': None, # upscale or downscale your images, None = no rescaling
            'mask_threshold': 0.0, # erode or dilate masks with higher or lower values between -5 and 5
            'flow_threshold': 0.0,
            'min_size': 10,
            'diameter': 0.0,
            'invert': False,
            'transparency': True, # transparency in flow output
            'omni': True, # we can turn off Omnipose mask reconstruction, not advised
            'cluster': True, # use DBSCAN clustering
            'resample': True, # whether or not to run dynamics on rescaled grid or original grid
            'verbose': False, # turn on if you want to see more output
            'tile': False, # average the outputs from flipped (augmented) images; slower, usually not needed
            'niter': None, # default None lets Omnipose calculate # of Euler iterations (usually <20) but you can tune it for over/under segmentation
            'augment': False, # Can optionally rotate the image and average network outputs, usually not needed
            'affinity_seg': False, # new feature, stay tuned...
        }

        
        mask, flow, style = model.eval(self.grgb[2, ...], **params)
        logging.info(f'Found {np.max(mask)} cells')

        self.dapi_masks = mask
        self.segmentation['dapi'] = params
        self.segmentation['dapi']['model_type'] = model_type


        maskfile_latest = Path(self.savepath) / f'DAPI_masks_model={model_type}_chan={str(params["channels"]).replace(" ", "")}_diameter={params["diameter"]}_minsize={params["min_size"]}_mask={params["mask_threshold"]}_flow={params["flow_threshold"]}.tif'
        io.imsave(maskfile_latest, mask)
        cellmaskfile = Path(self.savepath) / 'DAPI_masks.tif'
        #cellmaskfile.unlink(missing_ok=True)
        #cellmaskfile.symlink_to(maskfile_latest.parts[-1])
        io.imsave(cellmaskfile, mask)
        logging.info(f"writing DAPI mask to {maskfile_latest}")


    def postprocess_masks(self):
        logging.info(f'postprocessing masks')

        logging.info(f'..removing cell masks with no DAPI')
        self.cell_masks_with_dapi = remove_cell_masks_with_no_dapi(self.cell_masks, self.dapi_masks)[0]

        logging.info(f'..discarding masks of size < 200')
        self.regionprops = get_regionprops(self.cell_masks)
        self.cell_masks_by_area = filter_cells_by_area(self.cell_masks_with_dapi, self.regionprops, min_cell_area=200)[0]

        logging.info(f'..discarding large, round masks (min size 1000, min excentricity 0.8')
        self.cell_masks_by_shape = filter_cells_by_shape(self.cell_masks_by_area, self.regionprops,
                                                         min_clump_area=1000, max_clump_eccentricity=0.8)[0]

        cellmaskfile = Path(self.savepath) / 'DIC_masks_pp.tif'
        io.imsave(cellmaskfile, self.cell_masks_by_area)

        logging.info(f'..computing expanded masks (by {self.experiment.parameters["expand_masks_by"]})')
        self.cell_masks_expanded = expand_masks(self.cell_masks_by_shape, nr_of_pixels=self.experiment.parameters["expand_masks_by"])

        cellmaskfile = Path(self.savepath) / 'DIC_masks_pp_expanded.tif'
        io.imsave(cellmaskfile, self.cell_masks_expanded)



    def find_focus(self):
        logging.info(f'finding focus')
        self.load_data('cells')
        self.load_data('mrna')
        self.selected_patch = find_high_density_patch(self.cell_masks, patch_size=self.experiment.parameters['patch_size'])

        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.find_focus_channel(ch)


    def find_focus_channel(self, ch):
        mrna_data = self.mrna.get(ch)['aligned']
        self.results[ch] = {
            'colormap': self.experiment.channels[ch]['colormap']
        }

        img_patch = mrna_data[:,
                    self.selected_patch[0]:self.selected_patch[0] + self.experiment.parameters['patch_size'][0],
                    self.selected_patch[1]:self.selected_patch[1] + self.experiment.parameters['patch_size'][1]
                    ]
        focus = compute_focus(img_patch)
        projected_focus = np.max(focus, axis=(1, 2))
        projected_focus_smoothed = savgol_filter(projected_focus, 16, 2, 0)
        ifx_1, ifx_2 = find_in_focus_indices(projected_focus_smoothed)

        if ifx_1 < 0 or ifx_2 > mrna_data.shape[0]:
            logging.warning(f'....focus detection: max focus is too close to highest or lowest slice')
        ifx_1 = int(max(ifx_1, 0))
        ifx_2 = int(min(ifx_2, mrna_data.shape[0]))

        self.results[ch].update({
            'z_max_focus': int(np.argmax(projected_focus_smoothed)),
            'ifx_1': ifx_1,
            'ifx_2': ifx_2
        })
        logging.info(f'..channel {ch}: [{ifx_1}, {ifx_2}], max focus slice {self.results[ch]["z_max_focus"]}')


    def filter(self):
        logging.info(f'filtering: remove_background_gaussian')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.filter_channel(ch)


    def filter_channel(self, ch):
        data = self.mrna.get(ch)
        data['filtered'] = remove_background_gaussian(data['aligned'], sigma=self.experiment.parameters['sigma'])
        logging.info(f'..channel {ch}')


    def detect_spots(self):
        logging.info(f'detecting spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.detect_spots_channel(ch)


    def detect_spots_channel(self, ch):
        self.load_data(ch)

        mrna_data = self.mrna.get(ch)['aligned']
        mrna_filtered = self.mrna.get(ch)['filtered']
        ifx_1, ifx_2 = self.results[ch]['ifx_1'], self.results[ch]['ifx_2']
        mrna_filtered_selected = mrna_filtered[ifx_1:ifx_2, ...]
        spots, threshold = detect_spots(
            mrna_filtered_selected,
            threshold=self.experiment.channels[ch]['threshold'],
            voxel_size=self.experiment.parameters['scale'],
            spot_radius=self.experiment.parameters['spot_radius'],
            return_threshold=True
        )
        self.results[ch]['threshold'] = threshold

        # always elegant:
        filtered_padded_intensities = np.concatenate((np.zeros([ifx_1, mrna_data.shape[1], mrna_data.shape[2]]).astype(
            'uint16'), mrna_filtered_selected, np.zeros(
            [mrna_data.shape[0] - ifx_2, mrna_data.shape[1], mrna_data.shape[2]]).astype('uint16')), axis=0)
        np.save(Path(self.savepath) / f'{ch}_filtered', filtered_padded_intensities)

        # restore z-level
        spots[:, 0] = spots[:, 0] + ifx_1

        # adjustable out-of focus filtering
        #  we remove the bottom two slices because their detected spots look like noise:
        spots = spots[spots[:, 0] > ifx_1 + 2]

        spot_intensities = np.resize(np.array([mrna_data[s[0], s[1], s[2]] for s in spots]), (len(spots), 1))
        filtered_spot_intensities = np.resize(np.array([filtered_padded_intensities[s[0], s[1], s[2]] for s in spots]),
                                              (len(spots), 1))
        # should we use expanded cell masks here?
        labels = [self.cell_masks_expanded[y, x] for (y, x) in spots[:, 1:3]]
        spots_with_intensities = np.concatenate(
            (spots, spot_intensities, filtered_spot_intensities, np.array(labels).reshape((len(labels), 1))), axis=1)
        np.save(Path(self.savepath) / f'{ch}_spots.npy', spots_with_intensities)
        self.mrna.get(ch)['spots'] = spots_with_intensities
        self.results[ch]['nr_spots'] = len(spots)
        logging.info(f'..channel {ch}: found {len(spots)} spots, threshold={threshold}')


    def decompose_spots(self):
        logging.info(f'decomposing spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.decompose_spots_channel(ch)


    def decompose_spots_channel(self, ch):
        mrna_data = self.mrna.get(ch)['aligned']
        ifx_1, ifx_2 = self.results[ch]['ifx_1'], self.results[ch]['ifx_2']
        mrna_data_selected = mrna_data[ifx_1:ifx_2, ...]
        spots = self.mrna.get(ch)['spots'][:, 0:3]

        # we should probably only do spot decomposition on spots that are in cells
        decomp_spots, dense_regions, reference_spot = decompose_dense(
            mrna_data_selected,
            spots,
            voxel_size=self.experiment.parameters['scale'],
            spot_radius=self.experiment.parameters['spot_radius'],
            alpha=0.5,  # alpha impacts the number of spots per candidate region
            beta=2,  # beta impacts the number of candidate regions to decompose
            gamma=1  # gamma the filtering step to denoise the image
        )

        self.mrna.get(ch)['decomposed_spots'] = decomp_spots
        self.mrna.get(ch)['dense_regions'] = dense_regions
        self.mrna.get(ch)['reference_spot'] = reference_spot

        np.save(Path(self.savepath) / f'{ch}_decomposed_spots.npy', decomp_spots)
        np.save(Path(self.savepath) / f'{ch}_dense_regions.npy', dense_regions)
        io.imsave(Path(self.savepath) / f'{ch}_reference_spot.tif', reference_spot)

        logging.info(f'..channel {ch}: found {len(dense_regions)} dense regions, {len(decomp_spots)} decomposed spots (from {len(spots)}')



    def assign_spots(self):
        self.counts = ''
        self.histograms = ''
        logging.info(f'assign spots')
        for ch in self.mrna.keys():
            if ch != 'DAPI':
                self.assign_spots_channel(ch)


    def assign_spots_channel(self, ch):
        # reload these as changes could have been made to the cell and nuclear masks.
        self.cell_masks = io.imread(Path(self.savepath) / f'DIC_masks_pp.tif')
        self.cell_masks_expanded = io.imread(Path(self.savepath) / f'DIC_masks_pp_expanded.tif')
        self.dapi_masks = io.imread(Path(self.savepath) / f'DAPI_masks.tif')

        #load spot data 
        spot_data = self.mrna.get(ch)['spots'][:, 0:3]
        dense_data = self.mrna.get(ch)['dense_regions']

        df = spot_assignment(self.cell_masks, self.cell_masks_expanded, self.dapi_masks, spot_data, dense_data)

        df.rename(columns={'label': 'image_cell_id'}, inplace=True)
        cell_columns = ['image_cell_id',
                        'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 'area', 'eccentricity', 'axis_minor_length',
                        'axis_major_length', 'orientation', 'perimeter', 'solidity',
                        'bbox-0_expanded', 'bbox-1_expanded', 'bbox-2_expanded', 'bbox-3_expanded', 'area_expanded',
                        'eccentricity_expanded', 'axis_minor_length_expanded', 'axis_major_length_expanded',
                        'orientation_expanded', 'perimeter_expanded', 'solidity_expanded',
                        'nuclei']
        df_cells = df[cell_columns]
        #rna_columns = ['image_cell_id', 'spots', 'dense_regions', 'decomposed_RNAs', 'tx', 'nascent_RNAs', 'total_RNAs']
        #df_rnas = df.loc[:, rna_columns]
        # df_rnas['mrna'] = ch
        df_cells.to_csv(Path(self.savepath) / f'{ch}.csv', index=False)
        logging.info(f'..channel {ch}: saving results to {self.savepath}/{ch}.csv')




