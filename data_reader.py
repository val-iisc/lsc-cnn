"""
data_reader.py: code to use crowd counting datasets for training and testing.
Authors       : dbs, mns, svp
"""


import numpy as np
import cv2
import os
import random
import scipy.io
import pickle
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class DataReader:
    """
    Class to use crowd counting datasets for training and testing.
    Version: 1.0
    DataReader supports the following:
        ground truths: can create density or dot maps.
        testing: dense sampling of crops (with overlap) for evaluation and
            stitch it back to original sizel.
        training: extract random crops with flip augmentation.
    """

    def __init__(self, temporary_data_path):
        """
        Initiate the data_reader class

        Parameters
        ----------
        temporary_data_path: string
            Path to store processed dataset data.
        """
        self.dataset_ready = False
        try:
            with open(os.path.join(temporary_data_path,
                                   'meta_data.save'), 'rb') as fp:
                meta_data_dict = pickle.load(fp)
                self.__dict__.update(meta_data_dict)
                self.dataset_ready = True
                print('In data_reader.__init__: Meta data read.')
        except:
            print('In data_reader.__init__: Can\'t read meta data in ' \
                  '%s; call create_dataset_files.' % temporary_data_path)
        self.temporary_data_path = temporary_data_path
        self.train_iterator = None

    def create_dataset_files(self,
                             dataset_paths,
                             image_crop_size=224,
                             image_roi_size=80,
                             image_roi_stride=72,
                             prediction_downscale_factor=4,
                             valid_set_size=0,
                             prediction_sigma=0.0,
                             use_rgb=True,
                             image_scale_factor=1.0,
                             gt_roi_readout_function=None,
                             train_batch_size=4,
                             test_batch_size=16):
        """
        Create dataset processed dataset files for training and testing.
        The files are written to path given by `self.temporary_data_path`.

        Parameters
        ----------
        dataset_paths: Dict
            A dictionary with the keys 'train', 'test' or 'test_val'
            containing a list of paths of images, gt and optional roi.
            Format: {'train': [images, gt, <roi>],
                     'test': [images, gt, <roi>],
                     'test_val': [images, gt, <roi>]}
            A key with 'test' is processed as a test set.
        image_crop_size: int
            Size of the square image (image_crop) to be extracted from dataset
            images for training and testing. The value MUST BE multiple of 2.
            Ideally, the value should be less than size of dataset images.
        image_roi_size: int
            Size of the square image within image_crop (image_roi) to be
            considered for prediction; the remaining region in image_crop acts
            as context for the prediction model. image_roi is typically at the
            center of image_crop except at the borders of the image. The value
            MUST BE < `image_crop_size` & multiple of 2.
        image_roi_stride: int
            Specifies the stride with which the image_roi should be moved to
            sample densely for testing. The testing is done by extracting
            image_crops from the dataset images such that image_roi covers the
            entire image with an an overlap (`image_crop_size` -
            `image_roi_stride`). The value MUST BE < `image_crop_size` and
            multiple of 2.
        prediction_downscale_factor: int
            Scale factor specifying the size of square prediction map
            (pred_crop, typically a density map) in relation to the input. For
            instance, `prediction_downscale` = 4 means the size of input_crop
            is exactly 4 times that of pred_crop. The value MUST BE one of
            [1, 2, 4, 8, 16, 32].
        valid_set_size: int
            Number of images from train set to be randomly taken for
            validation. Value MUST BE < number of training images.
            Default is 0 and no validation set is created.
        prediction_sigma: float
            The sigma or variance of the Gaussian kernel used for creating
            density maps from dot annotations.
            Default is 0 and dot map is used.
        use_rgb: Bool
            If `True`, uses rgb images otherwise resorts to gray scale.
        image_scale_factor: float
            Indicates how much to downscale any dataset image (aspect ratio is
            maintained). MUST BE > 0.
        gt_roi_readout_function: function(paths) -> tuple[gt_points,
                                                          gt_roi_maps]
            A python function to read ground truth points and roi maps.
            Argument is `paths`: list[image_path: string,
                                      gt_path: string,
                                      gt_roi_path: string <optional>]
            MUST return a tuple of [gt_points: ndarray((N, 2)),
                                    gt_roi_maps: ndarray((gtH, gtW))],
            where (gtH, gtW) = (H, W) // self.prediction_downscale_factor.
            `gt_points` must contain the coordinates of the point annotations
            as gt_points[:, 0] -> x coordinates &
            gt_points[:, 1] -> y coordinates.
            `gt_roi_maps` can be None.
            Defaults to the function which reads ST crowd dataset.
            NOTE: This function is used only for creating dataset and hence
                  not stored with the meta data.
        train_batch_size: int
                          The default batch size used for training (< 256).
        test_batch_size: int
                         The default batch size used for testing (< 256).

        """
        self.dataset_paths = dataset_paths
        self.image_crop_size = image_crop_size
        self.image_roi_size = image_roi_size
        self.image_roi_stride = image_roi_stride
        assert(prediction_downscale_factor in [1, 2, 4, 8, 16, 32])
        self.prediction_downscale_factor =  prediction_downscale_factor
        self.prediction_crop_size = image_crop_size // prediction_downscale_factor
        self.prediction_roi_size = image_roi_size // prediction_downscale_factor
        self.prediction_roi_stride = image_roi_stride // prediction_downscale_factor
        self.rgb = use_rgb
        self.image_scale_factor = image_scale_factor
        self.gt_roi_readout_function = gt_roi_readout_function
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.prediction_sigma = prediction_sigma
        self.dataset_files = {}
        self.dump_paths = {}
        self.dump_files = {}

        if not os.path.isdir(self.temporary_data_path):
            os.makedirs(self.temporary_data_path)
            print('In data_reader.create_dataset_files: %s does not ' \
                  'exists; but created.' % self.temporary_data_path)

        if valid_set_size > 0:
            self.dataset_paths['test_valid'] = [None, None]

        for set_name, dataset_path in dataset_paths.items():
            path = os.path.join(self.temporary_data_path, set_name)
            if os.path.isdir(path):
                shutil.rmtree(path)
                print('In data_reader.create_dataset_files: Deleted old ' \
                      '%s.' % path)
            os.makedirs(path)
            print('In data_reader.create_dataset_files: %s created.' \
                  % path)
            images_path = dataset_path[0]
            if images_path is None:
                continue
            self.dataset_files[set_name] = [f \
                                            for f in sorted(os.listdir(images_path)) \
                                            if os.path.isfile(os.path.join(images_path, f))]
            self.dump_files[set_name] = self.dataset_files[set_name]

        if valid_set_size > 0:
            files = self.dataset_files['train']
            file_ids = random.sample(range(0, len(files)), valid_set_size)
            validation_files = [f for i, f in enumerate(files) \
                                if i in file_ids]
            train_files = [f for i, f in enumerate(files) \
                           if i not in file_ids]
            self.dataset_paths['test_valid'] = self.dataset_paths['train']
            self.dataset_files['test_valid'] = validation_files
            self.dump_files['test_valid'] = validation_files

        for set_name, dataset_path in dataset_paths.items():
            path = os.path.join(self.temporary_data_path, set_name)
            if 'test' in set_name:
                tmp = os.path.join(path, '0')
                os.makedirs(tmp)
                tmp2 = os.path.join(path, '1')
                os.makedirs(tmp2)
                self.dump_paths[set_name] = [tmp, tmp2]
                self._dump_all_test_images(set_name)
            elif 'train' in set_name:
                tmp = os.path.join(path, '0')
                os.makedirs(tmp)
                tmp2 = os.path.join(path, '1')
                os.makedirs(tmp2)
                self.dump_paths[set_name] = [tmp, tmp2]
                self._dump_prediction_maps(set_name)
            else:
                print('In data_reader.create_dataset_files: error in ' \
                      'dataset name %s; ignored.' % set_name)
                continue

        if valid_set_size > 0:
            self.dataset_files['train'] = train_files
            self.dataset_paths['train_full'] = self.dataset_paths['train']
            self.dataset_files['train_full'] = files
            self.dump_files['train_full'] = files
            self.dump_paths['train_full'] = self.dump_paths['train']

        if self.gt_roi_readout_function is not None:
            self.gt_roi_readout_function = -1
        with open(os.path.join(self.temporary_data_path,
                               'meta_data.save'), 'wb') as fp:
            pickle.dump(self.__dict__, fp, protocol=pickle.HIGHEST_PROTOCOL)
        self.dataset_ready = True

    def iterate_over_test_data(self,
                               test_function,
                               dataset_name='test'):
        """
        An iterator to run over images of test set and perform densely scanned
        evaluation.

        Parameters
        ----------
        test_function: function(ndarray img_batch,
                                ndarray gt_batch,
                                ndarray roi_batch) -> tuple[ndarray]
            A python function which is repeatedly called for model evaluation
            of crops from test image. The function arguments are:
             `img_batch` (image crop): ndarray((B, C, H, W)),
             `gt_batch` (corresponding ground truth):
                        ndarray(B, 1, H // self.prediction_downscale_factor,
                                      W // self.prediction_downscale_factor)),
             `roi_batch` (roi mask of same shape as `gt_batch`).
            The function can return tuple of arbitrary number of ndarrays,
            but MUST HAVE same shape as `gt_batch` except in dimension 1.
        dataset_name: string
            Name of the set ('test' or 'test_valid') for evaluation.

        Returns
        ----------
        An iterator which outputs a tuple of 3 items as:
            [a tuple of stitched outputs (of same shape as test set ground
             truth EXCEPT in dim 1) returned by `test_function`,
             image_path: string,
             ground truth map: ndarray()]

        Example Usage
        ----------
        for results, img_path, gt_map in _.iterate_over_test_data(test_fn)
            # process
        """
        dataset_paths = self.dataset_paths[dataset_name]
        files = self.dump_files[dataset_name]
        dump_path = self.dump_paths[dataset_name][0]

        for file_name in files:
            with open(os.path.join(dump_path, file_name), 'rb') as fp:
                crops = pickle.load(fp)
            """
            crops: a tuple of[images: ndarray((B, C, H, W)),
                              gt_prediction_maps: ndarray((B, C, gtH, gtW)),
                              roi_masks: ndarray((B, C, gtH, gtW)),
                              pred_map_roi_slices: ndarray((B, 4)),
                              pred_map_roi_relative_slices: ndarray((B, 4)),
                              overlap_count: ndarray((B, C, gtH, gtW)),
                              gt_map: ndarray((gtH, gtW)),
                              prediction_count: float]
            """
            pred_maps_full_size = self._test_one_image(crops, test_function)
            image_path = os.path.join(dataset_paths[0], file_name)
            yield pred_maps_full_size, image_path, crops[6]

    def train_get_batch(self, train_batch_size=None):
        """
        Returns a batch of randomly cropped images from train set
        (with flip augmentation).

        Parameters
        ----------
        train_batch_size: int
            Batch size value to override default setting.

        Returns
        ----------
        Tuple of [images: ndarray((B, C, H, W)),
                  gt_pred_maps: ndarray((B, 1, gtH, gtW)),
                  roi_masks: ndarray((B, 1, gtH, gtW))]
        where (gtH, gtW) = (H, W) // self.prediction_downscale_factor.
        """
        HEIGHT_IDX = 1
        WIDTH_IDX = 2
        dataset_name = 'train'
        files = self.dataset_files[dataset_name]
        dump_path = self.dump_paths[dataset_name][0]
        if train_batch_size is None:
            train_batch_size = self.train_batch_size

        if self.train_iterator is None or \
            (self.train_iterator + train_batch_size) > self.num_files_rounded:
            self.train_iterator = 0
            self.num_files_rounded = len(files) - \
                                        (len(files) % train_batch_size)
            self.file_ids = random.sample(range(0, len(files)),
                                          self.num_files_rounded)

        file_ids = self.file_ids[self.train_iterator: \
                                 self.train_iterator + train_batch_size]
        if len(file_ids) != train_batch_size:
            print(len(file_ids), train_batch_size, self.train_iterator)
        assert(len(file_ids) == train_batch_size)
        file_batch = [files[i] for i in file_ids]
        self.train_iterator += train_batch_size

        num_channels = 3 if self.rgb else 1
        images = np.empty((train_batch_size, num_channels, self.image_crop_size,
                           self.image_crop_size), dtype = np.float32)
        gt_pred_maps = np.empty((train_batch_size, 1,
                                 self.prediction_crop_size,
                                 self.prediction_crop_size), \
                                dtype = np.float32)
        roi_masks = np.zeros((train_batch_size, 1, self.prediction_crop_size,
                              self.prediction_crop_size), dtype = np.float32)
        flip_flags = np.random.randint(2, size = train_batch_size)

        for i, (file_name, flip_flag) in enumerate(zip(file_batch,
                                                       flip_flags)):
            with open(os.path.join(dump_path, file_name), 'rb') as fp:
                data = pickle.load(fp)
            y = np.random.randint(data[0].shape[HEIGHT_IDX])
            x = np.random.randint(data[0].shape[WIDTH_IDX])
            crop, _, _, _ = self._take_image_crop(data[0], y, x,
                                                  self.image_roi_size,
                                                  self.image_crop_size)
            if flip_flag == 1:
                crop = crop[:, :, : : -1]
            images[i, :, :, :] = crop
            y //= self.prediction_downscale_factor
            x //= self.prediction_downscale_factor
            crop, _,_, roi_rel_slice = self._take_image_crop(data[1], y, x,
                                                    self.prediction_roi_size,
                                                    self.prediction_crop_size)
            roi_masks[i, 0, roi_rel_slice[0]: roi_rel_slice[1],
                            roi_rel_slice[2]: roi_rel_slice[3]] = 1.0
            # for dataset gt roi
            if len(data) > 2:
                ds_gt_roi_crop, _, _, _ = self._take_image_crop(data[2], y, x,
                                                    self.prediction_roi_size,
                                                    self.prediction_crop_size)
                roi_masks[i] *= ds_gt_roi_crop
            if flip_flag == 1:
                crop = crop[:, :, : : -1]
                roi_masks[i, 0, :, :] = roi_masks[i, 0, :, : : -1]
            gt_pred_maps[i, :, :, :] = crop

        assert(np.all((0.0 <= images) * (images <= 255.0)))
        assert(np.all(np.isfinite(images)))
        assert(np.all(np.isfinite(gt_pred_maps)))
        assert(np.all(np.isfinite(roi_masks)))

        return images, gt_pred_maps, roi_masks


    ### ### ### Internal functions ### ### ###

    def _gaussian_kernel(self, sigma=1.0, kernel_size=None):
        '''
        Returns gaussian kernel if sigma > 0.0, otherwise dot kernel.
        '''
        if sigma <= 0.0:
            return np.array([[0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0]], dtype=np.float32)
        if kernel_size is None:
            kernel_size = int(3.0 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
            print('In data_reader.gaussian_kernel: Kernel size even; ' \
                  'increased by 1.')
        if kernel_size < 3:
            kernel_size = 3
            print('In data_reader.gaussian_kernel: Kernel size less than 3;' \
                  'set as 3.')
        tmp = np.arange((-kernel_size // 2) + 1.0, (kernel_size // 2) + 1.0)
        xx, yy = np.meshgrid(tmp, tmp)
        kernel = np.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2)))
        kernel_sum = np.sum(kernel)
        assert (kernel_sum > 1e-3)
        return kernel / kernel_sum

    def _create_heatmap(self, image_shape, heatmap_shape,
                       annotation_points, kernel):
        """
        Creates density map.
        annotation_points : ndarray Nx2,
                            annotation_points[:, 0] -> x coordinate
                            annotation_points[:, 1] -> y coordinate
        """
        assert (kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2
                and kernel.shape[0] > 1)
        indices = (annotation_points[:, 0] < image_shape[1]) & \
                  (annotation_points[:, 0] >= 0) & \
                  (annotation_points[:, 1] < image_shape[0]) & \
                  (annotation_points[:, 1] >= 0)
        annot_error_count = len(annotation_points)
        annotation_points = annotation_points[indices, :]

        hmap_height, hmap_width = heatmap_shape
        annotation_points[:, 0] *= (float(heatmap_shape[1]) / image_shape[1])
        annotation_points[:, 1] *= (float(heatmap_shape[0]) / image_shape[0])
        annotation_points = annotation_points.astype(np.int32)
        annot_error_count -= np.sum(indices)
        if annot_error_count:
            print('In data_reader.create_heatmap: Error in annotations; ' \
                  '%d point(s) skipped.' % annot_error_count)
        indices = (annotation_points[:, 0] >= heatmap_shape[1]) & \
                  (annotation_points[:, 0] < 0) & \
                  (annotation_points[:, 1] >= heatmap_shape[0]) & \
                  (annotation_points[:, 1] < 0)
        assert(np.sum(indices) == 0)

        prediction_map = np.zeros(heatmap_shape, dtype = np.float32)
        kernel_half_size = kernel.shape[0] // 2
        kernel_copy = np.empty_like(kernel)

        for x, y in annotation_points:
            y_start = y - kernel_half_size
            y_end = y_start + kernel.shape[0]
            x_start = x - kernel_half_size
            x_end = x_start + kernel.shape[1]
            kernel_copy[:] = kernel[:]
            kernel_tmp = kernel_copy
            if y_start < 0:
                i = -y_start
                kernel_tmp[i: 2 * i, :] += kernel_tmp[i - 1:: -1, :]
                kernel_tmp = kernel_tmp[i:, :]
                y_start = 0
            if x_start < 0:
                i = -x_start
                kernel_tmp[:, i: 2 * i] += kernel_tmp[:, i - 1:: -1]
                kernel_tmp = kernel_tmp[:, i:]
                x_start = 0
            if y_end > hmap_height:
                i = (hmap_height - y - 1) - kernel_half_size
                kernel_tmp[2 * i: i, :] += kernel_tmp[-1: i - 1: -1, :]
                kernel_tmp = kernel_tmp[: i, :]
                y_end = hmap_height
            if x_end > hmap_width:
                i = (hmap_width - x - 1) - kernel_half_size
                kernel_tmp[:, 2 * i: i] += kernel_tmp[:, -1: i - 1: -1]
                kernel_tmp = kernel_tmp[:, : i]
                x_end = hmap_width
            prediction_map[y_start: y_end, x_start: x_end] += kernel_tmp
        return prediction_map

    def _take_image_crop(self, image, y, x, roi_shape, crop_shape):
        """
        # All _size are sides of square.
        # (x, y) correspond to top-left corner of the roi.
        # ASSUMES: crop_size > roi_size; but DOES NOT CHECK.
        # image has to be (H, W, C)
        # Always return constant roi and crop sizes.
        """
        HEIGHT_IDX = 1
        WIDTH_IDX = 2
        if not isinstance(roi_shape, tuple):
            roi_shape = (roi_shape, roi_shape)
        if not isinstance(crop_shape, tuple):
            crop_shape = (crop_shape, crop_shape)
        crop_extension = ((crop_shape[0] - roi_shape[0]) // 2,
                          (crop_shape[1] - roi_shape[1]) // 2)

        assert (0 <= y < image.shape[HEIGHT_IDX] and
                0 <= x < image.shape[WIDTH_IDX] and
                roi_shape[0] <= crop_shape[0] and
                roi_shape[1] <= crop_shape[1] and ###? REDUNDANT AS BOTTOM
                crop_extension[0] >= 0 and
                crop_extension[1] >= 0 and
                crop_shape[0] <= image.shape[HEIGHT_IDX] and
                crop_shape[1] <= image.shape[WIDTH_IDX])  ###? CAN BE REMOVED

        roi_y_start = y
        roi_y_end = roi_y_start + roi_shape[0]
        roi_x_start = x
        roi_x_end = roi_x_start + roi_shape[1]
        if roi_y_end > image.shape[HEIGHT_IDX]:
            roi_y_end = image.shape[HEIGHT_IDX]
            roi_y_start = roi_y_end - roi_shape[0]
        if roi_x_end > image.shape[WIDTH_IDX]:
            roi_x_end = image.shape[WIDTH_IDX]
            roi_x_start = roi_x_end - roi_shape[1]

        crop_y_start = roi_y_start - crop_extension[0]
        crop_y_end = crop_y_start + crop_shape[0]
        crop_x_start = roi_x_start - crop_extension[1]
        crop_x_end = crop_x_start + crop_shape[1]
        if crop_y_start < 0:
            crop_y_start = 0
            crop_y_end = crop_y_start + crop_shape[0]
        if crop_x_start < 0:
            crop_x_start = 0
            crop_x_end = crop_x_start + crop_shape[1]
        if crop_y_end > image.shape[HEIGHT_IDX]:
            crop_y_end = image.shape[HEIGHT_IDX]
            crop_y_start = crop_y_end - crop_shape[0]
        if crop_x_end > image.shape[WIDTH_IDX]:
            crop_x_end = image.shape[WIDTH_IDX]
            crop_x_start = crop_x_end - crop_shape[1]

        crop = image[:, crop_y_start: crop_y_end, crop_x_start: crop_x_end]
        crop_slice = np.array([crop_y_start, crop_y_end,
                               crop_x_start, crop_x_end], dtype = np.int32)
        roi_slice = np.array([roi_y_start, roi_y_end, roi_x_start, roi_x_end],
                             dtype = np.int32)
        tmp1 = roi_y_start - crop_y_start
        tmp2 = roi_x_start - crop_x_start
        roi_relative_slice = np.array([tmp1, tmp1 + roi_shape[0],
                                       tmp2, tmp2 + roi_shape[1]],
                                      dtype = np.int32)
        # slices with respect to image
        return crop, crop_slice, roi_slice, roi_relative_slice

    def _get_one_image_test_crops(self, data):
        """
        # image -> (C, H, W)
        # gt -> (1, H, W)
        # data is list of [image, ground_truth_prediction_map, ground_truth_roi]

        Returns
        ----------
        A tuple of [images: ndarray((B, C, H, W)),
                    gt_prediction_maps: ndarray((B, C, gtH, gtW)),
                    roi_masks: ndarray((B, C, gtH, gtW)),
                    pred_map_roi_slices: ndarray((B, 4)),
                    pred_map_roi_relative_slices: ndarray((B, 4)),
                    overlap_count: ndarray((B, C, gtH, gtW)),
                    gt_map: ndarray((gtH, gtW)),
                    prediction_count: float]
        where (gtH, gtW) = (H, W) // self.prediction_downscale_factor.
        """
        HEIGHT_IDX = 1
        WIDTH_IDX = 2
        assert(data[0].shape[HEIGHT_IDX] >= self.image_crop_size \
                <= data[0].shape[WIDTH_IDX] and
               data[1].shape[HEIGHT_IDX] >= self.prediction_crop_size \
                <= data[1].shape[WIDTH_IDX])
        assert(self.image_crop_size >= self.image_roi_size and
               self.prediction_crop_size >= self.prediction_roi_size and
               self.image_roi_stride <= self.image_roi_size and
               self.prediction_roi_stride <= self.prediction_roi_size and
               self.image_roi_stride > 0 and self.prediction_roi_stride > 0)
        images = []
        gt_prediction_maps = []
        pred_map_roi_slices = []
        pred_map_roi_relative_slices = []
        roi_masks = []
        overlap_count = np.zeros((data[1].shape[HEIGHT_IDX],
                                  data[1].shape[WIDTH_IDX]),
                                 dtype=np.float32)

        def roi_iterator_variable_sized_roi(image_size, crop_size,
                                            roi_size, roi_stride):
            """
            Generates locations of the image_roi to cover the entire size with
            the specified stride.
            """
            i = 0
            roi_starts = []
            roi_sizes = []
            roi_extension = (crop_size - roi_size) // 2
            crop_end = i + roi_size + roi_extension
            assert (roi_extension > 0)

            if crop_size == image_size:
                return [0], [crop_size]

            while True:
                if i > 0 and ((i - roi_extension <= 0) or \
                                (crop_end >= image_size and \
                                (i - roi_extension + crop_size) > image_size)):
                    roi_sizes[-1] = min(i + roi_size, image_size) - roi_starts[-1]
                else:
                    roi_starts.append(i)
                    roi_sizes.append(roi_size)
                    crop_end = i + roi_size + roi_extension
                if (i + roi_size) >= image_size:
                    break
                i += roi_stride
            return roi_starts, roi_sizes

        roi_iterator = roi_iterator_variable_sized_roi

        image_roi_iters = [roi_iterator(data[0].shape[HEIGHT_IDX],
                                        self.image_crop_size,
                                        self.image_roi_size,
                                        self.image_roi_stride),
                           roi_iterator(data[0].shape[WIDTH_IDX],
                                        self.image_crop_size,
                                        self.image_roi_size,
                                        self.image_roi_stride)]
        prediction_roi_iters = [roi_iterator(data[1].shape[HEIGHT_IDX],
                                             self.prediction_crop_size,
                                             self.prediction_roi_size,
                                             self.prediction_roi_stride),
                                roi_iterator(data[1].shape[WIDTH_IDX],
                                             self.prediction_crop_size,
                                             self.prediction_roi_size,
                                             self.prediction_roi_stride)]

        assert(len(image_roi_iters[0][0]) > 0 and len(image_roi_iters[1][0]) > 0) ##? combine below?
        if len(image_roi_iters[0][0]) != len(prediction_roi_iters[0][0]) \
                or len(image_roi_iters[1][0]) != len(prediction_roi_iters[1][0]):
            print('In data_reader._get_one_image_test_crops: Error in iter;' \
                  ' value relations between image/roi sizes and strides.' \
                  'Exiting.')
            exit(1)

        for img_y, img_y_sz, gt_y, gt_y_sz in \
                zip(*(image_roi_iters[0] + prediction_roi_iters[0])):
            for img_x, img_x_sz, gt_x, gt_x_sz in \
                    zip(*(image_roi_iters[1] + prediction_roi_iters[1])):
                crop, _, _, _ = self._take_image_crop(data[0],
                                                      img_y, img_x,
                                                      (img_y_sz, img_x_sz),
                                                      self.image_crop_size)
                images.append(crop)
                crop, crop_slice, roi_slice, roi_relative_slice = \
                    self._take_image_crop(data[1],
                                          gt_y, gt_x,
                                          (gt_y_sz, gt_x_sz),
                                          self.prediction_crop_size)
                gt_prediction_maps.append(crop)
                pred_map_roi_slices.append(roi_slice)
                roi_mask = np.zeros((self.prediction_crop_size,
                                     self.prediction_crop_size),
                                    dtype=np.float32)
                roi_mask[roi_relative_slice[0]: roi_relative_slice[1],
                         roi_relative_slice[2]: roi_relative_slice[3]] = 1.0
                pred_map_roi_relative_slices.append(roi_relative_slice)
                if len(data) > 2:
                    crop_roi, crop_slice_roi, roi_slice_roi, roi_relative_slice_roi = \
                    self._take_image_crop(data[2],
                                          gt_y, gt_x,
                                          (gt_y_sz, gt_x_sz),
                                          self.prediction_crop_size)
                    roi_mask *= crop_roi[0]
                roi_masks.append(roi_mask)
                overlap_count[roi_slice[0]: roi_slice[1],
                              roi_slice[2]: roi_slice[3]] += 1

        if np.sum(overlap_count == 0) != 0:
            print('In data_reader._get_one_image_test_crops: Error in ;' \
                  'value relations between image/roi sizes and strides.' \
                  'Exiting.')
            exit(1)
        images = np.stack(images)
        gt_prediction_maps = np.stack(gt_prediction_maps)
        roi_masks = np.stack(roi_masks)
        pred_map_roi_slices = np.stack(pred_map_roi_slices)
        pred_map_roi_relative_slices = np.stack(pred_map_roi_relative_slices)
        prediction_count = np.sum(data[1])
        return (images, gt_prediction_maps, roi_masks,
                pred_map_roi_slices, pred_map_roi_relative_slices,
                overlap_count, data[1][0], prediction_count)

    def _test_one_image(self, crops, test_function):
        """
        Do overlapped testing of one image.

        Parameters
        ----------
        crops: tuple
            A tuple of [images: ndarray((B, C, H, W)),
                        gt_prediction_maps: ndarray((B, C, gtH, gtW)),
                        roi_masks: ndarray((B, C, gtH, gtW)),
                        pred_map_roi_slices: ndarray((B, 4)),
                        pred_map_roi_relative_slices: ndarray((B, 4)),
                        overlap_count: ndarray((B, C, gtH, gtW)),
                        gt_map: ndarray((gtH, gtW)),
                        prediction_count: float]
            where (gtH, gtW) = (H, W) // self.prediction_downscale_factor.
        test_function: function(ndarray img_batch,
                                ndarray gt_batch,
                                ndarray roi_batch) -> tuple[ndarray]
            A python function which is repeatedly called for model evaluation
            of crops from test image. The function arguments are:
             `img_batch` (image crop): ndarray((B, C, H, W)),
             `gt_batch` (corresponding ground truth):
                        ndarray(B, 1, H // self.prediction_downscale_factor,
                                      W // self.prediction_downscale_factor)),
             `roi_batch` (roi mask of same shape as `gt_batch`).
            The function can return tuple of arbitrary number of ndarrays,
            but MUST HAVE same shape as `gt_batch` except in dimension 1.

        Returns
        ----------
        A tuple of stitched outputs (of shape same as test set ground truth)
        returned by `test_function`.
        """
        for j in range(0, crops[0].shape[0], self.test_batch_size):
            current_slice = slice(j, j + self.test_batch_size)
            img_batch = crops[0][current_slice]
            gt_batch = crops[1][current_slice]
            roi_batch = crops[2][current_slice]
            roi_slice_batch = crops[3][current_slice]
            roi_relative_slice_batch = crops[4][current_slice]

            results = test_function(img_batch, gt_batch, roi_batch)
            assert(isinstance(results, tuple) and len(results) > 0)
            ##? SAFETY CHECK, CAN BE REMOVED.
            # for result in results:
            #     assert(result.shape == gt_batch.shape)
            try:
                predicted_maps_full_size
            except:
                predicted_maps_full_size = [np.zeros((pmap.shape[1], crops[6].shape[0], crops[6].shape[1])) for pmap in results]
            for batch_item in \
                    zip(roi_slice_batch, roi_relative_slice_batch, *results):
                roi_slice, roi_rel_slice = batch_item[: 2]
                pred_maps = batch_item[2: ] # same size as `gt_batch`
                
                for (pmap_full_size, pred_map) in zip(predicted_maps_full_size,
                                                    pred_maps):
                    pmap_full_size[:, roi_slice[0]: roi_slice[1],
                                      roi_slice[2]: roi_slice[3]] \
                        += pred_map[:, roi_rel_slice[0]: roi_rel_slice[1],
                                       roi_rel_slice[2]: roi_rel_slice[3]]

        predicted_maps_full_size = [pmap_full_size / crops[5]
                                    for pmap_full_size \
                                        in predicted_maps_full_size]
        return predicted_maps_full_size

    def _read_image_and_gt_prediction(self, paths, file_name, kernel = None):
        image = cv2.imread(os.path.join(paths[0], file_name))
        image = cv2.resize(image,
                           (int(image.shape[1] / self.image_scale_factor),
                            int(image.shape[0] / self.image_scale_factor)))
        assert (np.all(np.isfinite(image)))
        if self.rgb:
            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            assert (len(image.shape) == 3)
        else:
            if len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            assert(len(image.shape) == 2)
        orig_image_shape = image.shape
        if image.shape[0] == self.image_crop_size or \
                image.shape[1] == self.image_crop_size:
            print('In data_reader._read_image_and_gt_prediction: Image ' \
                  'side same as crop_size.')
        if image.shape[0] < self.image_crop_size or \
                image.shape[1] < self.image_crop_size:
            height = max(image.shape[0], self.image_crop_size)
            width = max(image.shape[1], self.image_crop_size)
            if image.shape[0] <= image.shape[1]:
                width = int((float(height * image.shape[1]) \
                             / image.shape[0]) + 0.5)
            else:
                height = int((float(width * image.shape[0]) \
                              / image.shape[1]) + 0.5)
            print('In data_reader._read_image_and_gt_prediction: Image ' \
                  '(%d, %d) resized to small size (%d, %d).' % \
                  (image.shape[0], image.shape[1], height, width))
            image = cv2.resize(src = image, dsize = (width, height))
        if kernel is None or len(paths) == 1:
            if self.rgb:
                image = image.transpose((2, 0, 1)).astype(np.float32)
            return image

        if self.gt_roi_readout_function is None:
            # ASSUMES: ST PartA Dataset
            tmp, _ = os.path.splitext(file_name)
            data_mat = scipy.io.loadmat(os.path.join(paths[1],
                                                     'GT_' + tmp + '.mat'))
            gt_annotation_points = data_mat['image_info'][0, 0]['location'][0, 0]
            gt_annotation_points -= 1  # MATLAB INDICES
            gt_roi_map = None
        else:
            gt_annotation_points, gt_roi_map = \
                                        self.gt_roi_readout_function(paths)
        pred_map_shape = (int(np.ceil(float(image.shape[0]) \
                                      / self.prediction_downscale_factor)),
                          int(np.ceil(float(image.shape[1]) \
                                      / self.prediction_downscale_factor)))
        gt_annotation_points = gt_annotation_points / self.image_scale_factor
        gt_pred_map = self._create_heatmap(orig_image_shape, pred_map_shape,
                                          gt_annotation_points, kernel)

        if self.rgb:
            image = image.transpose((2, 0, 1)).astype(np.float32)
        else:
            image = image[np.newaxis, ...].astype(np.float32)

        gt_pred_map = gt_pred_map[np.newaxis, ...]  # (1, gtH, gtW)
        if gt_roi_map is None:
            return image, gt_pred_map
        else:
            gt_roi_map = cv2.resize(src=gt_roi_map,
                                    dsize=(gt_pred_map[2], gt_pred_map[1]))
            gt_roi_map = gt_roi_map[np.newaxis, ...]  # (1, gtH, gtW)
            return image, gt_pred_map, gt_roi_map

    def _dump_all_test_images(self, dataset_name):
        files = self.dataset_files[dataset_name]
        paths = self.dataset_paths[dataset_name]
        dump_paths = self.dump_paths[dataset_name]
        kernel = self._gaussian_kernel(self.prediction_sigma)

        for file_name in files:
            print('Processing', file_name, '...')
            data = self._read_image_and_gt_prediction(paths, file_name, kernel)
            crops = self._get_one_image_test_crops(data)
            with open(os.path.join(dump_paths[0], file_name), 'wb') as fp:
                pickle.dump(crops, fp, protocol=pickle.HIGHEST_PROTOCOL)

            def test_function(img_batch, gt_batch, roi_batch):
                return (gt_batch, )

            gt_pred_count = crops[-1]
            pred_maps = self._test_one_image(crops, test_function)
            count = np.sum(pred_maps[0])
            count_error = np.abs(count - gt_pred_count)
            if len(data) > 2:
                maps = [(data[0][0], {'cmap': 'gray'}),
                        (data[1][0], {'cmap': 'jet'}),
                        (data[2][0], {'cmap': 'jet'}),
                        (pred_maps[0], {'cmap': 'jet'})]
            else:
                maps = [(data[0][0], {'cmap': 'gray'}),
                        (data[1][0], {'cmap': 'jet'}),
                        (pred_maps[0][0], {'cmap': 'jet'})]
            if self.rgb:
                maps[0] = (data[0], {})
            title = 'Actual: %g, Predicted: %g' % (gt_pred_count, count)
            graph_path = os.path.join(dump_paths[1], file_name + '.jpg')
            self._print_graph(maps, title, graph_path)
            #assert (count_error < 1e-3)
            assert(count_error == 0.0)
        print('Done test dumping.')

    def _dump_prediction_maps(self, dataset_name='train'):
        files = self.dataset_files[dataset_name]
        paths = self.dataset_paths[dataset_name]
        dump_paths = self.dump_paths[dataset_name]
        kernel = self._gaussian_kernel(self.prediction_sigma)

        for file_name in files:
            print('Processing', file_name, '...')
            data = self._read_image_and_gt_prediction(paths, file_name, kernel)
            with open(os.path.join(dump_paths[0], file_name), 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

            gt_pred_count = np.sum(data[1])
            if len(data) > 2:
                maps = [(data[0][0], {'cmap': 'gray'}),
                        (data[1][0], {'cmap': 'jet'}),
                        (data[2][0], {'cmap': 'jet'})]
            else:
                maps = [(data[0][0], {'cmap': 'gray'}),
                        (data[1][0], {'cmap': 'jet'})]
            if self.rgb:
                maps[0] = (data[0], {})
            title = 'Actual: %g.' % gt_pred_count
            graph_path = os.path.join(dump_paths[1], file_name + '.jpg')
            self._print_graph(maps, title, graph_path)
        print('Done dumping pred maps.')

    def _print_graph(self, maps, title, save_path):
        fig = plt.figure()
        st = fig.suptitle(title)
        for i, (map, args) in enumerate(maps):
            plt.subplot(1, len(maps), i + 1)
            if len(map.shape) > 2 and map.shape[0] == 3:
                plt.imshow(map.transpose((1, 2, 0)).astype(np.uint8),
                           aspect='equal', **args)
            else:
                plt.imshow(map, aspect='equal', **args)
            plt.axis('off')
        plt.savefig(save_path + ".png", bbox_inches='tight', pad_inches = 0)
        fig.clf()
        plt.clf()
        plt.close()

