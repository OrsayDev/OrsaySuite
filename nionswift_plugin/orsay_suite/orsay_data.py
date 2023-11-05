import os, logging, inspect

import numpy
from nion.data import Calibration
from nion.swift.model import DataItem
from nion.swift.model import Utility
from nion.data import DataAndMetadata
import hyperspy.api as hs

from . import read_data_gain

__author__ = "Yves Auad"

def get_source_code(func):
    source_lines, _ = inspect.getsourcelines(func)
    source_code = ''.join(source_lines)
    return source_code

class HspySignal1D:
    def __init__(self, di):
        self.di = di
        new_data = numpy.copy(di.data)
        self.hspy_gd = hs.signals.Signal1D(new_data)
        self.hspy_gd.set_signal_type("EELS")

        self.signal_calib = di.dimensional_calibrations[len(di.dimensional_calibrations)-1]
        self.signal_size = di.data.shape[-1]
        self.nav_len = len(self.hspy_gd.data.shape) - 1

        for index in range(len(di.dimensional_calibrations)):
            self.hspy_gd.axes_manager[index].offset = di.dimensional_calibrations[index].offset
            self.hspy_gd.axes_manager[index].scale = di.dimensional_calibrations[index].scale
            self.hspy_gd.axes_manager[index].units = di.dimensional_calibrations[index].units

    #Methods that modify the hyperspy object
    def flip(self) -> None:
        signal_axis = len(self.hspy_gd.data.shape)-1
        self.hspy_gd.data = numpy.flip(self.hspy_gd.data, axis=signal_axis)

    def interpolate(self) -> None:
        self.hspy_gd.interpolate_in_between(256 - 1, 256 + 1, show_progressbar=False)
        self.hspy_gd.interpolate_in_between(256*2 - 1, 256*2 + 1, show_progressbar=False)
        self.hspy_gd.interpolate_in_between(256*3 - 1, 256*3 + 1, show_progressbar=False)

    def rebin(self, scale) -> None:
        if self.nav_len == 0:
            self.hspy_gd = self.hspy_gd.rebin(scale=[scale[2]])
        elif self.nav_len == 1:
            self.hspy_gd = self.hspy_gd.rebin(scale=[scale[1], scale[2]])
        elif self.nav_len == 2:
            self.hspy_gd = self.hspy_gd.rebin(scale=scale)

    def align_zlp_signal_range(self, range) -> None:
        r1 = self._rel_to_abs(range[0])
        r2 = self._rel_to_abs(range[1])
        self.hspy_gd.align_zero_loss_peak(subpixel=False, show_progressbar=False, signal_range=[r1, r2])

    def _get_data(self, temp_data, prefix, is_sequence = False, collection_dimension = None, datum_dimension = None) -> DataItem.DataItem:
        timezone = Utility.get_local_timezone()
        timezone_offset = Utility.TimezoneMinutesToStringConverter().convert(Utility.local_utcoffset_minutes())
        calibration = Calibration.Calibration()
        dimensional_calibrations = [Calibration.Calibration() for _ in range(len(temp_data.data.shape))]
        for index in range(len(temp_data.data.shape)):
            dimensional_calibrations[index].offset = temp_data.axes_manager[index].offset
            dimensional_calibrations[index].scale = temp_data.axes_manager[index].scale
            dimensional_calibrations[index].units = temp_data.axes_manager[index].units

        if collection_dimension == None or datum_dimension == None:
            data_descriptor = None
        else:
            data_descriptor = DataAndMetadata.DataDescriptor(is_sequence, collection_dimension, datum_dimension)
        xdata = DataAndMetadata.new_data_and_metadata(temp_data.data, calibration, dimensional_calibrations,
                                                      data_descriptor=data_descriptor,
                                                      metadata=self.di.metadata,
                                                      timezone=timezone, timezone_offset=timezone_offset)
        data_item = DataItem.DataItem()
        data_item.set_xdata(xdata)
        data_item.title = prefix + self.di.title
        data_item.description = self.di.description
        data_item.caption = self.di.caption

        #logging.info(f'***HSPY***: The axes of the returned data is {temp_data.axes_manager}.')

        return data_item

    def get_di(self, inav=None, isig=None, sum_inav=None, sum_isig=None):
        temp_data = self.hspy_gd
        nav_len = len(temp_data.data.shape)-1
        if inav is not None:
            assert nav_len == len(inav)
            if nav_len == 0:
                pass
            elif nav_len == 1:
                temp_data = temp_data.inav[inav[0]: inav[1]]
            elif nav_len == 2:
                temp_data = temp_data.inav[inav[0][0]: inav[0][1], inav[1][0]: inav[1][1]]
        if isig is not None:
            temp_data = temp_data.isig[isig[0]: isig[1]]
        if sum_inav:
            temp_data = temp_data.sum(axis=0)
        if sum_isig:
            temp_data = temp_data.sum(axis=1)

        return self._get_data(temp_data, 'processed_')

    def remove_background(self, range, which) -> None:
        r1 = self._rel_to_abs(range[0])
        r2 = self._rel_to_abs(range[1])
        self.hspy_gd = self.hspy_gd.remove_background(signal_range=(r1, r2), background_type=which)

    def plot_gaussian(self, range: tuple) -> list:
        r1 = self._rel_to_abs(range[0])
        r2 = self._rel_to_abs(range[1])

        m = self.hspy_gd.create_model()
        m.set_signal_range(r1, r2)
        gaussian = hs.model.components1D.Gaussian()
        gaussian.centre.value = (r1+r2)/2
        gaussian.centre.bmin = r1
        gaussian.centre.bmax = r2
        gaussian.A.value = numpy.max(self.hspy_gd.isig[r1:r2].data)
        m.append(gaussian)
        m[0].active = False
        m.assign_current_values_to_all()
        m.multifit(bounded=False, show_progressbar=False)
        m[1].print_current_values()

        return [self._get_data(m.as_signal(show_progressbar=False).isig[r1:r2], 'gaussian_fit_'),
                self._get_data(gaussian.A.as_signal(), 'gaussian_fit_amplitude_'),
                self._get_data(gaussian.centre.as_signal(), 'gaussian_fit_centre_'),
                self._get_data(gaussian.sigma.as_signal(), 'gaussian_fit_sigma_')]

    def plot_lorentzian(self, range: tuple) -> list:
        r1 = self._rel_to_abs(range[0])
        r2 = self._rel_to_abs(range[1])

        m = self.hspy_gd.create_model()
        m.set_signal_range(r1, r2)
        lor = hs.model.components1D.Lorentzian()
        lor.centre.value = (r1 + r2) / 2
        lor.centre.bmin = r1
        lor.centre.bmax = r2
        lor.A.value = numpy.max(self.hspy_gd.isig[r1:r2].data)
        m.append(lor)
        m[0].active = False
        m.assign_current_values_to_all()
        m.multifit(bounded=False, show_progressbar=False)
        m[1].print_current_values()

        return [self._get_data(m.as_signal(show_progressbar=False).isig[r1:r2], 'lorentzian_fit_')]

    def signal_decomposition(self, range=[0, 1], components=3, mask=True) -> list:

        if mask:
            r1 = self._rel_to_abs(range[0])
            r2 = self._rel_to_abs(range[1])

            signal_mask2 = self.hspy_gd._get_signal_signal(dtype='bool')
            signal_mask2.isig[r1: r2] = True
            self.hspy_gd.decomposition(True, svd_solver='full', signal_mask=signal_mask2)
        else:
            self.hspy_gd.decomposition(True)  # , svd_solver = 'full')

        variance = self.hspy_gd.get_explained_variance_ratio().isig[0:50]
        spim_dec = self.hspy_gd.get_decomposition_model(components)
        factors = hs.signals.Signal1D(self.hspy_gd.learning_results.factors[:, :components].reshape(self.signal_size, components).transpose())
        shape_loading = self.hspy_gd.axes_manager._navigation_shape_in_array
        loadings_stacked = hs.signals.Signal1D(
            self.hspy_gd.learning_results.loadings[:, :components].reshape(shape_loading[0], shape_loading[1], components).transpose())

        # Calibrate factors
        factors.axes_manager[1].name = "Energy"
        factors.axes_manager[1].scale = self.hspy_gd.axes_manager[2].scale
        factors.axes_manager[1].offset = self.hspy_gd.axes_manager[2].offset
        factors.axes_manager[1].units = self.hspy_gd.axes_manager[2].units

        # Calibrate loadings
        loadings_stacked.axes_manager[1].name = "Width"
        loadings_stacked.axes_manager[1].scale = self.hspy_gd.axes_manager[0].scale
        loadings_stacked.axes_manager[1].offset = self.hspy_gd.axes_manager[0].offset
        loadings_stacked.axes_manager[1].units = self.hspy_gd.axes_manager[0].units

        loadings_stacked.axes_manager[2].name = "Height"
        loadings_stacked.axes_manager[2].scale = self.hspy_gd.axes_manager[1].scale
        loadings_stacked.axes_manager[2].offset = self.hspy_gd.axes_manager[1].offset
        loadings_stacked.axes_manager[2].units = self.hspy_gd.axes_manager[1].units

        return [
            self._get_data(variance, 'PCAvar' + str(components) + '_'),
            self._get_data(spim_dec, 'PCA_SpimFiltered' + str(components) + '_'),
            self._get_data(factors, 'PCA_factors' + str(components) + '_', is_sequence=True, collection_dimension=0, datum_dimension=1),
            self._get_data(loadings_stacked, 'PCA_loadings' + str(components) + '_', is_sequence=True, collection_dimension=0, datum_dimension=2)
        ]

    def _rel_to_abs(self, val):
        return self.signal_calib.offset + val*self.signal_calib.scale*self.signal_size

    def deconvolution(self, psf, type, interactions):
        # if type=='Fourier log':
        #     self.hspy_gd.fourier_log_deconvolution(psf.hspy_gd)
        # elif type=='Fourier ratio':
        #     self.hspy_gd.fourier_ratio_deconvolution(psf.hspy_gd)
        if type=='Richardson lucy':
            data = self.hspy_gd.richardson_lucy_deconvolution(psf.hspy_gd, interactions, show_progressbar=False)
        return self._get_data(data, 'RLdec' + str(interactions) + '_')

    def _pixel_filler_Datum1(self, pixel, corrected_data, raw_data):
        # Added by ltizei. This is used to complete missing pixels in the Medipix detector junctions. This will
        # be used in the interpolate function
        average1 = numpy.average(raw_data[..., pixel - 1])  # "last" normal pixel
        average2 = numpy.average(raw_data[..., pixel + 2])  # next normal pixel
        diff = average1 - average2

        cut1 = raw_data[..., pixel]
        cut2 = raw_data[..., pixel + 1]

        for i in range(5):
            corrected_data[..., pixel + i] = (1 - i / 4) * cut1 + (i / 4) * cut2
            average_correction = numpy.mean(corrected_data[..., pixel + i:pixel + 1 + i])
            corrected_data[..., pixel + i] = corrected_data[..., pixel + i] * (average1 - ((i + 1) / 5) * diff) / (
                average_correction)

        return corrected_data


    def detector_junctions(self):
        data_shape = self.hspy_gd.data.shape
        raw_data = self.hspy_gd.data
        # parameters
        pixel1 = 255  # intense pixels 255 and 256
        pixel2 = 514  # intense pixels 511 and 512
        pixel3 = 773  # intense pixels 768 and 769
        pixel_list = [pixel1, pixel2, pixel3]

        data_shape_mod = numpy.array(data_shape)
        data_shape_mod[-1] = data_shape_mod[-1] + 9
        corrected_data = numpy.zeros((data_shape_mod))

        # Fill up the correct data, already present
        corrected_data[..., 0:255] = raw_data[...,
                                     0:255]  # data up to pixel1 is unchanged, 3pixels shift with 5 zeros afterwards
        corrected_data[..., 260:514] = raw_data[..., 257:511]  # data copied and shifted by 3 pixels, with 5 zeros
        corrected_data[..., 519:773] = raw_data[..., 513:767]  # data copied and shifted by 3 pixels, with 5 zeros
        corrected_data[..., 778:] = raw_data[..., 769:]

        for pixel in pixel_list:
            corrected_data = self._pixel_filler_Datum1(pixel, corrected_data, raw_data)

        corrected_data_hs = hs.signals.Signal1D(corrected_data)
        corrected_data_hs.set_signal_type("EELS")

        for index in range(len(corrected_data.shape)):
            corrected_data_hs.axes_manager[index].offset = self.hspy_gd.axes_manager[index].offset
            corrected_data_hs.axes_manager[index].scale = self.hspy_gd.axes_manager[index].scale
            corrected_data_hs.axes_manager[index].units = self.hspy_gd.axes_manager[index].units

        return self._get_data(corrected_data_hs, 'Corrected ')

    def correct_gain_hs(self, ht, threshold, gain_roi):
        #This will need to change to a "setup" folder
        self.__gain_file = read_data_gain.FileManager(ht, threshold)

        gain_signal_mean = numpy.mean(self.__gain_file.gain.data)
        vertical_bin =gain_roi[3] - gain_roi[1]
        gain_signal = self.__gain_file.gain.isig[gain_roi[0]:gain_roi[2], gain_roi[1]:gain_roi[3]].sum(axis=1)

        self.hspy_gd = self.hspy_gd*gain_signal_mean*vertical_bin/(gain_signal)