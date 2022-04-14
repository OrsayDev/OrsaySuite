import numpy
from nion.data import Calibration
from nion.swift.model import DataItem
from nion.swift.model import Utility
from nion.data import DataAndMetadata
import hyperspy.api as hs

import logging

__author__ = "Yves Auad"

class HspySignal1D:
    def __init__(self, di):
        self.di = di
        new_data = numpy.copy(di.data)
        self.hspy_gd = hs.signals.Signal1D(new_data)
        self.hspy_gd.set_signal_type("EELS")

        self.signal_calib = di.dimensional_calibrations[len(di.dimensional_calibrations)-1]
        self.signal_size = di.data.shape[-1]

        for index in range(len(di.dimensional_calibrations)):
            self.hspy_gd.axes_manager[index].offset = di.dimensional_calibrations[index].offset
            self.hspy_gd.axes_manager[index].scale = di.dimensional_calibrations[index].scale
            self.hspy_gd.axes_manager[index].units = di.dimensional_calibrations[index].units

        #logging.info(f'***HSPY***: The axes of the collected data is {self.hspy_gd.axes_manager}.')

    def flip(self, axis):
        self.hspy_gd.data = numpy.flip(self.hspy_gd.data, axis=axis)

    def interpolate(self):
        self.hspy_gd.interpolate_in_between(256 - 1, 256 + 1, show_progressbar=False)
        self.hspy_gd.interpolate_in_between(256*2 - 1, 256*2 + 1, show_progressbar=False)
        self.hspy_gd.interpolate_in_between(256*3 - 1, 256*3 + 1, show_progressbar=False)

    def rebin(self, scale):
        self.hspy_gd = self.hspy_gd.rebin(scale=scale)

    def align_zlp(self):
        self.hspy_gd.align_zero_loss_peak(show_progressbar=False)

    def get_data(self):
        return self.hspy_gd.data

    def get_attr(self, which):
        return self.di.description[which]

    def get_axes_offset(self, index):
        return self.hspy_gd.axes_manager[index].offset

    def get_axes_offset_all(self):
        return [self.hspy_gd.axes_manager[index].offset for index in range(len(self.hspy_gd.data.shape))]

    def get_axes_scale(self, index):
        return self.hspy_gd.axes_manager[index].scale

    def get_axes_scale_all(self):
        return [self.hspy_gd.axes_manager[index].scale for index in range(len(self.hspy_gd.data.shape))]

    def get_axes_units_all(self):
        return [self.hspy_gd.axes_manager[index].units for index in range(len(self.hspy_gd.data.shape))]

    def _get_data(self, temp_data, prefix):
        timezone = Utility.get_local_timezone()
        timezone_offset = Utility.TimezoneMinutesToStringConverter().convert(Utility.local_utcoffset_minutes())
        calibration = Calibration.Calibration()
        dimensional_calibrations = [Calibration.Calibration() for _ in range(len(temp_data.data.shape))]
        for index in range(len(temp_data.data.shape)):
            dimensional_calibrations[index].offset = temp_data.axes_manager[index].offset
            dimensional_calibrations[index].scale = temp_data.axes_manager[index].scale
            dimensional_calibrations[index].units = temp_data.axes_manager[index].units

        xdata = DataAndMetadata.new_data_and_metadata(temp_data.data, calibration, dimensional_calibrations,
                                                      metadata=self.di.metadata,
                                                      timezone=timezone, timezone_offset=timezone_offset)
        data_item = DataItem.DataItem()
        data_item.set_xdata(xdata)
        data_item.title = prefix + self.di.title
        data_item.description = self.di.description
        data_item.caption = self.di.caption

        logging.info(f'***HSPY***: The axes of the returned data is {temp_data.axes_manager}.')

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

    def remove_background(self, range, which):
        r1 = self._rel_to_abs(range[0])
        r2 = self._rel_to_abs(range[1])
        self.hspy_gd = self.hspy_gd.remove_background(signal_range=(r1, r2), background_type=which)

    def plot_gaussian(self, range):
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
        m.multifit(bounded=True, show_progressbar=False)
        m[1].print_current_values()

        return self._get_data(m.as_signal(show_progressbar=False).isig[r1:r2], 'gaussian_fit_')

    def plot_lorentzian(self, range):
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
        m.multifit(bounded=True, show_progressbar=False)
        m[1].print_current_values()

        return self._get_data(m.as_signal().isig[r1:r2], 'lorentzian_fit_')

    def _rel_to_abs(self, val):
        return self.signal_calib.offset + val*self.signal_calib.scale*self.signal_size

class HspyGain(HspySignal1D):
    def __init__(self, di):
        super().__init__(di)

    def rebin(self):
        initial = self.hspy_gd.axes_manager[0].offset
        self.hspy_gd = self.hspy_gd.rebin(scale=[self.get_attr('averages'), 1])
        self.hspy_gd.axes_manager[0].offset = initial

    def get_gain_profile(self):
        return self.get_di(isig=[-2.4, -1.8], sum_isig=True)

    def get_gain_2d(self):
        return self.get_di(isig=[-2.4, -1.8])