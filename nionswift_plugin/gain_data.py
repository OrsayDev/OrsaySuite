import numpy
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
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
        self.hspy_gd = hs.signals.Signal1D(di.data)
        self.hspy_gd.set_signal_type("EELS")

        self.signal_calib = di.dimensional_calibrations[len(di.dimensional_calibrations)-1]
        self.signal_size = di.data.shape[-1]

        for index in range(len(di.dimensional_calibrations)):
            self.hspy_gd.axes_manager[index].offset = di.dimensional_calibrations[index].offset
            self.hspy_gd.axes_manager[index].scale = di.dimensional_calibrations[index].scale
            self.hspy_gd.axes_manager[index].units = di.dimensional_calibrations[index].units

        logging.info(f'***HSPY***: The axes of the collected data is {self.hspy_gd.axes_manager}.')

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


    def get_11_di(self, inav=None, isig=None, sum_inav=None, sum_isig=None):
        temp_data = self.hspy_gd
        if inav is not None:
            temp_data = temp_data.inav[inav[0]: inav[1]]
        if isig is not None:
            temp_data = temp_data.isig[isig[0]: isig[1]]
        if sum_inav:
            temp_data = temp_data.sum(axis=0)
        if sum_isig:
            temp_data = temp_data.sum(axis=1)

        return self._get_data(temp_data, 'new_')

    def plot_gaussian(self, range):
        m = self.hspy_gd.create_model()
        m.set_signal_range(self._rel_to_abs(range[0]), self._rel_to_abs(range[1]))
        gaussian = hs.model.components1D.Gaussian()
        gaussian.centre.value = (self._rel_to_abs(range[0]) + self._rel_to_abs(range[1]))/2
        m.append(gaussian)
        m.fit(bounded=True)
        m.print_current_values()

        return self._get_data(m.as_signal(), 'fit_')

    def _rel_to_abs(self, val):
        return self.signal_calib.offset + val*self.signal_calib.scale*self.signal_size

class HspyGain(HspySignal1D):
    def __init__(self, di):
        super().__init__(di)

    def rebin(self):
        initial = self.hspy_gd.axes_manager[0].offset
        self.hspy_gd = self.hspy_gd.rebin(scale=[self.get_attr('averages'), 1])
        self.hspy_gd.axes_manager[0].offset = initial

    def rebin_and_align(self):
        self.rebin()
        self.align_zlp()

    def get_gain_profile(self):
        return self.get_11_di(isig=[-2.4, -1.8], sum_isig=True)


class gainData:

    def __init__(self):
        pass
        self.__fixed_energy = False

    def send_raw_MetaData(self, rd):
        index = [] #max position for alignment
        array = [0] * len(rd[0][0].data[0])
        temp_data =  [array] * len(rd)


        for i in range(len(rd)):
            for j in range(len(rd[i])):
                cam_hor = numpy.sum(rd[i][j].data, axis=0)
                index.append(numpy.where(cam_hor==numpy.max(cam_hor))[0][0]) #double [0] is because each numpy.where is a array of values. We dont want to have two maximums because aligning will be messy. Remember that we dont expect two maximuns in a ZLP that aren't pixels close together (probably neighbour)
                cam_hor = numpy.roll(cam_hor, -index[len(index)-1]+index[0])
                temp_data[i] = temp_data[i] + cam_hor

        temp_data = numpy.asarray(temp_data) #need this because nion gets some numpy array atributes, such as .shape
        return temp_data, index[0]

    def data_item_calibration(self, ic, dc, start_wav, step_wav, disp, index_zlp):
        ic.units='counts'
        dc[0].units='nm'
        dc[0].offset=start_wav
        dc[0].scale=step_wav
        dc[1].units='eV'
        dc[1].scale=disp
        dc[1].offset=-disp*index_zlp
        intensity_calibration = ic
        dimensional_calibrations = dc
        return intensity_calibration, dimensional_calibrations

    def send_info_data(self, info_data):
        temp_wl_data=[]
        temp_pw_data=[]
        temp_di_data=[]
        for i in range(len(info_data)):
            for j in range(len(info_data[i])):
                temp_wl_data.append(info_data[i][j][0])
                temp_pw_data.append(info_data[i][j][1])
                temp_di_data.append(info_data[i][j][2])
        temp_wl_data = numpy.asarray(temp_wl_data)
        temp_pw_data = numpy.asarray(temp_pw_data)
        temp_di_data = numpy.asarray(temp_di_data)
        return temp_wl_data, temp_pw_data, temp_di_data

    def fit_data(self, data, pts, start, end, step, disp, fwhm, orders, tol=0.0):
        #if not self.__fixed_energy: ene = 0
        def _gaussian_fit(x, *p):
            A, sigma, A_1, A_2, A_3, A_4, fond, x_off, ene = p
            func = A * numpy.exp(-(x - x_off) ** 2 / (2. * sigma ** 2)) +\
                   A_1 * numpy.exp(-(x - ene - x_off) ** 2 / (2. * sigma ** 2)) +\
                   A_1 * numpy.exp(-(x + ene - x_off) ** 2 / (2. * sigma ** 2)) + fond
            if orders>1:
                func = func + A_2 * numpy.exp(-(x + 2.*ene - x_off) ** 2 / (2. * sigma ** 2)) +\
                   A_2 * numpy.exp(-(x - 2.*ene - x_off) ** 2 / (2. * sigma ** 2))
                if orders>2:
                    func = func + A_3 * numpy.exp(-(x + 3. * ene - x_off) ** 2 / (2. * sigma ** 2)) + \
                   A_3 * numpy.exp(-(x - 3. * ene - x_off) ** 2 / (2. * sigma ** 2))
                    if orders>3:
                        func = func + A_4 * numpy.exp(-(x + 4. * ene - x_off) ** 2 / (2. * sigma ** 2)) + \
                               A_4 * numpy.exp(-(x - 4. * ene - x_off) ** 2 / (2. * sigma ** 2))

            return func

        fit_array = numpy.zeros(data.shape)
        a_array = numpy.zeros(data.shape[0])
        a1_array = numpy.zeros(data.shape[0])
        a2_array = numpy.zeros(data.shape[0])
        a3_array = numpy.zeros(data.shape[0])
        a4_array = numpy.zeros(data.shape[0])
        sigma_array = numpy.zeros(data.shape[0])
        ene_array = numpy.zeros(data.shape[0])

        wavs = numpy.linspace(start, end, pts-1)
        energies_loss = numpy.divide(1239.8, wavs)
        energies_loss = numpy.append(energies_loss, 0.)

        for i in range(fit_array.shape[0]):
            x = numpy.linspace(-(fit_array.shape[1] / 2.) * disp, (fit_array.shape[1] / 2.) * disp, fit_array.shape[1])
            ene = energies_loss[i]
            p0 = [max(fit_array[i]), 1, 0., 0., 0., 0., data.min(), 0., ene]
            if ene: energy_window = (orders+1) * ene
            if not ene: energy_window = 3.0
            window_pixels = int(energy_window / disp)
            half_pixels = int(fit_array.shape[1] / 2)
            coeff, var_matrix = curve_fit(_gaussian_fit, x[half_pixels-window_pixels:half_pixels+window_pixels], data[i][half_pixels-window_pixels:half_pixels+window_pixels], p0=p0, bounds=([0., 0., 0., 0., 0., 0., 0., -numpy.inf, ene*(1.-tol)-10**(-3)], [numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, numpy.inf, ene*(1.+tol)+10**(-3)]))
            a_array[i], a1_array[i], a2_array[i], a3_array[i], a4_array[i], sigma_array[i], ene_array[i] = coeff[0], coeff[2], coeff[3], coeff[4], coeff[5], coeff[1], coeff[8]
            fit_array[i] = _gaussian_fit(x, *coeff)
            if ene: print(f'***ACQUISITION***: Fitting Data: ' + format(i/fit_array.shape[0]*100, '.0f') + '%. Current Wavelength is: ' + format(1239.8/ene, '.2f') + ' nm')
        return fit_array, a_array, a1_array, a2_array, a3_array, a4_array, sigma_array, ene_array


    def align_zlp(self, raw_array, pts, avg, pixels, disp, mode='max'):

        def _gaussian(x, *p):
            A, mu, sigma = p
            return A*numpy.exp(-(x-mu)**2/(2.*sigma**2))

        def _gaussian_two_replicas(x, *p): #ONE GAIN ONE LOSS.
            A, mu, sigma, AG, muG, AL, muL = p
            return A*numpy.exp(-(x-mu)**2/(2.*sigma**2)) + AG*numpy.exp(-(x-muG)**2/(2.*sigma**2)) + AL*numpy.exp(-(x-muL)**2/(2.*sigma**2))
        
        proc_array = numpy.zeros((pts, pixels))
        #zlp_fit = numpy.zeros(avg)

        raw_array[numpy.isnan((raw_array))] = 0.

        if 'max' in mode or 'fit' in mode:
            logging.info('***ACQUISITION***: MAX is the only method currently available.')
            for i in range(len(proc_array)):
                for j in range(avg):
                    current_max_index = numpy.where(raw_array[i*avg+j]==numpy.max(raw_array[i*avg+j]))[0][0]
                    proc_array[i] = proc_array[i] + numpy.roll(raw_array[i*avg+j], -current_max_index + int(pixels/2))
            return proc_array
                    #x = numpy.linspace((-pixels/2.-1)*disp, (pixels/2.)*disp, pixels)
                    #energy_window = 3.0
                    #window_pixels = int(energy_window / disp)
                    #half_pixels = int(pixels/2)
                    #p0 = [max(proc_array[i]), 0., 1]
                    #coeff, var_matrix = curve_fit(_gaussian, x[half_pixels-window_pixels:half_pixels+window_pixels], proc_array[i][half_pixels-window_pixels:half_pixels+window_pixels], p0 = p0)
                    #if i==(len(proc_array)-1):
                    #    zlp_fit[j] = coeff[2]

            #return proc_array, 2*numpy.mean(zlp_fit)*numpy.sqrt(2.*numpy.log(2)), energy_window

        #if 'fit' in mode: #I HAVE SUB PIXEL WITH MAX_INDEX. How to improve further with fit? I am not sure.
        #    for i in range(len(proc_array)):
        #        for j in range(avg):
        #            current_max_index = numpy.where(raw_array[i*avg+j]==numpy.max(raw_array[i*avg+j]))[0][0]
        #            proc_array[i] = proc_array[i] + numpy.roll(raw_array[i*avg+j], -current_max_index + int(pixels/2))
        #            x = numpy.linspace((-pixels/2.+1)*disp, (pixels/2.)*disp, pixels)
        #            p0 = [max(proc_array[i]), 0., 1]
        #            coeff, var_matrix = curve_fit(_gaussian, x, proc_array[i], p0=p0)
        #            if i==(len(proc_array)-1):
        #                zlp_fit[j] = coeff[2]
        #    return proc_array, 2*numpy.mean(zlp_fit)*numpy.sqrt(2.*numpy.log(2))

    def smooth_zlp(self, raw_array, window_size, poly_order, oversample, x, xx):
        smooth_array = numpy.zeros((raw_array.shape[0], raw_array.shape[1]*oversample))
        for i in range(len(raw_array)):
            itp = interp1d(x, raw_array[i], 'linear')
            smooth_array[i] = savgol_filter(itp(xx), window_size, poly_order)
        return smooth_array

    def as_power_func(self, raw_array, power_array):
        f = interp1d(power_array, raw_array, 'linear')
        power_array_new = numpy.linspace(power_array.min(), power_array.max(), len(power_array))
        raw_array_new = f(power_array_new)
        #the subtraction i am returning is to known the power increment
        return power_array_new, raw_array_new, power_array_new[1]-power_array_new[0]
