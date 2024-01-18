from nion.instrumentation import HardwareSource
from nion.utils import Registry
import logging, time, threading, numpy, json
from scipy import signal
import nion.instrumentation

class DriftCorrection:
    def __init__(self):

        self.__thread = None
        self.__abort = False
        self.datax = numpy.zeros(100)
        self.datay = numpy.zeros(100)
        self.cross_fft = numpy.zeros((256, 256))
        self.scan_systems = list()
        self.instrument_systems = list()
        self.scan_channels = dict()
        self.__array_index = 0
        self.__interval = 2
        self.__static_reference = False
        self.__should_correct = False
        self.__auto_track = True

        #Checking the scanning systems
        for hards in HardwareSource.HardwareSourceManager().hardware_sources:  # finding eels camera. If you don't
            if isinstance(hards, nion.instrumentation.scan_base.ConcreteScanHardwareSource):
                self.scan_systems.append(hards.hardware_source_id)
                channels = list()
                for index in range(hards.get_channel_count()): channels.append(hards.get_channel_name(index))
                self.scan_channels[hards] = channels

        #Checking the available instruments
        my_insts = Registry.get_components_by_type("stem_controller")
        for counter, my_inst in enumerate(list(my_insts)):
            self.instrument_systems.append(my_inst.instrument_id)

        self.__instrument = None if (len(self.instrument_systems)==0) else \
            HardwareSource.HardwareSourceManager().get_instrument_by_id(self.instrument_systems[0])

    def abort(self):
        self.__abort = True

    def start(self, callback, scan_system_index, instrument_system_index, interval, static_reference, should_correct, manual_correction, manual_correction_values):
        self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
            self.scan_systems[scan_system_index])
        self.__instrument = HardwareSource.HardwareSourceManager().get_instrument_by_id(self.instrument_systems[instrument_system_index])
        try:
            self.__interval = float(interval)
            self.__static_reference = static_reference
            self.__should_correct = should_correct
        except:
            pass

        self.__abort = False
        if self.__scan.is_playing == False:
            logging.info(f'***Drift Correction***: Please activate scan first.')
            return False
        if not manual_correction:
            self.__thread = threading.Thread(target=self.thread_func, args=(callback,))
        else:
            self.__thread = threading.Thread(target=self.thread_manual_func, args=(callback, manual_correction_values,))
        self.__thread.start()
        return True

    def append_to_array(self, xindex, yindex):
        self.datax[self.__array_index] = xindex
        self.datay[self.__array_index] = yindex
        if self.__array_index >= 99:
            self.__array_index = 0
        else:
            self.__array_index = self.__array_index + 1

    def get_shifters(self):
        if self.__instrument:
            return (self.__instrument.TryGetVal('CSH.u')[1], self.__instrument.TryGetVal('CSH.v')[1])
        else:
            logging.info("***Drift Correction***: TryGetVal and SetVal not implemented.")

    def displace_shifter_relative(self, dimension, value, **kwargs):
        if "instrument_id" in kwargs:
            self.__instrument = HardwareSource.HardwareSourceManager().get_instrument_by_id(kwargs['instrument_id'])
        if self.__instrument:
            to_add = self.get_shifters()[dimension]
            self.__instrument.SetVal('CSH.u' if dimension == 0 else 'CSH.v', to_add + value*1e-9)
        else:
            logging.info("***Drift Correction***: TryGetVal and SetVal not implemented.")

    def check_and_wait(self):
        time.sleep(self.__interval)

    def thread_manual_func(self, callback, manual_correction_values):
        x_val, y_val = manual_correction_values
        x_val, y_val = float(x_val), float(y_val)
        time_correction = 0
        while not self.__abort:
            start = time.time()
            self.check_and_wait()
            self.displace_shifter_relative(0, x_val * self.__interval + time_correction)
            self.displace_shifter_relative(1, y_val * self.__interval + time_correction)
            callback()
            end = time.time()
            time_correction = end - start - self.__interval


    def thread_func(self, callback):
        time_correction = 0
        reference_image = self.__scan.grab_next_to_finish()
        #TODO: search the correct channel index based on the scan selection drop down from user
        #channel_name = reference_image[0].metadata['channel_name'] #or channel_index
        while not self.__abort:
            start = time.time()
            self.check_and_wait()
            new_image = self.__scan.grab_next_to_finish()
            if reference_image[0].data.shape == new_image[0].data.shape: #Should correct
                self.cross_fft = signal.correlate(reference_image[0].data, new_image[0].data, method = 'fft')
                index = numpy.argmax(self.cross_fft)
                xindex = int(index % self.cross_fft.shape[0])
                yindex = int(index / self.cross_fft.shape[1])

                xdrift = reference_image[0].get_dimensional_calibration(0).scale / (self.__interval + time_correction) * (xindex - int(self.cross_fft.shape[0]/2))
                ydrift = reference_image[0].get_dimensional_calibration(1).scale / (self.__interval + time_correction) * (yindex - int(self.cross_fft.shape[1]/2))

                self.append_to_array(
                    xdrift,
                    ydrift
                )

                if self.__should_correct:
                    self.displace_shifter_relative(0, - 0.8 * xdrift * (self.__interval + time_correction))
                    self.displace_shifter_relative(1, - 0.8 * ydrift * (self.__interval + time_correction))

                callback()
            else:
                logging.info(f'***Drift Correction***: Image shape between reference and new image is different. '
                             f'Restart drift correction for proper calibration.')
            if not self.__static_reference: reference_image = new_image
            end = time.time()
            logging.info(f'***Drift Correction***: Drift.x {xdrift, xindex} (nm/s). Drift.y: {ydrift, yindex} (nm/s). '
                         f'Interval: {end - start} (s).')
            time_correction = end - start - self.__interval
