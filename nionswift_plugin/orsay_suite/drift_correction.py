from nion.swift.model import HardwareSource
from nion.utils import Registry
import logging, time, threading
from scipy import signal
import numpy
import nion.instrumentation
import json

class DriftCorrection:
    def __init__(self):

        self.__thread = None
        self.__abort = False
        self.datax = numpy.zeros(100)
        self.datay = numpy.zeros(100)
        self.cross_fft = numpy.zeros((256, 256))
        self.scan_systems = list()
        self.__array_index = 0
        self.__interval = 2
        self.__static_reference = False
        self.__should_correct = False
        self.__auto_track = True

        self.shifter_calibration = dict()
        self.shifter_calibration['calib.x'] = 0.
        self.shifter_calibration['calib.y'] = 0.
        try:
            f = open('persistent.json')
            self.shifter_calibration = json.load(f)
        except FileNotFoundError:
            with open('persistent.json', 'w') as f:
                json.dump(self.shifter_calibration, f)


        #cams = list()

        for hards in HardwareSource.HardwareSourceManager().hardware_sources:  # finding eels camera. If you don't
            if isinstance(hards, nion.instrumentation.scan_base.ConcreteScanHardwareSource):
                self.scan_systems.append(hards.hardware_source_id)
           # if isinstance(hards, nion.instrumentation.camera_base.CameraHardwareSource2):
           #     cams.append(hards.hardware_source_id)

        #print('Cameras:')
        #print(cams)
        #print('Scans')
        #print(self.scan_systems)

        #my_insts = Registry.get_components_by_type("stem_controller")
        #for counter, my_inst in enumerate(list(my_insts)):
        #    print(my_inst.instrument_id)
        #    print(counter)
            #print(dir(my_inst))

        scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id("open_scan_device")
        print(scan)

        # Checking if auto_stem is here. This is to control a Nion microscope
        AUTOSTEM_CONTROLLER_ID = "autostem_controller"
        self.__isChromaTEM = False
        autostem = HardwareSource.HardwareSourceManager().get_instrument_by_id(AUTOSTEM_CONTROLLER_ID)
        if autostem != None:
            tuning_manager = autostem.tuning_manager
            self.__instrument = tuning_manager.instrument_controller
            self.__isChromaTEM = True
            logging.info("*** Drift Correction ***: autostem_controller is found.")
        else:
            self.__instrument = HardwareSource.HardwareSourceManager().get_instrument_by_id("VG_controller")
            logging.info("*** Drift Correction ***: VG_controller is found.")

        self.__offset_valx, self.__offset_valy = self.get_shifters()

        #if self.__isChromaTEM:
        #    self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
        #            "superscan")
        #else:
        #    self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
        #            "open_scan_device")

    def abort(self):
        self.__abort = True

    def start(self, callback, scan_system, interval, static_reference, should_correct, manual_correction, manual_correction_values):
        self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
            scan_system)
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
            self.__thread = threading.Thread(target=self.thread_manual_func, args=(manual_correction_values,))
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
        return (self.__instrument.TryGetVal('CSH.u')[1], self.__instrument.TryGetVal('CSH.v')[1])

    def displace_shifter_relative(self, dimension, value):
        valx, valy = self.get_shifters()
        if dimension == 0:
            self.__instrument.SetVal('CSH.u', valx + value*1e-9)
        elif dimension == 1:
            self.__instrument.SetVal('CSH.v', valy + value*1e-9)

    def displace_shifter_absolute(self, dimension, value):
        if dimension == 0:
            self.__instrument.SetVal('CSH.u', value*1e-9)
        elif dimension == 1:
            self.__instrument.SetVal('CSH.v', value*1e-9)

    def displace_shifter_reset(self, dimension):
        if dimension == 0:
            self.__instrument.SetVal('CSH.u', self.__offset_valx*1e-9)
        elif dimension == 1:
            self.__instrument.SetVal('CSH.v', self.__offset_valy*1e-9)

    def set_calibration(self, dimension, value):
        if dimension == 0:
            key = 'calib.x'
        elif dimension == 1:
            key = 'calib.y'
        self.shifter_calibration[key] = value
        with open('persistent.json', 'w') as f:
            json.dump(self.shifter_calibration, f)


    def check_and_wait(self):
        time.sleep(self.__interval)
        #if not self.__scan.is_playing:
        #    self.abort()


    def thread_manual_func(self, manual_correction_values):
        x_val, y_val = manual_correction_values
        x_val = float(x_val)
        y_val = float(y_val)
        time_correction = 0
        while not self.__abort:
            start = time.time()
            self.check_and_wait()
            self.displace_shifter_relative(0, x_val * self.__interval + time_correction)
            self.displace_shifter_relative(1, y_val * self.__interval + time_correction)
            end = time.time()
            time_correction = end - start - self.__interval


    def thread_func(self, callback):
        time_correction = 0
        reference_image = self.__scan.grab_next_to_finish()
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
                #xdrift = reference_image[0].get_dimensional_calibration(0).scale * (xindex - int(corr.shape[0] / 2))
                #ydrift = reference_image[0].get_dimensional_calibration(1).scale * (xindex - int(corr.shape[1] / 2))
                #logging.info(f'***Drift Correction***: Drift.x {xdrift} (nm/s). Drift.y: {ydrift} (nm/s). '
                #             f'Interval: {end - start} (s).')
                if self.__should_correct:
                    self.displace_shifter_relative(0, - 0.8 * xdrift * (self.__interval + time_correction))
                    self.displace_shifter_relative(1, - 0.8 * ydrift * (self.__interval + time_correction))

                callback()
            if not self.__static_reference:
                logging.info(f'new reference image')
                reference_image = new_image
            end = time.time()
            logging.info(f'***Drift Correction***: Drift.x {xdrift} (nm/s). Drift.y: {ydrift} (nm/s). '
                         f'Interval: {end - start} (s).')
            time_correction = end - start - self.__interval



