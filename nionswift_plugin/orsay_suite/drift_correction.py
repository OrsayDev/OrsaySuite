from nion.swift.model import HardwareSource
from nion.utils import Registry
import logging, time, threading
from scipy import signal
import numpy

class DriftCorrection:
    def __init__(self):

        self.__thread = None
        self.__abort = False
        self.datax = numpy.zeros(100)
        self.datay = numpy.zeros(100)
        self.__array_index = 0
        self.__interval = 2
        self.__static_reference = False

        cams = dict()
        scans = dict()

        for hards in HardwareSource.HardwareSourceManager().hardware_sources:  # finding eels camera. If you don't
            if hasattr(hards, 'hardware_source_id'):
                if hasattr(hards, '_CameraHardwareSource2__instrument_controller_id'):
                    cams[hards.hardware_source_id] = hards._CameraHardwareSource2__instrument_controller_id
                if hasattr(hards, 'grab_next_to_start'):
                    scans[hards.hardware_source_id] = hards.hardware_source_id

        print('Cameras:')
        print(cams)
        print('Scans')
        print(scans)

        my_insts = Registry.get_components_by_type("stem_controller")
        for counter, my_inst in enumerate(list(my_insts)):
            print(my_inst.instrument_id)
            print(counter)
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

        if self.__isChromaTEM:
            self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
                    "superscan")
        else:
            self.__scan = HardwareSource.HardwareSourceManager().get_hardware_source_for_hardware_source_id(
                    "open_scan_device")

    def abort(self):
        print('abort')
        self.__abort = True

    def start(self, callback, interval, static_reference):
        print('start')
        try:
            self.__interval = float(interval)
            self.__static_reference = static_reference
        except:
            pass

        self.__abort = False
        self.__thread = threading.Thread(target=self.thread_func, args=(callback,))
        self.__thread.start()

    def append_to_array(self, xindex, yindex):
        self.datax[self.__array_index] = xindex
        self.datay[self.__array_index] = yindex
        if self.__array_index >= 99:
            self.__array_index = 0
        else:
            self.__array_index = self.__array_index + 1


    def check_and_wait(self):
        time.sleep(self.__interval)
        if not self.__scan.is_playing:
            self.abort()

    def thread_func(self, callback):
        reference_image = self.__scan.grab_next_to_finish()
        while not self.__abort:
            start = time.time()
            self.check_and_wait()
            new_image = self.__scan.grab_next_to_finish()
            if reference_image[0].data.shape == new_image[0].data.shape: #Should correct
                corr = signal.correlate(reference_image[0].data, new_image[0].data, method = 'fft')
                index = numpy.argmax(corr)
                xindex = int(index % corr.shape[0])
                yindex = int(index / corr.shape[1])
                self.append_to_array(
                    reference_image[0].get_dimensional_calibration(0).scale / self.__interval * (xindex - int(corr.shape[0]/2)),
                    reference_image[0].get_dimensional_calibration(1).scale / self.__interval * (yindex - int(corr.shape[1]/2))
                )

                xdrift = reference_image[0].get_dimensional_calibration(0).scale * (xindex - int(corr.shape[0] / 2))
                ydrift = reference_image[0].get_dimensional_calibration(1).scale * (xindex - int(corr.shape[1] / 2))
                end = time.time()
                #logging.info(f'{xindex} and {yindex} and {end - start}')
                #self.__instrument.SetVal('CSH.u', 1-1e-9)
                #self.__instrument.SetValAndConfirm('CSH.v', 1-1e-9, 0, 0)

                callback()
            if not self.__static_reference: reference_image = new_image



