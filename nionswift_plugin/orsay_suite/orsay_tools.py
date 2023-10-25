# standard libraries
import gettext
import threading
import time

from nion.swift import Panel
from nion.swift import Workspace

from nion.utils import Event
from nion.ui import Declarative
from nion.ui import UserInterface
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.model import Utility

from . import orsay_data
from . import drift_correction

import logging
import numpy

_ = gettext.gettext

class DataItemCreation():
    def __init__(self, name, array, signal_dim, offset: list, scale: list, units: list, **kwargs):
        self.metadata = kwargs
        self.timezone = Utility.get_local_timezone()
        self.timezone_offset = Utility.TimezoneMinutesToStringConverter().convert(Utility.local_utcoffset_minutes())

        self.calibration = Calibration.Calibration()
        self.dimensional_calibrations = [Calibration.Calibration() for _ in range(signal_dim)]
        assert len(offset)==len(scale) and len(offset)==len(units)

        for index, x in enumerate(zip(offset, scale, units)):
            self.dimensional_calibrations[index].offset, self.dimensional_calibrations[index].scale, self.dimensional_calibrations[index].units = x[0], x[1], x[2]

        self.xdata = DataAndMetadata.new_data_and_metadata(array, self.calibration, self.dimensional_calibrations,
                                                           metadata=self.metadata,
                                                           timezone=self.timezone, timezone_offset=self.timezone_offset)
        self.data_item = DataItem.DataItem()
        self.data_item.set_xdata(self.xdata)
        self.data_item.title = name
        self.data_item.description = kwargs
        self.data_item.caption = kwargs

    def fast_update_data_only(self, array: numpy.array):
        self.data_item.set_data(array)

    def update_data_only(self, array: numpy.array):
        self.xdata = DataAndMetadata.new_data_and_metadata(array, self.calibration, self.dimensional_calibrations,
                                                           metadata=self.metadata,
                                                           timezone=self.timezone, timezone_offset=self.timezone_offset)
        self.data_item.set_xdata(self.xdata)

class handler:

    def __init__(self, document_controller):

        self.property_changed_event = Event.Event()
        self.event_loop = document_controller.event_loop
        self.document_controller = document_controller
        self.__drift = drift_correction.DriftCorrection()
        self.__crossx_di = None
        self.__crossy_di = None
        self.__cross_fft = None

    def init_handler(self):
        self.event_loop.create_task(self.do_enable(True, ['']))
        self.x_le.text = '1'
        self.y_le.text = '1'
        self.E_le.text = '1'
        self.comp_le.text = '3'
        self.int_le.text = '10'
        self.static_reference_pb.checked = True
        self.display_fft_pb.checked = False
        self.time_interval_value.text = '2'
        self.time_interval_manual_value.text = '2'
        self.calib_shifter_dim1_value.text = '1.0'
        self.calib_shifter_dim2_value.text = '1.0'
        self.manual_drift_x_value.text = '1.0'
        self.manual_drift_y_value.text = '1.0'
        self.scan_selection_dd.items = self.__drift.scan_systems

    async def data_item_show(self, DI):
        self.document_controller.document_model.append_data_item(DI)

    async def data_item_remove(self, DI):
        self.document_controller.document_model.remove_data_item(DI)

    async def data_item_exit_live(self, DI):
        DI._exit_live_state()

    async def do_enable(self, enabled=True, not_affected_widget_name_list=None):
        for var in self.__dict__:
            if var not in not_affected_widget_name_list:
                if isinstance(getattr(self, var), UserInterface.Widget):
                    widg = getattr(self, var)
                    setattr(widg, "enabled", enabled)

    def prepare_widget_enable(self, value):
        self.event_loop.create_task(self.do_enable(False, ["init_pb"]))

    def prepare_widget_disable(self, value):
        self.event_loop.create_task(self.do_enable(False, ["init_pb"]))

    def prepare_free_widget_enable(self,
                                   value):  # THAT THE SECOND EVENT NEVER WORKS. WHAT IS THE DIF BETWEEN THE FIRST?
        self.event_loop.create_task(
            self.do_enable(True, ['init_pb']))

    def bin_laser(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspyGain(self.__current_DI.data_item)
            self.gd.rebin()
            self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    # def correct_junctions(self, widget):
    #     self.__current_DI = None
    #
    #     self.__current_DI = self._pick_di()
    #
    #     if self.__current_DI:
    #         self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
    #         corrected_data = self.gd.detector_junctions()
    #         self.event_loop.create_task(self.data_item_show(corrected_data))
    #     else:
    #         logging.info('***PANEL***: Could not find referenced Data Item.')

    def correct_gain(self, widget):
        #Not yet implemented. Needs ROI and threshold in metadata
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            metadata_keys = self.__current_DI.data_item.metadata.keys()
            metadata_name = 'hardware_source'
            if metadata_name in metadata_keys:
                ht = self.__current_DI.data_item.metadata['hardware_source']['high_tension']
                ht = str(int(ht/1000))
                threshold = 20 #NOT IMPLEMENTED: must come from metadata
                gain_roi = [0, 0, 1024, 64] #NOT IMPLEMENTED: must come from metadata
                #current implentation is x0, y0, x1, y1. CHECK IF TRUE LATER in metadata
                self.gd.correct_gain_hs(ht, threshold, gain_roi)

                self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
                logging.info('***PANEL***: Not implemented. Needs to implement ROI and THRESHOLD from metadata')
            else:
                logging.info('***PANEL***: No ' + metadata_name + ' metadata available for this data_item')
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    # def flip_signal(self, widget):
    #     self.__current_DI = None
    #
    #     self.__current_DI = self._pick_di()
    #
    #     if self.__current_DI:
    #         self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
    #         self.gd.flip()
    #         self.event_loop.create_task(self.data_item_show(self.gd.get_di()))

    # def _align_chrono(self, align=True, bin=False):
    #     self.__current_DI = None
    #
    #     self.__current_DI = self._pick_di()
    #
    #     if self.__current_DI:
    #         self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
    #         if align: self.gd.align_zlp()
    #         if bin:
    #             self.event_loop.create_task(self.data_item_show(self.gd.get_di(sum_inav=True)))
    #         else:
    #             self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
    #     else:
    #         logging.info('***PANEL***: Could not find referenced Data Item.')

    def gain_profile_data_item(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspyGain(self.__current_DI.data_item)
            self.event_loop.create_task(self.data_item_show(self.gd.get_gain_profile()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def gain_profile_2d_data_item(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspyGain(self.__current_DI.data_item)
            self.event_loop.create_task(self.data_item_show(self.gd.get_gain_2d()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    # def deconvolve_rl_hspec(self, widget):
    #     try:
    #         val = int(self.int_le.text)
    #     except ValueError:
    #         logging.info("***PANEL***: Interaction value must be integer.")
    #     self._deconvolve_hspec('Richardson lucy', val)
    #
    # def _deconvolve_hspec(self, type, interactions):
    #     self.__current_DI = None
    #
    #     self.__current_DI = self._pick_di()
    #
    #     if self.__current_DI:
    #         logging.info('***PANEL***: Must select to spectra for deconvolution.')
    #     else:
    #         dis = self._pick_dis()  # Multiple data items
    #         val_spec = list()
    #         if dis and len(dis) == 2:
    #             self.gd = orsay_data.HspySignal1D(dis[0].data_item)  # hspec
    #             self.spec_gd = orsay_data.HspySignal1D(dis[1].data_item) #spec
    #             new_di = self.gd.deconvolution(self.spec_gd, type, interactions)
    #             self.event_loop.create_task(self.data_item_show(new_di))
    #         else:
    #             logging.info('***PANEL***: Could not find referenced Data Item.')


    def _general_actions(self, type, which):
        self.__current_DI = None

        self.__current_DI = self._pick_di() #Single Data Item here

        def correct_junction(hspy_signal, *args):
            corrected_data = hspy_signal.detector_junctions()
            self.event_loop.create_task(self.data_item_show(corrected_data))

        def align_zlp_action(hspy_signal, val, *args):
            hspy_signal.align_zlp_signal_range(val)
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        def flip_signal(hspy_signal, *args):
            hspy_signal.gd.flip()
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        def fitting_action(hspy_signal, val, *args):
            if which == 'gaussian':
                new_di = hspy_signal.plot_gaussian(val)
            elif which == 'lorentzian':
                new_di = hspy_signal.plot_lorentzian(val)
            self.event_loop.create_task(self.data_item_show(new_di))

        def remove_background_action(hspy_signal, val, *args):
            hspy_signal.remove_background(val, which)
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        def deconvolve_hyperspec(hspy_signal, hspy_spec_signal, type, interactions, *args):
            new_di = hspy_signal.deconvolution(hspy_spec_signal, type, interactions)
            self.event_loop.create_task(self.data_item_show(new_di))

        def decomposition_action(hspy_signal, val, *args):
            var, new_di = hspy_signal.signal_decomposition(val, which)
            self.event_loop.create_task(self.data_item_show(new_di))
            self.event_loop.create_task(self.data_item_show(var))


        if type == 'fitting':
            action = fitting_action
        elif type == 'remove_background':
            action = remove_background_action
        elif type == 'decomposition':
            action = decomposition_action
        elif type == 'align_zlp':
            action = align_zlp_action
        elif type == 'correct_junctions':
            action = correct_junction
        elif type == 'flip_signal':
            action = flip_signal
        elif type == 'deconvolve':
            action = deconvolve_hyperspec
        else:
            raise Exception("***PANEL***: No action function was selected. Please check the correct type.")

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            for graphic in self.__current_DI.graphics:
                if graphic.type == 'rect-graphic':  # This is a hyperspectral image
                    #logging.info('***PANEL***: Hyperspectrum selected. If you wish to perform this action, please select two '
                    #             'data items.')
                    action(self.gd)
                if graphic.type == 'line-profile-graphic':  # This is Chrono
                    val = (graphic.start[1], graphic.end[1])
                    action(self.gd, val)
                    return
                if graphic.type == 'interval-graphic':  # This is 1D
                    val = graphic.interval
                    action(self.gd, val)
        else:
            dis = self._pick_dis()  # Multiple data items
            val_spec = list()
            if dis and len(dis) == 2:
                spec = dis[1]
                hspec = dis[0]
                for graphic in spec.graphics:
                    if graphic.type == 'interval-graphic':
                        val_spec = graphic.interval
                for graphic in hspec.graphics:
                    if graphic.type == 'rect-graphic' and len(val_spec) == 2:
                        self.gd = orsay_data.HspySignal1D(hspec.data_item)
                        action(self.gd, val_spec)
                        return
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')

    def correct_junctions(self, widget):
        self._general_actions('correct_junctions', 'None')
    def flip_signal(self, widget):
        self._general_actions('flip_signal', 'None')
    def remove_background_pl(self, widget):
        self._general_actions('remove_background', 'Power law')
    def remove_background_off(self, widget):
        self._general_actions('remove_background', 'Offset')

    def deconvolve_rl_hspec(self, widget):
        self._general_actions('deconvolve', 'rl')

    def fit_gaussian(self, widget):
        self._general_actions('fitting', 'gaussian')

    def fit_lorentzian(self, widget):
        self._general_actions('fitting', 'lorentzian')

    def align_zlp(self, widget):
        self._general_actions('align_zlp', 'None')

    def pca(self, widget):
        try:
            val = int(self.comp_le.text)
        except ValueError:
            logging.info("***PANEL***: Interaction value must be integer.")
        self._general_actions('decomposition', val)


    def pca_full(self, widget):
        self._pca_full()

    def _pca_full(self):
        try:
            val = int(self.comp_le.text)
        except ValueError:
            logging.info("***PANEL***: Components value must be integer.")
        self.__current_DI = self._pick_di()
        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)  # hspec
            var, new_di, factors, loadings_stacked = self.gd.signal_decomposition(components=val, mask=False)
            self.event_loop.create_task(self.data_item_show(new_di))
            self.event_loop.create_task(self.data_item_show(var))
            self.event_loop.create_task(self.data_item_show(factors))
            self.event_loop.create_task(self.data_item_show(loadings_stacked))

            """
            factors_np = numpy.copy(factors.data)
            for i in range(val):
                factor = factors_np[:, i]
                factor_hs = orsay_data.HspySignal1D(factor)
                #self._get_data(factor_hs, 'PCA_factors'+str(i)+'_'),
                self.event_loop.create_task(self.data_item_show(factor_hs))
                #self.event_loop.create_task(self.data_item_show(loadings))
            """
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def hspy_bin(self, widget):
        try:
            x = int(self.x_le.text)
            y = int(self.y_le.text)
            E = int(self.E_le.text)

            self.__current_DI = None

            self.__current_DI = self._pick_di()

            if self.__current_DI:
                self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
                self.gd.rebin(scale=[x, y, E])
                self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')

        except ValueError:
            logging.info("***PANEL***: Bin values must be integers.")

    def _pick_di(self):
        display_item = self.document_controller.selected_display_item
        return display_item

    def _pick_dis(self):
        display_item = self.document_controller.selected_display_items
        return display_item

    def activate_cross_correlation(self, widget):
        if 'Measure' in widget.text:
            should_correct = 'Correct' in widget.text
            if not self.__drift.start(self._update_cross_correlation, self.__drift.scan_systems[self.scan_selection_dd.current_index], self.time_interval_value.text,
                               self.static_reference_pb.checked, should_correct, False, (self.manual_drift_x_value.text, self.manual_drift_y_value.text)):
                self.note_label.text = 'Status: Not running (activate scan channel)'
                return

            if should_correct:
                widget.text = 'Stop'
                self.act_corr_pushb.enabled = False
            else:
                widget.text = 'Abort'
                self.act_meas_corr_pushb.enabled = False
            self.note_label.text = 'Status: Running'
            self.second_row.enabled = False

            if self.__crossx_di == None:
                self.__crossx_di = DataItemCreation('XDrift', self.__drift.datax, 1, [0], [1], ['Time interval'])
                self.event_loop.create_task(self.data_item_show(self.__crossx_di.data_item))
                self.__crossx_di.data_item._enter_live_state()

            if self.__crossy_di == None:
                self.__crossy_di = DataItemCreation('YDrift', self.__drift.datay, 1, [0], [1], ['Time interval'])
                self.__crossy_di.data_item._enter_live_state()
                self.event_loop.create_task(self.data_item_show(self.__crossy_di.data_item))

            self.__fft_show = self.display_fft_pb.checked
            if self.__cross_fft == None and self.__fft_show:
                self.__cross_fft = DataItemCreation('FFT Cross', self.__drift.cross_fft, 2, [-256, -256], [1, 1], ['arb. units', 'arb. units'])
                self.__cross_fft.data_item._enter_live_state()
                self.event_loop.create_task(self.data_item_show(self.__cross_fft.data_item))

        elif widget.text == 'Abort':
            self.act_meas_corr_pushb.enabled = True
            self.note_label.text = 'Status: Not running'
            widget.text = 'Measure'
            self.second_row.enabled = True
            self.__drift.abort()
        elif widget.text == 'Stop':
            self.act_corr_pushb.enabled = True
            self.note_label.text = 'Status: Not running'
            widget.text = 'Measure &&& Correct'
            self.second_row.enabled = True
            self.__drift.abort()

    def _update_cross_correlation(self):
        self.__crossx_di.update_data_only(self.__drift.datax)
        self.__crossy_di.update_data_only(self.__drift.datay)
        if self.__fft_show: self.__cross_fft.update_data_only(self.__drift.cross_fft)
        self.property_changed_event.fire("drift_u")
        self.property_changed_event.fire("drift_v")
        self.property_changed_event.fire("shifter_u")
        self.property_changed_event.fire("shifter_v")

    def displace_shifter(self, widget):
        if widget.text == 'Displace X':
            self.__drift.displace_shifter_relative(0, float(self.calib_shifter_dim1_value.text))
        if widget.text == 'Displace Y':
            self.__drift.displace_shifter_relative(1, float(self.calib_shifter_dim2_value.text))
        self.property_changed_event.fire("shifter_u")
        self.property_changed_event.fire("shifter_v")

    def start_manual_correction(self, widget):
        if widget.text == "Start":
            if not self.__drift.start(self._update_cross_correlation, self.__drift.scan_systems[self.scan_selection_dd.current_index], self.time_interval_manual_value.text,
                               self.static_reference_pb.checked, True, True, (self.manual_drift_x_value.text, self.manual_drift_y_value.text)):
                self.note_manual_label.text = 'Status: Not running (activate scan channel)'
                return
            widget.text = 'Abort'
            self.note_manual_label.text = 'Status: Running'
            self.manual_drift_x_value.enabled = False
            self.manual_drift_y_value.enabled = False
            self.time_interval_manual_value.enabled = False
        elif widget.text == 'Abort':
            widget.text = "Start"
            self.note_manual_label.text = 'Status: Not running'
            self.manual_drift_x_value.enabled = True
            self.manual_drift_y_value.enabled = True
            self.time_interval_manual_value.enabled = True
            self.__drift.abort()

    def reset_shifter(self, widget):
        self.__drift.displace_shifter_reset(0)
        self.__drift.displace_shifter_reset(1)
        self.property_changed_event.fire("shifter_u")
        self.property_changed_event.fire("shifter_v")

    def set_calibration(self, widget):
        if widget.text == 'Ok 100 nm X':
            dim = 0
            value = self.shifter_u
        elif widget.text == 'Ok 100 nm Y':
            dim = 1
            value = self.shifter_v
        self.__drift.set_calibration(dim, value)
        self.property_changed_event.fire("shifter_u_calib")
        self.property_changed_event.fire("shifter_v_calib")

    @property
    def shifter_u(self):
        return round(self.__drift.get_shifters()[0]*1e9, 3)

    @property
    def shifter_v(self):
        return round(self.__drift.get_shifters()[1]*1e9, 3)

    @property
    def shifter_u_calib(self):
        return round(self.__drift.shifter_calibration['calib.x']*1e9, 3)

    @property
    def shifter_v_calib(self):
        return round(self.__drift.shifter_calibration['calib.y']*1e9, 3)

class View:

    def __init__(self):
        ui = Declarative.DeclarativeUI()

        #General elements. Used often.
        self.close_par = ui.create_label(text=')')

        #Gain group
        self.bin_laser_pb = ui.create_push_button(text='Bin laser', name='bin_laser_pb', on_clicked='bin_laser')
        self.profile_2d_pb = ui.create_push_button(text='2D Gain profile ', name='profile_2d_pb',
                                                on_clicked='gain_profile_2d_data_item')
        self.profile_pb = ui.create_push_button(text='Gain profile ', name='profile_pb', on_clicked='gain_profile_data_item')
        self.pb_row = ui.create_row(self.bin_laser_pb, self.profile_2d_pb, self.profile_pb, ui.create_stretch())

        self.d2_group = ui.create_group(title='EEGS Tools', content=ui.create_column(
            self.pb_row, ui.create_stretch()))

        #General Group
        self.fit_text = ui.create_label(text='Fitting: ', name='fit_text')
        self.fit_gaussian_pb = ui.create_push_button(text='Gaussian', name='fit_gaussian_pb', on_clicked='fit_gaussian')
        self.fit_lorentzian_pb = ui.create_push_button(text='Lorentzian', name='fit_lorentzian_pb',
                                                     on_clicked='fit_lorentzian')

        self.simple_text = ui.create_label(text='Simple corrections: ', name='simple_text')
        self.cj_pb = ui.create_push_button(text='Correct Junctions', name='cj_pb', on_clicked='correct_junctions')
        self.align_zlp_pb = ui.create_push_button(text='Align ZLP', name='align_zlp_pb', on_clicked='align_zlp')
        self.cgain_pb = ui.create_push_button(text='Correct Gain', name='cgain_pb',
                                              on_clicked='correct_gain')
        self.flip_pb = ui.create_push_button(text='Flip signal', name='flip_pb',
                                              on_clicked='flip_signal')

        self.background_text = ui.create_label(text='Background Removal: ', name='background_text')
        self.remove_pl_pb = ui.create_push_button(text='Power Law', name='remove_pl_pb', on_clicked='remove_background_pl')
        self.remove_pl_offset = ui.create_push_button(text='Offset', name='remove_off_pb',
                                                  on_clicked='remove_background_off')

        self.binning_text = ui.create_label(text='Bin data (x, y, E): (', name='binning_text')
        self.x_le = ui.create_line_edit(name='x_le', width=25)
        self.y_le = ui.create_line_edit(name='y_le', width=25)
        self.E_le = ui.create_line_edit(name='E_le', width=25)
        self.bin_pb = ui.create_push_button(text='Bin', name='bin_pb', on_clicked='hspy_bin')

        self.pb_row = ui.create_row(self.simple_text, self.cj_pb, self.align_zlp_pb, self.cgain_pb,
                                    self.flip_pb, ui.create_stretch())
        self.pb_remove = ui.create_row(self.background_text, self.remove_pl_pb, self.remove_pl_offset, ui.create_stretch())
        self.pb_row_fitting = ui.create_row(self.fit_text, self.fit_gaussian_pb, self.fit_lorentzian_pb,
                                            ui.create_stretch())
        self.binning_row = ui.create_row(self.binning_text, self.x_le, self.y_le,
                                         self.E_le, self.close_par, ui.create_spacing(5), self.bin_pb,
                                         ui.create_stretch())

        self.general_group = ui.create_group(title='General Tools', content=ui.create_column(
            self.pb_row, self.pb_remove, self.pb_row_fitting, self.binning_row, ui.create_stretch()))


        #Hyperspectral group
        self.deconvolution_text = ui.create_label(text='Signal deconvolution (interactions): (', name='deconvolution_text')
        self.int_le = ui.create_line_edit(name='int_le', width=15)
        self.dec_rl_pb = ui.create_push_button(text='Richardson-Lucy', name='dec_rl_pb',
                                            on_clicked='deconvolve_rl_hspec')
        self.pb_row = ui.create_row(self.deconvolution_text, self.int_le, self.close_par, ui.create_spacing(5),
                                    self.dec_rl_pb, ui.create_stretch())

        self.dec_text = ui.create_label(text='Signal decomposition (components): (', name='dec_text')
        self.comp_le = ui.create_line_edit(name='comp_le', width=15)
        self.pca_pb = ui.create_push_button(text='Masked SVD', name='pca3_pb', on_clicked='pca')
        self.pca2_pb = ui.create_push_button(text='Full SVD', name='pca3_pb', on_clicked='pca_full')
        self.decomposition_row = ui.create_row(self.dec_text, self.comp_le, self.close_par, ui.create_spacing(5),
                                               self.pca_pb, self.pca2_pb,
                                               ui.create_stretch())


        self.hspec_group = ui.create_group(title='Hyperspectral Image', content=ui.create_column(
            self.pb_row, self.decomposition_row, ui.create_stretch()
        ))

        self.last_text = ui.create_label(text='<a href="https://github.com/OrsayDev/OrsaySuite">Orsay Tools v0.1.0</a>', name='last_text')
        self.left_text = ui.create_label(text='<a href="https://hyperspy.org/hyperspy-doc/current/index.html">HyperSpy v1.6.5</a>', name='left_text')
        self.last_row = ui.create_row(self.left_text, ui.create_stretch(), self.last_text)

        self.hyperspytab = ui.create_tab(label = 'Processing', content = ui.create_column(self.general_group,
                                        self.hspec_group, self.d2_group, self.last_row))

        ##End of hyperspy data processing tab
        # Begin of cross-correlation tab

        #Scan Engine
        self.scan_selection_label = ui.create_label(text='Choose the scan system: ')
        self.scan_selection_dd = ui.create_combo_box(name='scan_selection_dd', items=['a'])
        self.first_row = ui.create_row(self.scan_selection_label, self.scan_selection_dd, ui.create_stretch())
        self.first_group = ui.create_group(title='Scan Engine', content=self.first_row)

        #Automatic Correction
        self.act_corr_pushb = ui.create_push_button(name='act_corr_pushb', text='Measure', on_clicked='activate_cross_correlation')
        self.act_meas_corr_pushb = ui.create_push_button(name='act_meas_corr_pushb', text='Measure &&& Correct',
                                                    on_clicked='activate_cross_correlation')
        self.static_reference_pb = ui.create_check_box(text='Static reference', name='static_reference_pb')
        self.display_fft_pb = ui.create_check_box(text='Display FFT', name='display_fft_pb')
        self.time_interval_label = ui.create_label(text='Time interval (s)', name ='time_interval_label')
        self.time_interval_value = ui.create_line_edit(name='time_interval_value', width = 50)

        self.note_label = ui.create_label(name='note_label', text='Status: Not running')


        self.second_row = ui.create_row(self.static_reference_pb, ui.create_spacing(10), self.display_fft_pb,
                                        ui.create_stretch(), self.time_interval_label,
                                        self.time_interval_value, name='second_row')
        self.fifth_row = ui.create_row(self.act_corr_pushb, self.act_meas_corr_pushb, ui.create_stretch(), self.note_label)
        self.second_group = ui.create_group(name='second_group', title='Automatic tracking',
                                            content=ui.create_column(self.second_row,
                                                                     self.fifth_row))

        #Manual tracking
        self.note_manual_label = ui.create_label(name='note_manual_label', text='Status: Not running')
        self.time_interval_manual_label = ui.create_label(text='Time interval (s)', name='time_interval_manual_label')
        self.time_interval_manual_value = ui.create_line_edit(name='time_interval_manual_value', width=50)

        self.manual_drift_x_label = ui.create_label(name='manual_drift_x_label', text='Manual drift.x (nm/s): ')
        self.manual_drift_x_value = ui.create_line_edit(name='manual_drift_x_value', width = 50)
        self.manual_drift_y_label = ui.create_label(name='manual_drift_y_label', text='Manual drift.y (nm/s): ')
        self.manual_drift_y_value = ui.create_line_edit(name='manual_drift_y_value', width = 50)
        self.start_manual_pb = ui.create_push_button(name='start_manual_pb', text='Start', on_clicked='start_manual_correction')

        self.manual_x_row = ui.create_row(self.manual_drift_x_label, self.manual_drift_x_value, ui.create_stretch(),
                                          self.time_interval_manual_label, self.time_interval_manual_value)
        self.manual_y_row = ui.create_row(self.manual_drift_y_label, self.manual_drift_y_value, ui.create_stretch())
        self.manual_pb_row = ui.create_row(self.start_manual_pb, ui.create_stretch(), self.note_manual_label)
        self.manual_group = ui.create_group(name = 'manual_group', title='Manual correction', content=ui.create_column(
            self.manual_x_row,
            self.manual_y_row,
            self.manual_pb_row,
            ui.create_stretch()
        ))

        #Shifters
        self.shifter_dim1_label = ui.create_label(name='shifter_dim1_label', text='Shifter.x: ')
        self.shifter_dim1_value = ui.create_label(name='shifter_dim1_value', text='@binding(shifter_u)')
        self.shifter_dim1_calib_label = ui.create_label(name='shifter_dim1_calib_label', text='Shifter.x (100 nm): ')
        self.shifter_dim1_calib_value = ui.create_label(name='shifter_dim1_calib_value', text='@binding(shifter_u_calib)')

        self.shifter_dim2_label = ui.create_label(name='shifter_dim2_label', text='Shifter.y: ')
        self.shifter_dim2_value = ui.create_label(name='shifter_dim2_value', text='@binding(shifter_v)')
        self.shifter_dim2_calib_label = ui.create_label(name='shifter_dim2_calib_label', text='Shifter.y (100 nm): ')
        self.shifter_dim2_calib_value = ui.create_label(name='shifter_dim2_calib_value', text='@binding(shifter_v_calib)')

        self.shifter_dim1_row = ui.create_row(self.shifter_dim1_label, self.shifter_dim1_value, ui.create_spacing(100),
                                              self.shifter_dim1_calib_label, self.shifter_dim1_calib_value, ui.create_stretch())
        self.shifter_dim2_row = ui.create_row(self.shifter_dim2_label, self.shifter_dim2_value, ui.create_spacing(100),
                                              self.shifter_dim2_calib_label, self.shifter_dim2_calib_value, ui.create_stretch())

        self.calib_shifter_dim1_label = ui.create_label(name = 'calib_shifter_dim1_label', text='Shiter.x step: ')
        self.calib_shifter_dim1_value = ui.create_line_edit(name='calib_shifter_dim1_value', width=50)
        self.calib_shifter_dim2_label = ui.create_label(name='calib_shifter_dim2_label', text='Shiter.y step: ')
        self.calib_shifter_dim2_value = ui.create_line_edit(name='calib_shifter_dim2_value', width=50)
        self.calib_shifter_row = ui.create_row(self.calib_shifter_dim1_label, self.calib_shifter_dim1_value,
                                               ui.create_stretch(),
                                               self.calib_shifter_dim2_label, self.calib_shifter_dim2_value)

        self.calib_dim1_pb = ui.create_push_button(name='calib_dim1_pb', text='Displace X',
                                                   on_clicked='displace_shifter')
        self.calib_dim2_pb = ui.create_push_button(name='calib_dim1_pb', text='Displace Y',
                                                   on_clicked='displace_shifter')
        self.calib_reset_pb = ui.create_push_button(name='calib_reset_pb', text='Reset Shifter',
                                                   on_clicked='reset_shifter')

        self.shifter_dim1_pb = ui.create_push_button(name='shifter_dim1_pb', text='Ok 100 nm X',
                                                     on_clicked='set_calibration')
        self.shifter_dim2_pb = ui.create_push_button(name='shifter_dim2_pb', text='Ok 100 nm Y',
                                                     on_clicked='set_calibration')

        self.calib_dim_row = ui.create_row(self.calib_dim1_pb, self.shifter_dim1_pb, ui.create_stretch(), self.calib_reset_pb,
                                           ui.create_stretch(), self.calib_dim2_pb, self.shifter_dim2_pb)

        self.third_group = ui.create_group(name='third_group', title='Shifters',
                                            content=ui.create_column(self.shifter_dim1_row, self.shifter_dim2_row,
                                                                     self.calib_shifter_row, self.calib_dim_row))



        self.cross_tab = ui.create_tab(label='Drift Correction',
                                       content=ui.create_column(self.first_group, self.second_group, self.manual_group,
                                                                self.third_group,
                                                                ui.create_stretch()))
        # End of cross-correlation tab

        self.collection_tabs = ui.create_tabs(self.hyperspytab, self.cross_tab)

        self.ui_view = ui.create_column(self.collection_tabs)
        #self.ui_view = ui.create_column(self.general_group,
        #                                self.hspec_group, self.d2_group, self.last_row)

def create_panel(document_controller, panel_id, properties):
    ui_handler = handler(document_controller)
    ui_view = View()
    panel = Panel.Panel(document_controller, panel_id, properties)

    finishes = list()
    panel.widget = Declarative.construct(document_controller.ui, None, ui_view.ui_view, ui_handler, finishes)

    for finish in finishes:
        finish()
    if ui_handler and hasattr(ui_handler, "init_handler"):
        ui_handler.init_handler()
    return panel


def run() -> None:
    panel_id = "Orsay Tools"  # make sure it is unique, otherwise only one of the panel will be displayed
    name = _("Orsay Tools")
    Workspace.WorkspaceManager().register_panel(create_panel, panel_id, name, ["left", "right"], "left")
