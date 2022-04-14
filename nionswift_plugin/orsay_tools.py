# standard libraries
import gettext
from nion.swift import Panel
from nion.swift import Workspace

from nion.ui import Declarative
from nion.ui import UserInterface
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import HardwareSource
from nion.swift.model import DataItem
from nion.swift.model import Utility
from nion.swift import Facade

from . import orsay_data

import os
import json
import logging
import numpy
import datetime

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

        self.event_loop = document_controller.event_loop
        self.document_controller = document_controller


    def init_handler(self):
        self.event_loop.create_task(self.do_enable(True, ['']))


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
        self.event_loop.create_task(self.do_enable(False, ["init_pb", 'host_value', 'port_value', 'abt_pb']))

    def prepare_widget_disable(self, value):
        self.event_loop.create_task(self.do_enable(False, ["init_pb", 'host_value', 'port_value', 'abt_pb']))

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

    def correct_gain(self, widget):
        pass

    def flip_signal(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            self.gd.flip()
            self.event_loop.create_task(self.data_item_show(self.gd.get_di()))

    def _align_chrono(self, align=True, bin=False):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            if align: self.gd.align_zlp()
            if bin:
                self.event_loop.create_task(self.data_item_show(self.gd.get_di(sum_inav=True)))
            else:
                self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def align_zlp(self, widget):
        self._align_chrono()

    def align_bin_chrono(self, widget):
        self._align_chrono(align=False, bin=True)

    def interpolate(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            self.gd.interpolate()
            self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

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

    def deconvolve_rl_hspec(self, widget):
        self._deconvolve_hspec('Richardson lucy')

    def _deconvolve_hspec(self, type):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            logging.info('***PANEL***: Must select to spectra for deconvolution.')
        else:
            dis = self._pick_dis()  # Multiple data items
            val_spec = list()
            if dis and len(dis) == 2:
                self.gd = orsay_data.HspySignal1D(dis[0].data_item)  # hspec
                self.spec_gd = orsay_data.HspySignal1D(dis[1].data_item) #spec
                self.gd.deconvolution(self.spec_gd, type)
                self.event_loop.create_task(self.data_item_show(self.gd.get_di()))
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')


    def _general_actions(self, type, which):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        def fitting_action(hspy_signal, val):
            if which == 'gaussian':
                new_di = hspy_signal.plot_gaussian(val)
            elif which == 'lorentzian':
                new_di = hspy_signal.plot_lorentzian(val)
            self.event_loop.create_task(self.data_item_show(new_di))

        def remove_background_action(hspy_signal, val):
            hspy_signal.remove_background(val, which)
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        if type == 'fitting':
            action = fitting_action
        elif type == 'remove_background':
            action = remove_background_action
        else:
            raise Exception("***PANEL***: No action function was selected. Please check the correct type.")

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            for graphic in self.__current_DI.graphics:
                if graphic.type == 'rect-graphic':  # This is a hyperspectral image
                    logging.info('***PANEL***: Hyperspectrum selected. If you wish to fit, please select two'
                                 'data items.')
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
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')

    def remove_background_pl(self, widget):
        self._general_actions('remove_background', 'Power law')

    def remove_background_off(self, widget):
        self._general_actions('remove_background', 'Offset')

    def fit_gaussian(self, widget):
        self._general_actions('fitting', 'gaussian')

    def fit_lorentzian(self, widget):
        self._general_actions('fitting', 'lorentzian')

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

class View:

    def __init__(self):
        ui = Declarative.DeclarativeUI()

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
        self.interpolate_pb = ui.create_push_button(text='Interpolate', name='interpolate_pb', on_clicked='interpolate')
        self.align_zlp_pb = ui.create_push_button(text='Align ZLP', name='align_zlp_pb', on_clicked='align_zlp')
        self.cgain_pb = ui.create_push_button(text='Correct Gain', name='cgain_pb',
                                              on_clicked='correct_gain')
        self.flip_pb = ui.create_push_button(text='Flip signal', name='flip_pb',
                                              on_clicked='flip_signal')

        self.background_text = ui.create_label(text='Background Removal: ', name='background_text')
        self.remove_pl_pb = ui.create_push_button(text='Power Law', name='remove_pl_pb', on_clicked='remove_background_pl')
        self.remove_pl_offset = ui.create_push_button(text='Offset', name='remove_off_pb',
                                                  on_clicked='remove_background_off')

        self.pb_row = ui.create_row(self.simple_text, self.interpolate_pb, self.align_zlp_pb, self.cgain_pb,
                                    self.flip_pb, ui.create_stretch())
        self.pb_remove = ui.create_row(self.background_text, self.remove_pl_pb, self.remove_pl_offset, ui.create_stretch())
        self.pb_row_fitting = ui.create_row(self.fit_text, self.fit_gaussian_pb, self.fit_lorentzian_pb,
                                            ui.create_stretch())

        self.general_group = ui.create_group(title='General Tools', content=ui.create_column(
            self.pb_row, self.pb_remove, self.pb_row_fitting, ui.create_stretch()))

        #Chrono Group
        self.transform_text = ui.create_label(text='Transform to spectrum: ', name='transform_text')
        self.chrono_align_bin_pb = ui.create_push_button(text='Full binning', name='chrono_align_bin_pb',
                                                     on_clicked='align_bin_chrono')

        self.pb_row = ui.create_row(self.transform_text, self.chrono_align_bin_pb, ui.create_stretch())
        self.d1_chrono_group = ui.create_group(title='1D Chrono Tools', content=ui.create_column(
            self.pb_row, ui.create_stretch()
        ))

        #Hyperspectral group
        self.deconvolution_text = ui.create_label(text='Signal deconvolution: ', name='deconvolution_text')
        self.dec_rl_pb = ui.create_push_button(text='Richardson-Lucy', name='dec_rl_pb',
                                            on_clicked='deconvolve_rl_hspec')
        self.pb_row = ui.create_row(self.deconvolution_text, self.dec_rl_pb, ui.create_stretch())

        self.binning_text = ui.create_label(text='Bin data (x, y, E): ', name='binning_text')
        self.x_le = ui.create_line_edit(name='x_le', width=15)
        self.y_le = ui.create_line_edit(name='y_le', width=15)
        self.E_le = ui.create_line_edit(name='E_le', width=15)
        self.bin_pb = ui.create_push_button(text='Ok', name='bin_pb', on_clicked='hspy_bin')
        self.binning_row = ui.create_row(self.binning_text, self.x_le, self.y_le,
                                         self.E_le, self.bin_pb, ui.create_stretch())
        self.hspec_group = ui.create_group(title='Hyperspectral Image', content=ui.create_column(
            self.pb_row, self.binning_row, ui.create_stretch()
        ))

        self.last_text = ui.create_label(text='Orsay Tools v0.1.0', name='last_text')
        self.last_row = ui.create_row(ui.create_stretch(), self.last_text)

        self.ui_view = ui.create_column(self.general_group, self.d1_chrono_group,
                                        self.hspec_group, self.d2_group, self.last_row)

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
