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

from . import gain_data

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

class DataItemLaserCreation():
    def __init__(self, title, array, which, start=None, final=None, pts=None, avg=None, step=None, delay=None,
                 time_width=None, start_ps_cur=None, ctrl=None, trans=None, is_live=True, cam_dispersion=1.0, cam_offset=0,
                 power_min=0, power_inc=1, **kwargs):
        self.acq_parameters = {
            "title": title,
            "which": which,
            "start_wav": start,
            "final_wav": final,
            "pts": pts,
            "averages": avg,
            "step_wav": step,
            "delay": delay,
            "time_width": time_width,
            "start_ps_cur": start_ps_cur,
            "control": ctrl,
            "initial_trans": trans
        }
        self.timezone = Utility.get_local_timezone()
        self.timezone_offset = Utility.TimezoneMinutesToStringConverter().convert(Utility.local_utcoffset_minutes())

        self.calibration = Calibration.Calibration()
        self.dimensional_calibrations = [Calibration.Calibration()]

        if which == 'WAV':
            self.calibration.units = 'nm'
        if which == 'POW':
            self.calibration.units = 'μW'
        if which == 'SER':
            self.calibration.units = '°'
        if which == 'PS':
            self.calibration.units = 'A'
        if which == 'transmission_as_wav':
            self.calibration.units = 'T'
            self.dimensional_calibrations[0].units = 'nm'
            self.dimensional_calibrations[0].offset = start
            self.dimensional_calibrations[0].scale = step
        if which == 'power_as_wav':
            self.calibration.units = 'μW'
            self.dimensional_calibrations[0].units = 'nm'
            self.dimensional_calibrations[0].offset = start
            self.dimensional_calibrations[0].scale = step
        if which == 'sEEGS/sEELS_power':
            self.calibration.units = 'A.U.'
            self.dimensional_calibrations[0].units = 'μW'
            self.dimensional_calibrations[0].offset = power_min
            self.dimensional_calibrations[0].scale = power_inc
        if which == 'sEEGS/sEELS':
            self.calibration.units = 'A.U.'
            self.dimensional_calibrations[0].units = 'nm'
            self.dimensional_calibrations[0].offset = start
            self.dimensional_calibrations[0].scale = step
        if which == "CAM_DATA":
            self.dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration()]
            self.dimensional_calibrations[0].units = 'nm'
            self.dimensional_calibrations[0].offset = start
            self.dimensional_calibrations[0].scale = (step) / avg
            self.dimensional_calibrations[1].units = 'eV'
        if which == "POWER_CAM_DATA":
            self.dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration()]
            self.dimensional_calibrations[0].units = 'μW'
            self.dimensional_calibrations[0].offset = 0
            self.dimensional_calibrations[0].scale = 1
            self.dimensional_calibrations[1].units = 'eV'
        if which == 'ALIGNED_CAM_DATA':
            self.dimensional_calibrations = [Calibration.Calibration(), Calibration.Calibration()]
            self.dimensional_calibrations[0].units = 'nm'
            self.dimensional_calibrations[0].offset = start
            self.dimensional_calibrations[0].scale = step
            self.dimensional_calibrations[1].units = 'eV'
            self.dimensional_calibrations[1].scale = cam_dispersion
            self.dimensional_calibrations[1].offset = cam_offset

        self.xdata = DataAndMetadata.new_data_and_metadata(array, self.calibration, self.dimensional_calibrations,
                                                           metadata=self.acq_parameters,
                                                           timezone=self.timezone, timezone_offset=self.timezone_offset)

        self.data_item = DataItem.DataItem()
        self.data_item.set_xdata(self.xdata)
        #self.data_item.define_property("title", title)
        self.data_item.title = title
        self.data_item.description = self.acq_parameters
        self.data_item.caption = self.acq_parameters

        if is_live: self.data_item._enter_live_state()

    def update_data_only(self, array: numpy.array):
        self.xdata = DataAndMetadata.new_data_and_metadata(array, self.calibration, self.dimensional_calibrations,
                                                           metadata = self.acq_parameters,
                                                           timezone=self.timezone, timezone_offset=self.timezone_offset)
        self.data_item.set_xdata(self.xdata)

    def fast_update_data_only(self, array: numpy.array):
        self.data_item.set_data(array)

    def set_cam_di_calibration(self, calib: Calibration.Calibration()):
        self.dimensional_calibrations[1] = calib

    def set_cam_di_calibratrion_from_di(self, di: DataItem):
        pass

    def set_dim_calibration(self):
        self.data_item.dimensional_calibrations = self.dimensional_calibrations


class gainhandler:

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
            self.gd = gain_data.HspyGain(self.__current_DI.data_item)
            self.gd.rebin()
            self.event_loop.create_task(self.data_item_show(self.gd.get_11_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def _align_chrono(self, align=True, bin=False, flip=False):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspySignal1D(self.__current_DI.data_item)
            if flip: self.gd.flip(axis=1)
            if align: self.gd.align_zlp()
            if bin:
                self.event_loop.create_task(self.data_item_show(self.gd.get_11_di(sum_inav=True)))
            else:
                self.event_loop.create_task(self.data_item_show(self.gd.get_11_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def align_zlp(self, widget):
        self._align_chrono()

    def align_bin_chrono(self, widget):
        self._align_chrono(align=False, bin=True)

    def align_bin_flip_chrono(self, widget):
        self._align_chrono(align=False, bin=True, flip=True)

    def interpolate(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspySignal1D(self.__current_DI.data_item)
            self.gd.interpolate()
            self.event_loop.create_task(self.data_item_show(self.gd.get_11_di()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def gain_profile_data_item(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspyGain(self.__current_DI.data_item)
            self.event_loop.create_task(self.data_item_show(self.gd.get_gain_profile()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def gain_profile_2d_data_item(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspyGain(self.__current_DI.data_item)
            self.event_loop.create_task(self.data_item_show(self.gd.get_gain_2d()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def fit_gaussian(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspySignal1D(self.__current_DI.data_item)
            for graphic in self.__current_DI.graphics:
                print(graphic.type)
                if graphic.type == 'rect-graphic':  # This is a hyperspectral image
                    print(dir(graphic))
                    print(dir(graphic.region))
                    print(graphic.bounds)
                if graphic.type == 'line-profile-graphic': #This is Chrono
                    val = (graphic.start[1], graphic.end[1])
                    new_di = self.gd.plot_gaussian(val)
                    self.event_loop.create_task(self.data_item_show(new_di))
                    return
                if graphic.type == 'interval-graphic': #This is 1D
                    val = graphic.interval
                    new_di = self.gd.plot_gaussian(graphic.interval)
                    self.event_loop.create_task(self.data_item_show(new_di))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def fit_lorentzian(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = gain_data.HspySignal1D(self.__current_DI.data_item)
            api_data_item = Facade.DataItem(self.__current_DI)
            for graphic in api_data_item.graphics:
                if graphic.graphic_type == 'interval-graphic':
                    new_di = self.gd.plot_lorentzian(graphic.interval)
                    self.event_loop.create_task(self.data_item_show(new_di))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

    def _pick_di(self):
        display_item = self.document_controller.selected_display_item
        #data_item = display_item.data_items[0] if display_item and len(display_item.data_items) > 0 else None
        #print(data_item)
        #print(display_item.data_item)
        return display_item

    def _pick_di_by_name(self):
        for data_items in self.document_controller.document_model._DocumentModel__data_items:
            if data_items.title == self.file_name_value.text:
                self.__current_DI = data_items

class gainView:

    def __init__(self):
        ui = Declarative.DeclarativeUI()

        #Gain group
        self.bin_laser_pb = ui.create_push_button(text='Bin laser', name='bin_laser_pb', on_clicked='bin_laser')
        self.profile_2d_pb = ui.create_push_button(text='2D Gain profile ', name='profile_2d_pb',
                                                on_clicked='gain_profile_2d_data_item')
        self.profile_pb = ui.create_push_button(text='Gain profile ', name='profile_pb', on_clicked='gain_profile_data_item')
        self.pb_row = ui.create_row(self.bin_laser_pb, self.profile_2d_pb, self.profile_pb, ui.create_stretch())

        self.d2_group = ui.create_group(title='2D Tools', content=ui.create_column(
            self.pb_row, ui.create_stretch()))

        #General Group
        self.fit_gaussian_pb = ui.create_push_button(text='Fit Gaussian', name='fit_gaussian_pb', on_clicked='fit_gaussian')
        self.fit_lorentzian_pb = ui.create_push_button(text='Fit Lorentzian', name='fit_lorentzian_pb',
                                                     on_clicked='fit_lorentzian')
        self.interpolate_pb = ui.create_push_button(text='Interpolate', name='interpolate_pb', on_clicked='interpolate')
        self.align_zlp_pb = ui.create_push_button(text='Align ZLP', name='align_zlp_pb', on_clicked='align_zlp')

        self.pb_row = ui.create_row(self.align_zlp_pb, self.interpolate_pb,
                                    self.fit_gaussian_pb, self.fit_lorentzian_pb,
                                    ui.create_stretch())
        self.general_group = ui.create_group(title='General Tools', content=ui.create_column(
            self.pb_row, ui.create_stretch()))

        #Chrono Group
        self.chrono_align_bin_pb = ui.create_push_button(text='Bin Chrono', name='chrono_align_bin_pb',
                                                     on_clicked='align_bin_chrono')
        self.chrono_align_bin_flip_pb = ui.create_push_button(text='Bin w/ flip Chrono', name='chrono_align_bin_flip_pb',
                                                         on_clicked='align_bin_flip_chrono')

        self.pb_row = ui.create_row(self.chrono_align_bin_pb, self.chrono_align_bin_flip_pb, ui.create_stretch())
        self.d1_chrono_group = ui.create_group(title='1D Chrono Tools', content=ui.create_column(
            self.pb_row, ui.create_stretch()
        ))


        self.ui_view = ui.create_column(self.general_group, self.d1_chrono_group, self.d2_group)

def create_spectro_panel(document_controller, panel_id, properties):
    ui_handler = gainhandler(document_controller)
    ui_view = gainView()
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
    Workspace.WorkspaceManager().register_panel(create_spectro_panel, panel_id, name, ["left", "right"], "left")
