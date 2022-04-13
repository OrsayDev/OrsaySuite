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
            self.do_enable(True, ['init_pb', 'host_value', 'port_value', 'plot_power_wav', 'align_zlp_max', 'align_zlp_fit', 'smooth_zlp',
                                  'process_eegs_pb',
                                  'process_power_pb', 'fit_pb',
                                  'cancel_pb'
                                  ]))

    def show_det(self, xdatas, mode, nacq, npic, show):

        for data_items in self.document_controller.document_model._DocumentModel__data_items:
            if data_items.title == 'Laser Wavelength ' + str(nacq):
                nacq += 1

        #while self.document_controller.document_model.get_data_item_by_title(
        #        'Laser Wavelength ' + str(nacq)) is not None:
        #    nacq += 1  # this puts always a new set even if swift crashes and counts perfectly

        for i, xdata in enumerate(xdatas):
            data_item = DataItem.DataItem()
            data_item.set_xdata(xdata)
            # this nacq-1 is bad. Because Laser Wavelength DI would already be created, this is the easy solution.
            # so, in order for this to work you need to create laser wavelength before creating my haadf/bf DI
            if mode == 'init' or mode == 'end': data_item.define_property("title",
                                                                          mode + '_det' + str(i) + ' ' + str(nacq - 1))
            # this nacq-1 is bad. Because Laser Wavelength DI would already be created, this is the easy solution.
            if mode == 'middle': data_item.define_property("title",
                                                           mode + str(npic) + '_det' + str(i) + ' ' + str(nacq - 1))

            if show: self.event_loop.create_task(self.data_item_show(data_item))

    def call_monitor(self):
        self.pow02_mon_array = numpy.zeros(200)
        self.pow02_mon_di = DataItemCreation("Power Fiber", self.pow02_mon_array, 1, [0], [1], ['time (arb. units)'])
        self.event_loop.create_task(self.data_item_show(self.pow02_mon_di.data_item))

    def append_monitor_data(self, value, index):
        power02 = value
        if index==0:
            self.pow02_mon_array = numpy.zeros(200)
        self.pow02_mon_array[index] = power02
        self.pow02_mon_di.fast_update_data_only(self.pow02_mon_array)

    def call_data(self, nacq, pts, avg, start, end, step, cam_acq, **kwargs):

        if len(cam_acq.data.shape) > 1:
            cam_pixels = cam_acq.data.shape[1]
            cam_calibration = cam_acq.get_dimensional_calibration(1)
        else:
            cam_pixels = cam_acq.data.shape[0]
            cam_calibration = cam_acq.get_dimensional_calibration(0)

        self.cam_array = numpy.zeros((pts * avg, cam_pixels))
        self.avg = avg
        self.pts = pts

        for data_items in self.document_controller.document_model._DocumentModel__data_items:
            if data_items.title == 'Gain Data ' + str(nacq):
                nacq += 1

        # Power Meter call
        self.pow02_array = numpy.zeros(pts * avg)
        self.pow02_di = DataItemCreation("Power 02 " + str(nacq), self.pow02_array, 1, [0], [1], ['uW'])

        # CAMERA CALL
        if start == end and step == 0.0:
            self.cam_di = DataItemCreation('Gain Data ' + str(nacq), self.cam_array, 2,
                                           [0, cam_calibration.offset], [1 / avg, cam_calibration.scale],
                                           ['nm', 'eV'], title='Gain Data ' + str(nacq), start_wav=start, end_wav=end,
                                           pts=pts, averages=avg, **kwargs)
        else:
            self.cam_di = DataItemCreation('Gain Data ' + str(nacq), self.cam_array, 2,
                                           [start, cam_calibration.offset], [step / avg, cam_calibration.scale],
                                           ['nm', 'eV'], title='Gain Data ' + str(nacq), start_wav=start, end_wav=end,
                                           pts=pts, averages=avg, **kwargs)
        self.event_loop.create_task(self.data_item_show(self.cam_di.data_item))

    def append_data(self, value, index1, index2, camera_data, update=True):

        if len(camera_data.data.shape)>1:
            cam_hor = numpy.sum(camera_data.data, axis=0)
        else:
            cam_hor = camera_data.data

        power02 = value
        self.pow02_array[index2 + index1 * self.avg] = power02
        self.cam_array[index2 + index1 * self.avg] = cam_hor  # Get raw data

        if update: self.cam_di.update_data_only(self.cam_array)

    def end_data_monitor(self):
        if self.pow02_mon_di:
            self.event_loop.create_task(self.data_item_exit_live(self.pow02_mon_di.data_item))

    def end_data(self):
        if self.pow02_di:
            self.event_loop.create_task(self.data_item_show(self.pow02_di.data_item))
            self.event_loop.create_task(self.data_item_exit_live(self.pow02_di.data_item))
        if self.cam_di: self.event_loop.create_task(self.data_item_exit_live(self.cam_di.data_item))

    def stop_function(self, wiget):
        self.instrument.Laser_stop_all()

    def grab_data_item(self, widget):
        self.__current_DI = None

        for data_items in self.document_controller.document_model._DocumentModel__data_items:
            if data_items.title == self.file_name_value.text:
                self.__current_DI = data_items
        if self.__current_DI:
            self.gd = gain_data.HspyGain(self.__current_DI)
            api_data_item = Facade.DataItem(self.__current_DI)
            for graphic in api_data_item.graphics:
                if graphic.graphic_type == 'interval-graphic':
                    new_di = self.gd.plot_gaussian(graphic.interval)
                    self.event_loop.create_task(self.data_item_show(new_di))




            #new_di = self.gd.plot_gaussian([575.0, 580.0])
            #self.gd.rebin_and_align()
            #new_di = DataItemCreation('Aligned_and_summed_'+self.gd.get_attr('title'), self.gd.get_data(), 2, self.gd.get_axes_offset_all(), self.gd.get_axes_scale_all(), self.gd.get_axes_units_all())
            #self.event_loop.create_task(self.data_item_show(new_di))
            #self.event_loop.create_task(self.data_item_show(self.gd.get_gain_profile()))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')


class gainView:

    def __init__(self):
        ui = Declarative.DeclarativeUI()

        ### BEGIN ANALYSIS TAB ##

        self.grab_pb = ui.create_push_button(text='Grab', name='grab_pb', on_clicked='grab_data_item')
        #self.grab_pb = ui.create_push_button(text='Align and sum ', name='grab_pb', on_clicked='grab_data_item')
        self.pb_row = ui.create_row(self.grab_pb, ui.create_stretch())

        self.file_name_value = ui.create_line_edit(name='file_name_value', width=150)
        self.file_name_row = ui.create_row(self.file_name_value, ui.create_stretch())

        self.pick_group = ui.create_group(title='Pick Tool', content=ui.create_column(
            self.file_name_row, self.pb_row, ui.create_stretch()))

        self.ana_tab = ui.create_tab(label='Gain Data', content=ui.create_column(
            self.pick_group, ui.create_stretch())
                                     )
        ## END ANALYSYS TAB

        self.tabs = ui.create_tabs(self.ana_tab)
        self.ui_view = ui.create_column(self.tabs)

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
    panel_id = "Laser"  # make sure it is unique, otherwise only one of the panel will be displayed
    name = _("Laser")
    Workspace.WorkspaceManager().register_panel(create_spectro_panel, panel_id, name, ["left", "right"], "left")
