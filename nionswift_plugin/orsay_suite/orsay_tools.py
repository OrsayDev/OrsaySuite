# standard libraries
import gettext
from nion.swift import Panel
from nion.swift import Workspace

from nion.ui import Declarative
from nion.ui import UserInterface
from nion.data import Calibration
from nion.data import DataAndMetadata
from nion.swift.model import DataItem
from nion.swift.model import Utility

from . import orsay_data

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

        self.event_loop = document_controller.event_loop
        self.document_controller = document_controller


    def init_handler(self):
        self.event_loop.create_task(self.do_enable(True, ['']))
        self.x_le.text = '1'
        self.y_le.text = '1'
        self.E_le.text = '1'
        self.comp_le.text = '3'
        self.int_le.text = '10'


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

    def correct_junctions(self, widget):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            corrected_data = self.gd.detector_junctions()
            self.event_loop.create_task(self.data_item_show(corrected_data))
        else:
            logging.info('***PANEL***: Could not find referenced Data Item.')

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
        try:
            val = int(self.int_le.text)
            self._deconvolve_hspec('Richardson lucy', val)
        except ValueError:
            logging.info("***PANEL***: Interaction value must be integer.")




    def _deconvolve_hspec(self, type, interactions):
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
                new_di = self.gd.deconvolution(self.spec_gd, type, interactions)
                self.event_loop.create_task(self.data_item_show(new_di))
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')


    def _general_actions(self, type, which):
        self.__current_DI = None

        self.__current_DI = self._pick_di()

        def align_zlp_action(hspy_signal, val):
            hspy_signal.align_zlp_signal_range(val)
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        def fitting_action(hspy_signal, val):
            if which == 'gaussian':
                new_di = hspy_signal.plot_gaussian(val)
            elif which == 'lorentzian':
                new_di = hspy_signal.plot_lorentzian(val)
            self.event_loop.create_task(self.data_item_show(new_di))

        def remove_background_action(hspy_signal, val):
            hspy_signal.remove_background(val, which)
            self.event_loop.create_task(self.data_item_show(hspy_signal.get_di()))

        def decomposition_action(hspy_signal, val):
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
        else:
            raise Exception("***PANEL***: No action function was selected. Please check the correct type.")

        if self.__current_DI:
            self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)
            for graphic in self.__current_DI.graphics:
                if graphic.type == 'rect-graphic':  # This is a hyperspectral image
                    logging.info('***PANEL***: Hyperspectrum selected. If you wish to perform this action, please select two '
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
                        return
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

    def align_zlp(self, widget):
        self._general_actions('align_zlp', 'None')

    def pca(self, widget):
        try:
            val = int(self.comp_le.text)
            self._general_actions('decomposition', val)
        except ValueError:
            logging.info("***PANEL***: Interaction value must be integer.")

    def pca_full(self, widget):
        self._pca_full()

    def _pca_full(self):
        try:
            val = int(self.comp_le.text)
            self.__current_DI = self._pick_di()
            if self.__current_DI:
                self.gd = orsay_data.HspySignal1D(self.__current_DI.data_item)  # hspec
                var, new_di = self.gd.signal_decomposition(components=val, mask=False)
                self.event_loop.create_task(self.data_item_show(new_di))
                self.event_loop.create_task(self.data_item_show(var))
            else:
                logging.info('***PANEL***: Could not find referenced Data Item.')
        except ValueError:
            logging.info("***PANEL***: Components value must be integer.")

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

        self.ui_view = ui.create_column(self.general_group,
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
