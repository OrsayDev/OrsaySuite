from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UI
from nion.typeshed import Interactive_1_0 as Interactive
import nionswift_plugin.orsay_suite.orsay_data as OD

#Hyperspy import for Typing
from hyperspy._signals.signal1d import Signal1D

def script_main(api_broker):
    interactive: Interactive.Interactive = api_broker.get_interactive(Interactive.version)
    api = api_broker.get_api(API.version, UI.version)  # type: API
    window = api.application.document_windows[0]

    ### Getting the DataItem ###
    data_item: API.DataItem = window.target_data_item

    ### Getting a hyperspy object ###
    """
    wrapper data is the entire object. It has the Nionswift DataItem + the Hspy Object
    hspy_data is only the hspy data. You are going to modify/replace this data
    """
    wrapper_data: OD.HspySignal1D = OD.HspySignal1D(data_item)
    hspy_data:  Signal1D = wrapper_data.hspy_gd

    interactive.print_info("Original metadata : {}".format(hspy_data.axes_manager))

    #Integration windows
    # Ask for the limits
    lim1 = interactive.get_float("Start of the energy window: ")
    lim2 = interactive.get_float("End of the energy window: ")
    # interactive.
    integrate = hspy_data.integrate1D(axis=-1)
    # interactive.print_info("Integrate shape : {}".format(integrate.data.shape))
    new_data_item: API.DataItem = api.library.create_data_item_from_data_and_metadata(integrate.data, title="Integratation")
    api.application.document_windows[0].display_data_item(new_data_item)
