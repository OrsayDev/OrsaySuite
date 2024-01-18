from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UI
from nion.typeshed import Interactive_1_0 as Interactive
import nionswift_plugin.orsay_suite.orsay_data as OD

#Hyperspy import for Typing
from hyperspy._signals.eels import EELSSpectrum

def script_main(api_broker):
    interactive: Interactive.Interactive = api_broker.get_interactive(Interactive.version)
    api = api_broker.get_api(API.version, UI.version)  # type: API
    window = api.application.document_windows[0]

    ### Getting the DataItem ###
    interactive.confirm_ok_cancel("Select the Hyperspectral image and press OK.")
    data_item: API.DataItem = window.target_data_item
    print("Done.")

    interactive.confirm_ok_cancel("Select the spectre and press OK.")
    data_item2: API.DataItem = window.target_data_item
    print("Done.")

    interactions = interactive.get_integer("How many interactions?", 3)

    ### Getting a hyperspy object ###
    """
    wrapper data is the entire Hyperspectral. It has the Nionswift DataItem + the Hspy Object.
    wrapper_data_spec is the reference spectra.
    """
    wrapper_data: OD.HspySignal1D = OD.HspySignal1D(data_item)
    hspy_data:  Signal1D = wrapper_data.hspy_gd

    wrapper_data_spec: OD.HspySignal1D = OD.HspySignal1D(data_item2)
    hspy_data_spec: EELSSpectrum = wrapper_data_spec.hspy_gd

    ### Doing some stuff using Hyperspy ###
    """
    Documentation for the one Richardson Lucy Deconvolution
    https://hyperspy.org/hyperspy-doc/v1.7/api/hyperspy._signals.eels.html#hyperspy._signals.eels.EELSSpectrum.richardson_lucy_deconvolution
    """
    spim_decomvoluted = hspy_data.richardson_lucy_deconvolution(hspy_data_spec, interactions, show_progressbar=True)

    ### Getting a calibrated DataItem back ###
    """
    wrapper data is used to recover everything from the initial unmodified DataItem (metadata + caption + calibration, etc)
    """
    data_and_metadata = wrapper_data.get_data_and_metadata(spim_decomvoluted)
    new_data_item: API.DataItem = api.library.create_data_item_from_data_and_metadata(data_and_metadata,"MyHspyData!")
    api.application.document_windows[0].display_data_item(new_data_item)