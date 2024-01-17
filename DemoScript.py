from nion.typeshed import API_1_0 as API
from nion.typeshed import UI_1_0 as UI
import nionswift_plugin.orsay_suite.orsay_data as OD

#Hyperspy import for Typing
from hyperspy._signals.signal1d import Signal1D
from hyperspy._signals.eels import EELSSpectrum


api = api_broker.get_api(API.version, UI.version)  # type: API
window = api.application.document_windows[0]

### Getting the DataItem ###
data_item: API.DataItem = window.target_data_item

### Getting a hyperspy object ###
wrapper_data: OD.HspySignal1D = OD.HspySignal1D(data_item)
hspy_data:  Signal1D = wrapper_data.hspy_gd

### Doing some stuff using Hyperspy ###
hspy_data.align1D(start=2.0, end=2.5, show_progressbar=True)

### Getting a calibrated DataItem back ###
data_and_metadata = wrapper_data.get_data_and_metadata(hspy_data)
new_data_item: API.DataItem = api.library.create_data_item_from_data_and_metadata(data_and_metadata,"MyHspyData!")
api.application.document_windows[0].display_data_item(new_data_item)