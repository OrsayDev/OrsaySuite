import os
import logging
import hyperspy.api as hs
class FileManager:
    def __init__(self, ht, threshold):
        self.ht = ht
        self.ht = threshold
        self.filename = str(ht)+'kV_Thresh_'+str(threshold)+'.hspy'

        if (os.name == 'posix'): #this is linux
            abs_path = os.path.abspath('/home/ProgramData/Microscope/' + self.filename)
            try:
                with open(abs_path) as savfile:
                    self.abs_path = abs_path
                    self.gain = hs.load(abs_path + self.filename)
            except FileNotFoundError:
                abs_path = os.path.join(os.path.dirname(__file__)+'/Gain_Merlin/' +str(ht)+'kV', self.filename)  # this works
                self.abs_path = abs_path
                with open(abs_path) as savfile:
                    self.abs_path = abs_path
                    self.gain = hs.load(abs_path)

        if (os.name == 'nt'):  # this is Windows
            print('Read_data_gain : Not implemented for Windows')



