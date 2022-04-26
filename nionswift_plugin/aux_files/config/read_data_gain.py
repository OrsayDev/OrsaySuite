import os
import logging
import hyperspy.api as hs
class FileManager:
    def __init__(self, ht, threshold):
        self.ht = ht
        self.ht = threshold

        if (os.name == 'posix'): #this is linux
            self.path_gain = './OrsaySuite/nionswift_plugin/aux_files/config/Gain_Merlin/'
            list_dir = os.listdir(self.path_gain)
            for dir in list_dir:
                if dir.startswith(str(ht)):
                    dir_gain = dir

            self.path_gain = self.path_gain + '/' + dir_gain+'/'
            list_dir = os.listdir(self.path_gain)

            for file in list_dir:
                if str(threshold) in file and '.hspy' in file:
                    self.filename = file
                    pass



            abs_path = os.path.abspath('/home/ProgramData/Microscope/' + self.filename + '.hspy')

            if os.path.isfile(abs_path+self.filename):
                self.gain = hs.load(abs_path + self.filename)

            elif os.path.isfile(self.path_gain + self.filename):
                self.gain = hs.load(self.path_gain + self.filename)


        if (os.name == 'nt'):  # this is Windows
            print('Read_data_gain : Not implemented for Windows')



