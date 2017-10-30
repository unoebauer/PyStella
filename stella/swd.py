import numpy as np
import pandas as pd
import astropy.units as units


class swd(object):
    def __init__(self, fname):

        self.fname = fname
        try:
            self.fstream = open(fname, "r")
        except IOError:
            raise

        self._parse_main_data()

    def _parse_main_data(self):

        # This should be replaced by the actual column names
        labels = list("ABCDEFGHIJKLM")

        self.raw_data = pd.read_table(self.fstream, header=None, sep='\s+',
                                      names=labels)
