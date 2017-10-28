"""This module provides a parser for the .rho files containing the input model
of Stella runs (typically located in <stella_root>/eve/run."""
from collections import Counter
import logging
import numpy as np
import pandas as pd
import astropy.units as units
from astropy.utils.decorators import lazyproperty


class Logger(object):
    def myLogger(self):
        logger = logging.getLogger('__name__')
        if not len(logger.handlers):
            logger.setLevel(logging.DEBUG)
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(levelname)s [%(asctime)s]: %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger


log = Logger()
logger = log.myLogger()


class eve(object):
    def __init__(self, fname):

        self.fname = fname
        try:
            self.fstream = open(fname, "r")
        except IOError:
            logger.error("Stella .rho file '{:s}' not found".format(fname))
            raise

        self.X = pd.DataFrame(columns=["H", "He", "C", "N", "O", "Ne", "Na",
                                       "Mg", "Al", "Si", "S", "Ar", "Ca",
                                       "Fe", "Ni", "Ni56", "Fe52", "Cr48"])

        self._parse_header()
        self._parse_grid_info()
        self._parse_main_data()

    def _parse_header(self):

        header_line = self.fstream.readline()
        header_line = header_line.replace("lg ", "lg")

        header_line = header_line.replace("(", "")
        header_line = header_line.replace(")", "")

        counts = Counter(header_line.rsplit())

        if counts["Ni"] == 2:
            header_line = header_line.replace("Ni", "Ni56", 1)

        self.labels = header_line.rsplit()

    def _parse_grid_info(self):

        info_line = self.fstream.readline()

        self.Nzones = int(info_line.rsplit()[0])

        self.time = float(info_line.rsplit()[1])

    def _parse_main_data(self):

        self.raw_data = np.loadtxt(self.fstream)

        for i, label in enumerate(self.labels):

            if label in self.X.columns:

                self.X[label] = 10**self.raw_data[:, i]

            else:

                setattr(self, "_{:s}".format(label), self.raw_data[:, i])

    @lazyproperty
    def r(self):
        return 10**self._lgr * units.cm

    @lazyproperty
    def mr(self):
        return self._mass * units.solMass

    @lazyproperty
    def dm(self):
        return (4 * np.pi / 3. *
                (self.r**3 - np.insert(self.r, 0, 0)[:-1]**3) *
                self.rho).to("solMass")

    @lazyproperty
    def u(self):
        return self._u * units.cm / units.s

    @lazyproperty
    def rho(self):
        return 10**self._lgrho * units.g / units.cm**3

    @lazyproperty
    def T(self):
        return 10**self._lgTp * units.K
