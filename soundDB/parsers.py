# Python 2 and 3 cross-compatibility:
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import (bytes, str, int, dict, object, range, map, filter, zip, round, pow, open)
from future.utils import (iteritems, itervalues)
from past.builtins import basestring

from .accessor import Accessor

import pandas as pd
import numpy as np
import re

"""
This file containes subclasses of ``Accessor`` for reading each specific kind of data file (NVSPL, SRCID, etc.).

Each subclass should specify:

    - ``endpointName``: name of the ``iyore.Endpoint`` in a Dataset where the kind of data the Accessor handles is found

    - ``parse()``: method which, given an ``iyore.Entry``, parses that single data file into a pandas structure and returns it

    - ``prepareState()`` (optional): method which prepares a ``state`` object to be passed between repeated calls of ``parse``,
                                     which is only useful for data composed of many files per site (like NVSPL and audibility)

The subclasses should only be defined here, *not* instantiated. In ``__init__.py``, all subclasses of ``Accessor``
in ``parsers.py`` are automatically detected and instantiated. Then each instance is added to the top-level ``soundDB`` namespace
under its ``endpointName``.
"""

# def initializeFastNVSPL(endpoint, endpointParams, timestamps= None, columns= None):

#     columnNames = ['SiteID', 'date', '12.5', '15.8', '20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000', '12500', '16000', '20000', 'dBA', 'dBC', 'dBF', 'Voltage', 'WindSpeed', 'WindDir', 'TempIns', 'TempOut', 'Humidity', 'INVID', 'INSID', 'GChar1', 'GChar2', 'GChar3', 'AdjustmentsApplied', 'CalibrationAdjustment', 'GPSTimeAdjustment', 'GainAdjustment', 'Status']
#     index_index = 1 # Default position of the index column (STime)
    
#     if columns == "spl":
#         # Convenience option for just getting columns containing actual SPL data
#         columns = columnNames[:38]
#     elif columns is not None:
#         # Ensure we read the STime (date) column, otherwise indexing will be messed up
#         if all(isinstance(column, basestring) for column in columns):
#             columnNamesSet = set(columnNames)
#             if all(column in columnNamesSet for column in columns):
#                 if "date" not in columns:
#                     columns = ["date"] + columns
#                 index_index = 0
#             else:
#                 raise TypeError("These column names are not found in an NVSPL DataFrame: {}".format(", ".join(set(columns).difference(columnNamesSet))))
#         elif all(isinstance(column, int) for column in columns):
#             if 1 not in columns:
#                 columns = [1] + columns
#                 columns.sort()
#             index_index = columns.index(1)
#         else:
#             raise TypeError("columns must be a list of strings or of integers")

# def fastNVSPL(nvsplFileEntry):
#     timestamps, columns, index_index = params

#     columnNames = ['SiteID', '12.5', '15.8', '20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', '800', '1000', '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000', '10000', '12500', '16000', '20000', 'dbA', 'dbC', 'dbF', 'Voltage', 'WindSpeed', 'WindDir', 'TempIns', 'TempOut', 'Humidity', 'INVID', 'INSID', 'GChar1', 'GChar2', 'GChar3', 'AdjustmentsApplied', 'CalibrationAdjustment', 'GPSTimeAdjustment', 'GainAdjustment', 'Status']

#     df = pd.read_csv(str(nvsplFileEntry),
#                      engine= 'c',
#                      # sep= ',',
#                      parse_dates= True,
#                      index_col= index_index,
#                      infer_datetime_format= True,
#                      usecols= columns,
#                      header= 0,
#                      names= columnNames
#                      )




class NVSPL(Accessor):
    """
    NVSPL-specific Parameters
    -------------------------

    timestamps : iterable of datetime-like, or pandas.DatetimeIndex

        Specific seconds to read data from

    columns : list of str or int

        Columns to read, either by name or number

    Resulting DataFrame
    -------------------

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3600 entries, 0 to 3599
    Data columns (total 44 columns):
    12.5         float64
    15.8         float64
    20           float64
    25           float64
    31.5         float64
    40           float64
    50           float64
    63           float64
    80           float64
    100          float64
    125          float64
    160          float64
    200          float64
    250          float64
    315          float64
    400          float64
    500          float64
    630          float64
    800          float64
    1000         float64
    1250         float64
    1600         float64
    2000         float64
    2500         float64
    3150         float64
    4000         float64
    5000         float64
    6300         float64
    8000         float64
    10000        float64
    12500        float64
    16000        float64
    20000        float64
    dbA          float64
    dbC          float64
    dbF          float64
    Voltage      float64
    WindSpeed    float64
    WindDir      float64
    TempIns      float64
    TempOut      float64
    Humidity     float64
    INVID        float64
    INSID        float64
    dtypes: float64(44)
    memory usage: 1.2 MB
    """

    endpointName = "nvspl"

    def parse(nvsplFileEntry, state= (None, None, 1)):
        timestamps, columns, index_index = state

        df = pd.read_csv(str(nvsplFileEntry),
                         engine= 'c',
                         # sep= ',',
                         parse_dates= True,
                         index_col= index_index,
                         infer_datetime_format= True,
                         usecols= columns
                         )

        # Make column names slightly nicer
        df.index.name = "date"
        renamedColumns = { column: column.replace('H', '').replace('p', '.') for column in df.columns if re.match(r"H\d+p?\d*", column) is not None }
        df.rename(columns= renamedColumns, inplace= True)

        # TODO: rename dbA, dbT to dBA, dBT for consistencty
        # TODO: potentially drop siteID column

        # Coerce numeric columns to floats, in case of "-Infinity" values
        try:
            numericCols = [
                '12.5', '15.8', '20', '25', '31.5', '40', '50', '63', '80', '100',
                '125', '160', '200', '250', '315', '400', '500', '630', '800', '1000',
                '1250', '1600', '2000', '2500', '3150', '4000', '5000', '6300', '8000',
                '10000', '12500', '16000', '20000', 'dbA', 'dbC', 'dbF',
                'Voltage','WindSpeed', 'WindDir', 'TempIns', 'TempOut', 'Humidity'
             ]
            presentNumericCols = df.columns.intersection(numericCols)
            if len(presentNumericCols) > 0:
                # df[presentNumericCols] = df[presentNumericCols].convert_objects(convert_dates= False, convert_numeric= True, convert_timedeltas= False, copy= False)
                df[presentNumericCols].astype('float32', copy= False, raise_on_error= False)

        except KeyError:
            pass

        return df

    def prepareState(endpoint, endpointParams, timestamps= None, columns= None):

        if timestamps is not None:
            # make dict of endpoint restriction args
            # make dict of argtuple: list of secs to read
            pass

        index_index = 1 # Default position of the index column (STime)
        if columns is not None:
            # Ensure we read the STime (date) column, otherwise indexing will be messed up
            # TODO: conversion between reasonable column names and 12p5h style names
            if all(isinstance(column, basestring) for column in columns):
                if "STime" not in columns:
                    columns = ["STime"] + columns
                index_index = 0
            elif all(isinstance(column, int) for column in columns):
                if 1 not in columns:
                    columns = [1] + columns
                    columns.sort()
                index_index = columns.index(1)
            else:
                raise TypeError("columns must be a list of strings or of integers")

        return (timestamps, columns, index_index)


class SRCID(Accessor):
    """
    Resulting DataFrame
    -------------------

    The ``nvsplDate``, ``hr``, and ``secs`` columns are combined into a single DatetimeIndex for the DataFrame and dropped.
    The ``len`` column (length of the noise event) is converted to a pandas Timedelta.

    <TODO>
    """

    endpointName = "srcid"

    def parse(entry):

        with open(str(entry)) as f:
            # Determine version; older versions immediately start with header, newer has version comment
            firstline = f.readline()
            if not firstline.startswith(r"%%"):
                f.seek(0)   # Rewind, as we just consumed the header row.
                            # Otherwise, we consumed the comment row, which is fine

            data = pd.read_csv(f,
                                engine= "c",
                                sep= "\t",
                                # skiprows= 1,
                                parse_dates= False)

        # Combine nvsplDate, hr, secs columns into one DatetimeIndex
        dates = pd.to_datetime(data.nvsplDate, infer_datetime_format= True)
        hrs   = pd.to_timedelta(data.hr, unit= "h")
        secs  = pd.to_timedelta(data.secs, unit= "s")

        data.drop(["nvsplDate", "hr", "secs"], axis= 1, inplace= True)
        data.index = dates + hrs + secs

        # Turn len into timedelta
        data.len = pd.to_timedelta(data.len, unit= "s")

        if 'sID' in data.columns:
            # Some bizzare old files have a different name for srcID (really only DENAUSLC2008 so far)
            data.rename(columns= {'sID': 'srcID'}, inplace= True)

        # Days with no noise events are entered with the `MaxSPLt` and `SELt` columns skipped,
        # so they get filled with userName and tagDate, giving them a mixed type instead of float
        # Resolve this by finding zero-noise rows and setting all their values to NaN (more appropriate than 0)
        if 'MaxSPLt' in data.columns:
            if data.MaxSPLt.dtype == np.object:
                # There are noise-free days in this dataset
                converted = data[['MaxSPLt', 'SELt']].convert_objects(convert_numeric= True)
                noisefree = converted.isnull().all(axis= 1)
                data.loc[noisefree, ["userName", "tagDate"]] = data.loc[noisefree, ['MaxSPLt', 'SELt']].values
                data[['MaxSPLt', 'SELt']] = converted

                nanCols = data.columns.difference(("userName", "tagDate"))
                data.loc[ noisefree, nanCols ] = np.nan
        
        # Parse tagDate to datetime (though old versions don't have tagDate)
        if 'tagDate' in data.columns:
            data.tagDate = pd.to_datetime(data.tagDate, infer_datetime_format= True)

        return data

class LoudEvents(Accessor):
    """
    Resulting Panel
    -------------------

    * The items axis (axis 0) is ["above", "all", "percent"]. So you'd use ``events["above"]`` to get a
      sub-DataFrame of events that exceeded $L_{nat}$, where rows are indexed by date, and columns from 0 to 23 hours.
    * The major_axis (axis 1) is ``date``: pandas DateTime objects
    * The minor_axis (axis 2) is ``hour``: 0 to 23

    <TODO>
    """


    endpointName = "loudevents"


    def parse(entry):

        data = pd.read_csv(str(entry),
                            engine= "c",
                            sep= "\t",
                            index_col= 0,
                            parse_dates= True,
                            infer_datetime_format= True)

        if data.index.name is not None: data.index.name = data.index.name.lower()
        data.columns = list(range(24)) * 3

        paneldata = pd.Panel({
                                "above": data.iloc[:, 0:24],
                                "all": data.iloc[:, 24:48],
                                "percent": data.iloc[:, 48:72]
                            })

        paneldata.minor_axis.name = "hour"

        return paneldata