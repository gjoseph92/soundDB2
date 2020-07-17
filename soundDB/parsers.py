# Python 2 and 3 cross-compatibility:
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import (bytes, str, int, dict, object, range, map, filter, zip, round, pow, open)
from future.utils import (iteritems, itervalues)
from past.builtins import basestring

from .accessor import Accessor

import pandas as pd
import numpy as np
import xarray as xr
import re
import collections
import warnings

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

    Example Resulting DataFrame
    ---------------------------

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 3600 entries, 2015-05-15 00:00:00 to 2015-05-15 00:59:59
    Data columns (total 53 columns):
    SiteID                   3600 non-null object
    12.5                     3600 non-null float64
    15.8                     3600 non-null float64
    20                       3600 non-null float64
    25                       3600 non-null float64
    31.5                     3600 non-null float64
    40                       3600 non-null float64
    50                       3600 non-null float64
    63                       3600 non-null float64
    80                       3600 non-null float64
    100                      3600 non-null float64
    125                      3600 non-null float64
    160                      3600 non-null float64
    200                      3600 non-null float64
    250                      3600 non-null float64
    315                      3600 non-null float64
    400                      3600 non-null float64
    500                      3600 non-null float64
    630                      3600 non-null float64
    800                      3600 non-null float64
    1000                     3600 non-null float64
    1250                     3600 non-null float64
    1600                     3600 non-null float64
    2000                     3600 non-null float64
    2500                     3600 non-null float64
    3150                     3600 non-null float64
    4000                     3600 non-null float64
    5000                     3600 non-null float64
    6300                     3600 non-null float64
    8000                     3600 non-null float64
    10000                    3600 non-null float64
    12500                    3600 non-null float64
    16000                    3600 non-null float64
    20000                    3600 non-null float64
    dbA                      3600 non-null float64
    dbC                      3600 non-null float64
    dbF                      3600 non-null float64
    Voltage                  3600 non-null float64
    WindSpeed                0 non-null float64
    WindDir                  0 non-null float64
    TempIns                  3600 non-null float64
    TempOut                  3600 non-null float64
    Humidity                 3600 non-null float64
    INVID                    0 non-null float64
    INSID                    0 non-null float64
    GChar1                   3600 non-null object
    GChar2                   0 non-null float64
    GChar3                   3600 non-null object
    AdjustmentsApplied       0 non-null float64
    CalibrationAdjustment    0 non-null float64
    GPSTimeAdjustment        0 non-null float64
    GainAdjustment           3600 non-null int64
    Status                   3600 non-null int64
    dtypes: float64(48), int64(2), object(3)
    memory usage: 1.5+ MB
    """

    endpointName = "nvspl"

    def parse(self, nvsplFileEntry, state= (None, None, 1)):
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
                df[presentNumericCols].astype('float32', copy= False, errors= 'ignore')

        except KeyError:
            pass

        return df

    def prepareState(self, endpoint, endpointParams, timestamps= None, columns= None):

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
    The ``nvsplDate``, ``hr``, and ``secs`` columns are combined into a single DatetimeIndex for the DataFrame and dropped.
    The ``len`` column (length of the noise event) is converted to a pandas Timedelta.

    Example Resulting DataFrame
    ---------------------------

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 971 entries, 2015-05-12 05:40:03 to 2015-07-14 23:55:11
    Data columns (total 10 columns):
    len         971 non-null timedelta64[ns]
    srcID       971 non-null float64
    Hz_L        971 non-null float64
    Hz_U        971 non-null int64
    MaxSPL      971 non-null float64
    SEL         971 non-null float64
    MaxSPLt     971 non-null float64
    SELt        971 non-null float64
    userName    971 non-null object
    tagDate     971 non-null datetime64[ns]
    dtypes: datetime64[ns](1), float64(6), int64(1), object(1), timedelta64[ns](1)
    memory usage: 83.4+ KB
    """

    endpointName = "srcid"

    def parse(self, entry):

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
                converted = pd.to_numeric(data[['MaxSPLt', 'SELt']], errors= "coerce")
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
    * The items axis (axis 0) is ["above", "all", "percent"]. So you'd use ``events["above"]`` to get a
      sub-DataFrame of events that exceeded $L_{nat}$, where rows are indexed by date, and columns from 0 to 23 hours.
    * The major_axis (axis 1) is ``date``: pandas DateTime objects
    * The minor_axis (axis 2) is ``hour``: 0 to 23

    Example Resulting Panel
    -----------------------

    <class 'pandas.core.panel.Panel'>
    Dimensions: 3 (items) x 63 (major_axis) x 24 (minor_axis)
    Items axis: above to percent
    Major_axis axis: 2015-05-12 00:00:00 to 2015-07-14 00:00:00
    Minor_axis axis: 0 to 23

    """

    endpointName = "loudevents"

    def parse(self, entry):

        data = pd.read_csv(str(entry),
                            engine= "c",
                            sep= "\t",
                            index_col= 0,
                            parse_dates= True,
                            infer_datetime_format= True)

        # if data.index.name is not None: data.index.name = data.index.name.lower()
        data.columns = list(range(24)) * 3
        arr_2d = xr.DataArray(data, dims=["date", "hour"])
        return xr.concat([arr_2d[:, 0:24],
                         arr_2d[:, 24:48],
                         arr_2d[:, 48:72]],
                        dim=pd.Index(["above", "all", "percent"], name="type"))

        # TODO(davyd): should this be a Dataset instead? (like this)
        # return xr.Dataset({
        #     "above": arr_2d[:, 0:24],
        #     "all": arr_2d[:, 24:48],
        #     "percent": arr_2d[:, 48:72]
        # })

class Audibility(Accessor):
    """
    Resulting DataFrame
    -------------------

    The ``time`` column is dropped, and instead combined with the date specified in the header to give
    the DataFrame a DatetimeIndex. If present, the ``listener`` field in the header is added as a column.

    <TODO>
    """
    endpointName = "audibility"

    def parse(self, entry):
        with open(str(entry), "rb") as f:
            header = None
            metadata = {}
            while header is None:
                line = str(f.readline(), encoding= "utf-8")
                keyVal = line.split(": ")
                if len(keyVal) == 2:
                    metadata[ keyVal[0][1:].lower() ] = keyVal[1].strip()
                else:
                    header = line[1:].lower().split()

            df = pd.read_csv(f,
                             engine= "c",
                             sep= "\t",
                             header= None,
                             comment= "#")
            df.columns = header

        # Combine date and time to one datetime index
        day = pd.to_datetime( metadata["date"] )
        df.index = day + pd.to_timedelta(df.time)
        df.index.name = "date"
        df.drop("time", axis= 1)

        # Keep terminology consistent with other types of files
        df.rename(columns= {"tagdate": "tagDate"}, inplace= True)
        try:
            df.tagDate = pd.to_datetime(df.tagDate)
        except AttributeError:
            pass

        if 'listener' in metadata:
            df["listener"] = metadata["listener"]

        # TODO: should this be included, or just count on getting it from mapper?
        # if 'site' in metadata:
        #     df.insert(0, "site", metadata["site"])

        return df

class DailyPA(Accessor):
    """
    The result will be indexed on two levels: `date` and `srcid`.
    This allows for interesting sub-indexing, such as::

        >> data.loc["2013-06-29", :]
        -> just srcid rows for 6/29/13, and all columns

        >> data.loc[(slice(None), "Total_All"), :]
        -> all dates, but just "Total_All" srcid rows, and all columns

        >> df.loc[(slice(None), "Total_All"), "00-23h"]
        -> all dates, just "Total_All" srcid rows, and only the 00-23h column.
           (Basically, a Series of total percent time audible per day.)

        >> df.loc[(slice(None), slice("1.1", "1.3")), "00h":"23h"]
        -> all dates, but just srcid rows between 1.1 and 1.3, and only columns from 00h to 23h.

    For more, read the :ref:`pandas docs for heirarchical indexing <pandas:advanced.hierarchical>`.

    Example Resulting MultiIndexed DataFrame
    ----------------------------------------

    <class 'pandas.core.frame.DataFrame'>
    MultiIndex: 272 entries, (2015-05-12, 1.1) to (2015-07-14, Total_All)
    Data columns (total 39 columns):
    00h                272 non-null float64
    01h                272 non-null float64
    02h                272 non-null float64
    03h                272 non-null float64
    04h                272 non-null float64
    05h                272 non-null float64
    06h                272 non-null float64
    07h                272 non-null float64
    08h                272 non-null float64
    09h                272 non-null float64
    10h                272 non-null float64
    11h                272 non-null float64
    12h                272 non-null float64
    13h                272 non-null float64
    14h                272 non-null float64
    15h                272 non-null float64
    16h                272 non-null float64
    17h                272 non-null float64
    18h                272 non-null float64
    19h                272 non-null float64
    20h                272 non-null float64
    21h                272 non-null float64
    22h                272 non-null float64
    23h                272 non-null float64
    07-18h             272 non-null float64
    19h-06h            272 non-null float64
    08-15h             272 non-null float64
    16-07h             272 non-null float64
    00-23h             272 non-null float64
    nEvents_07-18h     272 non-null int64
    nEvents_19-06h     272 non-null int64
    nEvents_08-15h     272 non-null int64
    nEvents_16-07h     272 non-null int64
    nEvents_24Hr       272 non-null int64
    eLenMean_07-18h    272 non-null int64
    eLenMean_19-06h    272 non-null int64
    eLenMean_08-15h    272 non-null int64
    eLenMean_16-07h    272 non-null int64
    eLenMean_24Hr      272 non-null int64
    dtypes: float64(29), int64(10)
    memory usage: 84.0+ KB
    """
    endpointName = "dailypa"

    def parse(self, entry):
        data = pd.read_csv(str(entry),
                           engine= "c",
                           sep= "\t",
                           parse_dates= False,
                           index_col= [0, 1])

        data.index.names = ["date", "srcid"]

        # Check for AMT bug that adds row of ('nvsplDate', 'Total_All') with all 0s, drop if exists
        if data.index[-1][0] == 'nvsplDate':
            data = data.iloc[:-1, :]

        ## Pandas cannot seem to handle a MultiIndex with dates;
        ## slicing syntax becomes even crazier, and often doesn't even work.
        ## So date conversion is disabled for now.

        # # Convert dates
        # datetimes = data.index.get_level_values('date').to_datetime()
        # data.index.set_levels(datetimes, level= 'date', inplace= True)

        # Ensure MultiIndex sortedness
        data.sort_index(inplace= True)

        return data.apply(pd.to_numeric, raw= True, errors= "coerce")

class Metrics(Accessor):
    """
    Read all tables from a metrics file.

    Resulting data is an object (a named tuple, really) with attributes for each metric in the file, as well as
    ``metadata``, which is a dict of the colon-seperated key-value pairs in the file's header.
    Missing metrics are stored as None. (SPLAT-related metrics such as ``noiseFreeInterval`` are often missing.)

    Otherwise, each attribute for a table has two attributes itself: ``data`` and ``n``. ``data`` contains a
    pandas Panel of that table's data. ``n`` contains a DataFrame or Series of TimeDeltas of the
    lengths of the dataset, by season and table type.

    In other words, the retured object is structured::

        metrics
            metadata: {"Day": "07:00:00 to 18:59:59", "Source of Interest": "Aircraft", ...}
            hourlyMedian
                data: Panel
                n: DataFrame
            frequency
                data: Panel
                n: DataFrame
            ambient
                data: Panel
                n: DataFrame
            ...
            ...


    A primary purpose of this reader is to combine multiple tables of related data in the metrics file
    into single structures. (For example, Median Hourly Metrics could have four tables, for dBA and dBT
    in both Summer and Winter. These are combined into a single Panel, making it easy to perform complex
    selections across the tables---e.g. dBA in both seasons.)

    Here's how these ``data`` Panels are indexed:

        + For metrics composed of multiple tables:
            - Labels axis: Season        ("Winter", "Summer", ...)
            - Items axis:  Table type    ("dBA" and "dBT"; "night" and "day"; "l90", "lnat", and "l50"; ...)
            - Major axis:  Table columns ("12.5Hz" to "20000Hz"; 0 to 23; "Lmin", "L099", "Lnat", ...)
            - Minor axis:  Table rows    ("L090", "Lnat", "L050"; "Day", "Night", "overall"; 0 to 23; 1.1, 1.2, 1.3, ...)
            - **So these are accessed** ``data.loc[ <season>, <tableType>, <columns>, <rows> ]``
        + For metrics composed of just one table:
            - Items axis: Season        ("Winter", "Summer", ...)
            - Major axis: Table columns ("12.5Hz" to "20000Hz"; 0 to 23; "Lmin", "L099", "Lnat", ...)
            - Minor axis: Table rows    ("L090", "Lnat", "L050"; "Day", "Night", "overall"; 0 to 23; 1.1, 1.2, 1.3, ...)
            - **So these are accessed** ``data.loc[ <season>, <columns>, <rows> ]``

    And the ``n`` DataFrames or Series are indexed:

        + For metrics composed of multiple tables (DataFrame):
            - Columns: Season     ("Winter", "Summer", ...)
            - Rows:    Table type ("dBA" and "dBT"; "night" and "day"; "l90", "lnat", and "l50"; ...)
            - **So these are accessed** ``n.loc[ <season>, <tableType> ]``
        + For metrics composed of just one table (Series):
            - Rows: Season     ("Winter", "Summer", ...)
            - **So these are accessed** ``n[ <season> ]``

    Examples (where the object returned from this function is stored as ``metrics``)::

        >> metrics.noiseFreeInterval.data
        -> a Panel of the SPLAT Noise Free Interval (sec) table for each season, indexed by [season, percentile, hour]

        >> metrics.noiseFreeInterval.n
        -> a Series of the number of days used to compute the noise free interval metric, with one row per season

        >> metrics.hourlyMedian.data
        -> a Panel4D of the Median Hourly Metrics tables for dBA and dBT for each season, indexed by [season, spl weighting, percentile, hour]

        >> metrics.hourlyMedian.data.Summer.dBA
        >> metrics.hourlyMedian.data.loc["Summer", "dBA"]
        -> a DataFrame subselecting the dBA item from the Summer label in the Median Hourly Metrics panel
        -> (essentially, just the original ``Median Hourly Metrics (dBA), Summer`` table in the metrics file)

        >> metrics.hourlyMedian.data.loc["Winter", :, "Leq", 0:12]
        -> a DataFrame of median Leq values from the hours 0-12 (rows) for both dBA and dBT (columns) in Winter

        >> metrics.hourlyMedian.data.loc[:, :, "Leq", 0:12].mean(axis= "items")
        -> a DataFrame of the mean across all seasons of median Leq values from the hours 0-12 (columns) for both dBA and dBT (rows)

    Special Cases:

        + Hour axes have the ``'h'`` removed, so they are just integers 0-23.
        + The ``frequency``, ``ambient``, and ``percentTimeAbove`` metrics have the additional row ``"overall"``
          added along with ``"Day"`` and ``"Night"``. This is the logarithmic mean SPL for both day and night.

    Raises
    ------
    TypeError
        If the version of the file does not match the reader
    ValueError
        If the header cannot be parsed
    OSError
        Unhandled---raised if the file is missing
    """
    endpointName = "metrics"

    def parse(self, entry):
        with open(str(entry)) as f:
            versionLine = f.readline()

        version = self.MetricsReader.parseVersionLine(versionLine)
        try:
            return self.metricsReaders[version](entry)
        except KeyError:
            raise TypeError("No metrics reader for version {}".format(version))

    def __init__(self, *args, **kwargs):
        self.metricsVersions = {
            "1.35": {
                "hourlyMedian"              : {'dBA': "Median Hourly Metrics (dBA)", 'dBT': "Median Hourly Metrics (dBT)"},
                "frequency"                 : {'night': "Median Nighttime Frequency Metrics (dB)", 'day': "Median Daytime Frequency Metrics (dB)"},
                "ambient"                   : {'dBA': "Ambient (dBA)", 'dBT': "Ambient (dBT)"},
                "percentTimeAbove"          : {'dBA': "Time Above dBA (%)", 'dBT': "Time Above dBT (%)"},  # the dBA and dBT are added during parsing as special case
                "contour"                   : {'l90': "L90 Contour Data (dB)", 'lnat': "Lnat Contour Data (dB)", 'l50': "L50 Contour Data (dB)", 'l05': "L05 Contour Data (dB)"},
                "eventAudibilityPct"        : "SPLAT Detailed Event Audibility (%)",
                "eventAvg"                  : {'counts': "SPLAT Detailed Average Event Counts", 'lengths': "SPLAT Detailed Average Event Lengths (sec)"},
                "categoricalEventAudibility": "SPLAT Categorical Event Audibility (%)",
                "noiseFreeInterval"         : "SPLAT Noise Free Interval (sec)",
                "percentTimeAudible"        : "Time Audible (%)"
            }
        }

        self.metricsReaders = { version: self.MetricsReader(version, metricNames) for version, metricNames in iteritems(self.metricsVersions) }
        super(Metrics, self).__init__(*args, **kwargs)

    class MetricsReader(object):
        """
        A reader for metrics files, which is instantiated for a specific file version.

        In metrics files, there are often multiple related tables, just with different units
        and different seasons. The metrics reader handles combining these related tables into
        multidimensional structures, so all the data can be easily sliced and diced across and
        within tables.

        Read a metrics file by calling an instance of this class with the path to the file.
        Returns a named tuple of the metrics tables.
        """
        def __init__(self, readerVersion, metricNames):
            """
            Create a reader for metrics files of a specific version.

            readerVersion : str
            metricNames   : dict of {str: str} or str

            First, some terms:

            `table` : a tab-seperated table of data in the raw metrics file.
                      Specified in the first line of every table are three parameters:
                      - `title` : ex. "Median Hourly Metrics (dBA)", "SPLAT Detailed Average Event Counts", "L90 Contour Data (dB)"
                      - `season`: ex. "Summer", "Winter"
                      - `n`     : ex. "n = 32 days", "n = 467hrs", "n = 16"

            `metric`: a specific kind of data. Metrics can be composed from multiple tables, using a multidimensional Panel data structure
                     (i.e. the "ambient" metric might be composed from the "Ambient (dBA)" and "Ambient (dBT)" tables
                      from both the seasons "Summer" and "Winter")

            `metricName`: the abbreviated, canonicalized name for a metric. Table titles may change across versions,
                          but metric names should always be the same. Metrics are accessed by these names as attributes of
                          the returned named tuple

            `tableType` : the distinguishing factor between tables of the same metric.
                          ex. "dBA", "dBT", "night", "day", "l90", "lnat", "l50", ...
                          `tableType` becomes the items axis in a metrics Panel:
                          the index order goes [season, tableType, <percentile, hour, srcid, etc>, <percentile, hour, srcid, etc>]

            So, `metricNames` should be a mapping from `{ metricName : { tableType: tableName, tableType: tableName... }, ... }`.
            Essentially, it specifies which tables are used to form a metric.

            For example:
                - `{ "frequency" : {'night': "Median Nighttime Frequency Metrics (dB)", 'day': "Median Daytime Frequency Metrics (dB)"} }`
                means the `frequency` metric is composed from the table "Median Nighttime Frequency Metrics (dB)", with type `night`,
                and the table "Median Daytime Frequency Metrics (dB)", with type `day`.
                - `{"noiseFreeInterval" : "SPLAT Noise Free Interval (sec)"}` means the `noiseFreeInterval` metric is composed of
                just the one table "SPLAT Noise Free Interval (sec)".

            Special case: there are two different tables titled "Time Above (%)": one for dBA, one for dBT.
            When reading a file, this is handled by internally checking the units and changing the titles to
            "Time Above dBA (%)" and "Time Above dBT (%)", so those two names should be used in the metric names dict.
            """
            self.readerVersion = readerVersion
            self.metricNames = metricNames

            # map of { table title: (metricName, tableType) }
            # used for looking up which metric a table goes to (and how it's used in that metric)
            self.titlesToMetricNamesAndTypes = { title: (metricName, tableType) for metricName, typesAndTitles in iteritems(metricNames) for tableType, title in (iteritems(typesAndTitles) if isinstance(typesAndTitles, dict) else [(None, typesAndTitles)]) }

            self.Metrics = collections.namedtuple('Metrics', list(metricNames.keys()) + ["metadata"] )
            self.Metric = collections.namedtuple('Metric', ["data", "n"])

            try:
                self.Metrics.__doc__ += '\n\nEach metric is stored as a named tupe, with attributes `data` and `n`.'
                self.Metrics.__doc__ += '\nFor example, if this named tuple is named `metrics`:'
                self.Metrics.__doc__ += '\n    `metrics.noiseFreeInterval.data` would return a pandas DataFrame'
                self.Metrics.__doc__ += '\n    `metrics.noiseFreeInterval.n`    would return a pandas Series of TimeDeltas of the length of the dataset for each season'
                self.Metrics.__doc__ += '\n\nMetadata is available as a dict in `metrics.metadata`.'

                self.Metric.__doc__   += '\n\n`data`: a pandas Panel, indexed either by [season, tableType, <column>, <row>] or [season, <column>, <row>]'
                self.Metric.__doc__   += '\n`n`     : a pandas DataFrame or Series of the length of this dataset (as TimeDeltas). The primary key is season, secondary key (if applicable) is tableType'
            except AttributeError:
                # TODO: docstrings are not writable in py2. This could likely be circumvented using a metaclass,
                # that's a problem for another time, and we'll just leave py2 users in the dark.
                pass
            # TODO: repr() for Metric and Metrics

        @staticmethod
        def parseVersionLine(versionLine):
            """
            Parse the version as a float from the first line in a metrics file.

            The line should be formatted like "### Metrics File V1.35". Raises a ValueError if line cannot be parsed.
            """
            match = re.match(r"^### Metrics File V(.+)$", versionLine)
            if match:
                version = match.group(1)
                return version
            else:
                raise ValueError('Unrecognized metrics file version: "{}"'.format(versionLine))

        @staticmethod
        def splMean(*spls):
            return 10 * np.log10( np.sum(10**(spl/10) for spl in spls) / len(spls) )

        def __call__(self, entry):
            with open(str(entry)) as f:
                txt = f.read()

            sections = txt.split("\n\n")[:-1] # file is terminated by double-linebreak, so we don't need the final empty section
            header, tableText = sections[0], sections[1:]

            ## Parse metrics file version and check for compatability
            try:
                headerSplit = header.split("\n")
                versionLine, titleLine, headerLines = headerSplit[0], headerSplit[1], headerSplit[2:]
            except ValueError:
                raise ValueError("Unrecognized header in metrics file")

            version = self.parseVersionLine(versionLine)
            if version != self.readerVersion:
                raise TypeError('Metrics file "{}" is version "{}", expected version "{}"'.format(path, version, self.readerVersion))

            ## Parse header into metadata dict
            header = {}
            for headerLine in headerLines:
                k, v = headerLine.split(": ")
                header[k.lower()] = v.strip()

            ## Parse each table into a DataFrame
            metrics = collections.defaultdict( lambda: collections.defaultdict(dict) ) # map of { metricName: {season: {tableType: DataFrame}} }
            ns = collections.defaultdict( lambda: collections.defaultdict(dict) )      # map of { metricName: {season: {tableType: n}} }

            titleRe = re.compile(r"(.*),\s?(.*?)\s?\((.*)\)")  # match metric name, season, n
            nRe     = re.compile(r"n = (\d+) ?(.*)")           # match (optional) unit and length from an n

            for table in tableText:
                tableLines = table.split("\n")
                titleLine, lines = tableLines[0], tableLines[1:]
                cells = [ line.split("\t") for line in lines ]
                columns, body = cells[0], cells[1:]

                # Split title parts, and look up the canonical table name for this metric name
                match = titleRe.match(titleLine)
                try:
                    title, season, n = match.groups()
                    if title == "Time Above (%)":
                        # Infuriatingly, there are two tables named "Time Above (%)": one is dBA and one is dBT
                        # In the metricNames mapping, we pretend these tables are actually titled "Time Above dBA (%)"
                        # and "Time Above dBT (%)", so we'll rename the table before looking it up, by checking the units
                        # in the header row.
                        if columns[1][-1] == "A":
                            title = "Time Above dBA (%)"
                        elif columns[1][-1] == "T":
                            title = "Time Above dBT (%)"
                        else:
                            raise ValueError("Time Above (%) table with unexpected units: {}".format(columns))
                    metricName, tableType = self.titlesToMetricNamesAndTypes[title]
                except ValueError:
                    warnings.warn("Unparseable title: {} (in {})".format(title, path))
                    continue
                except KeyError:
                    warnings.warn("Unknown metric {} (in {})".format(metric, path))
                    continue

                # Get n-value from title
                match = nRe.match(n)
                try:
                    amt, unit = match.groups()
                    if unit == "hrs" or unit == "": n = pd.to_timedelta(amt, unit= "h")
                    elif unit == "days": n = pd.to_timedelta(amt, unit= "d")
                    else: n = pd.to_timedelta(amt, unit= unit)
                except (ValueError, AttributeError):
                    n = pd.to_timedelta('NaT')
                ns[metricName][season][tableType] = n

                # Create DataFrame
                df = pd.DataFrame(body)

                df.set_index(0, inplace= True)
                df.index.name= None
                df.columns = columns[1:]
                df = df.apply(pd.to_numeric, raw= True, errors= "coerce")

                # Guess the type of the index (if it's noise level, or hour)
                for axname in ("index", "columns"):
                    axis = getattr(df, axname)
                    if axis.str.startswith("L").all():
                        #L_x levels
                        axis.name = "percentile"
                    elif axis.str.endswith("h").all():
                        # Hours
                        axis = axis.str.rstrip("h").astype("int")
                        axis.name = "hour"
                        setattr(df, axname, axis)

                # Ensure percentTimeAbove has the same columns names in both tables: just dB instead of dBA and dBT
                if metricName == "percentTimeAbove":
                    df.columns = df.columns.str.slice(stop= -1)

                metrics[metricName][season][tableType] = xr.DataArray(df)

            ## Prepare a dict of DataArrays to turn into an xarray Dataset.
            # We concatenate each 2D table together, then concatenate those per-season 3D tables
            # to end up with the 4D table for each metric.
            # This is a rather dumb and lazy way of doing it, but it works for now.
            xr_metrics = {}
            for metricName, metric_dict in iteritems(metrics):
                seasons = []
                table_arrs = []
                for season, season_dict in iteritems(metric_dict):
                    tables, arrs = zip(*list(iteritems(season_dict)))
                    # ^ get a list of table names, corresponding to a list of 2D DataArrays for those tables
                    if len(tables) == 1 and tables[0] is None:
                        # Metrics derived from a single table will have seasons as labels, and a superfluous table name of [None].
                        # In that case, just don't create a `Table` dimension
                        table_arr = arrs[0]
                    else:
                        table_arr = xr.concat(arrs, dim=pd.Index(tables, name="Table"))
                    table_arrs.append(table_arr)
                    seasons.append(season)

                arr = xr.concat(table_arrs, dim=pd.Index(seasons, name="Season"))
                xr_metrics[metricName] = arr

            ## Create DataFrame/Series of n values for each table in metric
            # TODO(davyd): figure out how to do this as coordinates on the dataset, not just attrs
            ns = { metricName: pd.DataFrame(nVals) for metricName, nVals in iteritems(ns) }
            # xr_ns = {}
            for metricName, n in iteritems(ns):
                # Ns derived from a single table will have a superfluous row of NaN
                # Reduce them to just a Series, with season as the index
                n.columns.name = "Season"
                n.index.name = "Table"
                if all( n.index == [None] ):
                    ns[metricName] = n.iloc[0]
                # arr = xr.DataArray(n)
                # print(arr.dtype)
                # print(arr)
                # xr_ns[metricName + "_n"] = arr

            ds = xr.Dataset(xr_metrics, attrs=ns)
            return ds

            # TODO(davyd): implement overall daily levels on the xarray dataset
            # (or perhaps do it on the `xr_metrics` dict first, before turning it into a Dataset---whichever is easier)

            ## Combine day and night levels
            try:
                freq = metrics["frequency"]
                for season in freq.labels:
                    freq.loc[season, "overall"] = self.splMean(freq.loc[season, "day"], freq.loc[season, "night"])
            except KeyError:
                pass

            try:
                ambient = metrics["ambient"].transpose('labels', 'major_axis', 'items', 'minor_axis')
                for season in ambient.labels:
                    ambient.loc[season, "overall"] = self.splMean(ambient.loc[season, "Day"], ambient.loc[season, "Night"])
                metrics["ambient"] = ambient.transpose('labels', 'major_axis', 'items', 'minor_axis')
            except KeyError:
                pass

            try:
                pta = metrics["percentTimeAbove"].transpose('labels', 'major_axis', 'items', 'minor_axis')
                for season in pta.labels:
                    pta.loc[season, "overall"] = (pta.loc[season, "Day"] + pta.loc[season, "Night"]) / 2
                metrics["percentTimeAbove"] = pta.transpose('labels', 'major_axis', 'items', 'minor_axis')
            except KeyError:
                pass

            ## Bundle into a Metrics named-tuple and return
            return self.Metrics(metadata= header, **{ metricName: self.Metric(metrics[metricName], ns[metricName]) if metricName in metrics else None for metricName in self.metricNames.keys() })
