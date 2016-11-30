# Python 2 and 3 cross-compatibility:
from __future__ import print_function, division, unicode_literals, absolute_import
from builtins import (bytes, str, int, dict, object, range, map, filter, zip, round, pow, open)
from future.utils import (iteritems, itervalues, with_metaclass)
from past.builtins import basestring

import itertools
import functools
import operator
import collections
import traceback
import inspect
import warnings
import sys

import numpy as np
import pandas as pd
import re
import iyore
from tqdm import tqdm, tqdm_notebook

class AccessorMetaclass(type):
    """
    Metaclass to insert boilerplate documentation into each Accessor subclass,
    pull out keyword arguments for prepareState, and generally ensure Accessors are implemented correctly.

    Each Accessor subclass's docstring should only have information specific to that kind of data.
    These sections should be:

    <Accessor_name>-Specific Parameters (optional)
    -----------------------------------

    If the Accessor has a prepareState method that takes extra keyword arguments
    (like NVSPL takes the ``columns`` and ``timestamps`` arguments), document them here.
    The first line of the docstring---the argument spec for calling the Accessor---will
    have these extra keyword arguments and their default values nicely inserted automatically. Neat!

    Resulting Series/DataFrame/Panel/Object
    ---------------------------------------

    Document the structure of the data this Accessor returns. Typically, this can just be the
    output of ``data.info()`` if ``data`` is a pandas structure.

    """

    def __new__(mcls, clsname, bases, dct):

        # We only want to modify the docstrings for *subclasses* of Accessor, not Accessor itself.
        # For now, I hardcode the name of the base class whose subclasses should have their documentation patched.
        # That's pretty nasty, I know.
        # There is probably a more general way to figure out the hierarchy of the class-to-be,
        # and whether it's a subclass of Accessor or the Accessor class itself. However, with the class-to-be
        # not yet instantiated, there's not much that Python's internals can do to help us.
        # For now, we're leaving it at this
        if clsname != "Accessor":

            if dct.get("endpointName", None) is None:
                raise NotImplementedError('No Endpoint name set for the Accessor "{}"'.format(clsname))

            # Format any special keyword arguments the Accessor's prepareState function has
            prepareStateArgspec = ""
            prepareStateKwargs = []
            if "prepareState" in dct:
                prepareArgspec = inspect.getargspec(dct["prepareState"])
                if prepareArgspec.defaults is not None:
                    # ensure prepareState's kwargs are not the same as any for Accessor.__init__
                    initArgspec = inspect.getargspec(Accessor.__init__)
                    initKwargs = initArgspec.args[ -len(initArgspec.defaults): ]
                    prepareStateKwargs = prepareArgspec.args[ -len(prepareArgspec.defaults): ]
                    for reserved in initKwargs:
                        if reserved in prepareStateKwargs:
                            raise TypeError("The keyword argument '{}' is already used by Accessor, please pick a different name in {}".format(reserved, clsname))

                    prepareStateArgspec = " "+inspect.formatargspec(prepareStateKwargs, None, None, prepareArgspec.defaults)[1:-1] + ","

            # Fill in the subclassDocTemplate with information from the subclass
            fillin = {
                "endpointName": dct.get("endpointName", clsname),
                "subclassDocstring": inspect.cleandoc(dct.get("__doc__", "")),
                "className": clsname,
                "prepareStateArgspec": prepareStateArgspec
            }

            newdoc = inspect.cleandoc(mcls.subclassDocTemplate).format(**fillin)
            dct["__doc__"] = newdoc
            dct["_prepareStateKwargs"] = prepareStateKwargs

        return super(AccessorMetaclass, mcls).__new__(mcls, clsname, bases, dct)

    subclassDocTemplate = """
        {endpointName}(ds: iyore.Dataset, n=None, items=None, sort=None,{prepareStateArgspec} **filters)

        Access {className} data from the dataset ``ds`` that matches the given filters, and apply operations to it.

        {subclassDocstring}

        Documentation common to all Accessors
        =====================================

        A quick-reference for how Accessors for all types of files are used. 

        Parameters
        ----------

        ds : iyore.Dataset
            
            The Dataset from which to access the NVSPL files

        n : int, default None

            Maximum number of Entries to read

        items : iterable of dict, or pandas.DataFrame, default None

            Specific entries to access from the Dataset

        sort : str, iterable of str, or function, default None

            How to sort the results. If ``str`` or iterable of ``str``, must be field(s) of the NVSPL Endpoint of the given Dataset.
            If function, must take an ``iyore.Entry`` and return a value to represent that Entry when sorting.

        **filters : str, number, dict of {{str: False}}, iterable of str, or function

            Restrict results to Entries which match the given values in the specified fields

        Returns
        -------
        For more, see the documentation for ``soundDB.Query``.

        ``soundDB.Query`` object, which can be used in these ways:

            - ``.groupby()``: bundle the data into groups by year, site, etc., apply operations to summarize each group,
                              combine the results computed from all the groups into a DataFrame (like pandas groupby).   

            - ``.all()``: return a DataFrame of all the data contained in the Dataset.

            - ``.one()``: return one Entry object whose ``data`` attribute contains a pandas structure

            - ``.sorted()``: iterate in a particular order through Entry objects whose ``data`` attributes contain pandas structures.
            
            - As an iterable, which yields Entry objects whose ``data`` attributes contain pandas structures.
    """



class Accessor(with_metaclass(AccessorMetaclass, object)):
    """
    Abstract base class to create classes for accessing a specific kind of data from an iyore Dataset.

    By subclassing Accessor and implementing a ``parse`` function, which holds the logic for
    reading a single file of a specific type into a pandas structure, Accessor provides the
    logic to vectorize that function so it can parse all instances of that data file from an Endpoint.
    """

    # Overridden in each subclass: name of the ``iyore.Endpoint`` where the kind of data
    # the Accessor handles is found
    endpointName = None

    def parse(self, entry, state= None):
        """
        Parse a single file of a specific type into a pandas structure.

        This method should be overridden in each subclass to actually read the type of file it handles.

        Parameters
        ----------
        entry : iyore.Entry
            An entry for a single file.
            If possible, parsers should not access any attributes from the Entry, and get its path using
            ``str(entry)`` rather than ``entry.path``, so that the function can also be used with just
            a string of the path to a file to read.
        state : client-determined keyword argument, optional
            An optional object, created in the initialization function,
            which is passed into the parser function on every call for a Query.
            (Often a dict or tuple.) This can be used to pass state between files read.
            Though the object is created in ``prepareState``, it's safe for ``parserFunc``
            to mutate.
        """
        raise NotImplementedError

    def prepareState(self, endpoint, endpointParams, **kwargs):
        """
        Optionally overridden in subclasses which need to pass state between multiple calls of
        ``parse()`` for the same Query

        Should return the ``state`` object which will be passed into ``parse`` on every call for a Query.
        The state object can be any type (though dict usually makes most sense).
        ``prepareState`` is called once, before each time the Query is used.

        When the Query is used (by .all(), .groupby(), etc), any keyword arguments it's given
        that match the keyword arguments taken by ``prepareState`` will be given to ``prepareState``.
        Any remaining keyword arguments that do not share the same names are assumed to be filters for the Endpoint.
        The Endpoint instance, and a dict of these endpoint parameters, are given to
        ``prepareState`` as its first two arguments, in case they're useful.
        """
        return None

    def __init__(self, ds, n= None, items= None, sort= None, **filters):


        try:
            endpoint = getattr(ds, self.endpointName)
        except AttributeError:
            raise ValueError('No endpoint "{}" exists in the given dataset'.format(self.endpointName))

        if items is not None:
            # TODO: selection based on parsing ID strings from DataFrame index, not just columns
            if isinstance(items, pd.DataFrame):
                paramColumns = items.columns.intersection(endpoint.fields)
                # TODO: may need to map a function which converts numerics to strings
                items = paramColumns.iterrows()

            filters["items"] = items

        # split prepareState params from endpoint filters
        # note: self._prepareStateKwargs was precomputed by the AccessorMetaclass
        self._prepareStateParams = { kwarg: filters.pop(kwarg) for kwarg in self._prepareStateKwargs if kwarg in filters }

        self._endpoint = endpoint
        self._filters = filters
        self._sort = sort
        self._chain = []
        self._n = n

        # if n is not None:
        #     self._chain.append(lambda iterable: itertools.islice(iterable, n))


    @classmethod
    def ID(cls, key):
        if isinstance(key, iyore.Entry):
            id_elems = []
            if "unit" in key.fields: id_elems.append(key.unit)
            if "site" in key.fields: id_elems.append(key.site)
            if "year" in key.fields: id_elems.append(key.year)
            if "month" in key.fields:
                if len(id_elems) > 0: id_elems.append(' ')
                id_elems.append(key.month)
                month = True
            if "day" in key.fields:
                if month: id_elems.append('-')
                id_elems.append(key.day)
            if "hour" in key.fields:
                if len(id_elems) > 0: id_elems.append(' ')
                id_elems.append(key.hour+':')
            if len(id_elems) > 0:
                return "".join(id_elems)
            else:
                return key.path
        else:
            return key

    def combine(self, func= lambda x: x, into= None, ID= None, *args, **kwargs):
        # TODO: deprecate processing function in favor of .pipe on pandas objects?

        if ID is None:
            ID = self.ID

        results = collections.defaultdict(list)
        # build map of {ID: [data, data, ...]} (same ID may have multiple data, i.e. NVSPL or LA)
        for key, data in iter(self):
            results[ID(key)].append(data)

        # flatten data for each ID by concatenating, or unpacking list if just one dataframe,
        # then apply processing function to (maybe-)concatenated data
        for ID_name, datas in iteritems(results):
            try:
                flat = pd.concat(datas, copy= False) if len(datas) > 1 else datas[0]
            except TypeError:
                # Issues arise when more than one file maps to the same ID (i.e. NVSPL and LA when *not* using .group),
                # AND the ops chain results in a scalar (or non-pandas type) for each file.

                # pd.concat obviously doesn't work on scalars. But just turning the list of datas
                # into a Series isn't that easy: what would you use for the index?
                # Currently, the solution is a more comprehensive ID function,
                # so that any data will almost certainly get a different ID for each file.
                # Eventually, a better solution than just using the list might (or might not) be good,
                # but it's unclear what that would be.
                warnings.warn("Tried to concatenate non-pandas data. Please raise an issue if you find a use-case which triggers this.")
                flat = datas
            try:
                # apply processing function
                results[ID_name] = func(flat, *args, **kwargs)
            except:
                print('Error in final processing function while processing data for "{}":'.format(ID_name))
                print( traceback.format_exc() )


        if len(results) == 0:
            return results

        if len(results) == 1:
            key, result = results.popitem()
            return result

        ########################################################################################
        # Combine results depending on whether data are scalars, Series, DataFrames, or Panels #
        ########################################################################################
        overlapThreshold = 0.75     # fraction of columns/rows all data must have in common to be considered worth combining
        def percentIndexOverlap(attr, results):
            """
            Returns what fraction of index values are common to all NDFrames in results,
            compared to the NDFrame with the most rows/columns.

            Pass "index" or "columns" as ``attr`` to specify rows or columns.
            """
            getter = operator.attrgetter(attr)

            longest = max(len(getter(df)) for df in itervalues(results))
            columnsIter = (getter(df) for df in itervalues(results))
            mutualColumns = functools.reduce(lambda colA, colB: colA.intersection(colB), columnsIter)
            return len(mutualColumns)/longest

        # Sanity check: are all data at least of the same type?
        exampleResult = next(iter(itervalues(results)))
        if all(type(data) == type(exampleResult) for data in itervalues(results)):

            if np.isscalar(exampleResult) or isinstance(exampleResult, (pd.Timedelta, pd.Timestamp, pd.Period)):
                # combine scalars to Series
                return pd.Series(results)
            elif isinstance(exampleResult, (pd.Series, list)):
                # combine Series (or lists) to DataFrame
                # if indicies overlap, return a DataFrame, otherwise a MultiIndexed Series
                try:
                    if percentIndexOverlap("index", results) >= overlapThreshold:
                        return pd.DataFrame.from_dict(results, orient= "columns")
                    else:
                        return pd.concat(results)
                except AttributeError:
                    return results


            elif isinstance(exampleResult, pd.DataFrame):
                # if all DataFrames have the same have the same indicies and columns --> Panel
                # if just have same columns and different indicies (of same dtype, i.e. DatetimeIndex) --> MultiIndexed DataFrame (like .all())
                
                # if at least 75% of columns overlap in all results, consider them worth combining
                # (75% is a pretty arbitrary number...)
                if percentIndexOverlap("columns", results) >= overlapThreshold:

                    # if at least 75% of rows overlap, combine to a Panel
                    if percentIndexOverlap("index", results) >= overlapThreshold:
                        return pd.Panel.from_dict(results, orient= 'items')

                    # otherwise, ensure the indicies at least are all the same dtype, and make a MultiIndexed DataFrame (like .all() does)
                    elif all(df.index.dtype == exampleResult.index.dtype for df in itervalues(results)):
                        return pd.concat(results)

            elif isinstance(exampleResult, pd.Panel) and not isinstance(exampleResult, pd.Panel4D):
                # if at least 75% of all axes overlap, combine to a Panel4D
                if all(percentIndexOverlap(attr, results) >= overlapThreshold for attr in ["items", "major_axis", "minor_axis"]):
                    return pd.Panel4D.from_dict(results)

        # If types are inconsistent, or not pandas, or a Panel4D, just give back results as a dict---we can't help you any more here
        return results

    def group(self, *groups):
        if len(groups) == 0:
            raise TypeError("No groups given to groupby")
        elif len(groups) > 1:
            if all(isinstance(group, basestring) for group in groups):
                groupFunc = lambda e: tuple(getattr(e, group) for group in groups)
            else:
                raise TypeError("If multiple groups are given, all must be strings")
        else:
            group = groups[0]
            if isinstance(group, basestring):
                groupFunc = operator.attrgetter(group)
            else:
                if hasattr(group, "__call__"):
                    groupFunc = group
                else:
                    raise ValueError('Argument to groupby must be a string or function, instead got "{}"'.format(type(group)))

        def concat_maybe(datas):
            datas = tuple(datas)
            if len(datas) == 1:
                return datas[0]
            else:
                # TODO: use GroupbyApplier.compute()-like logic for concat/promote?
                try:
                    return pd.concat(datas)
                except TypeError:
                    return datas

        self._sort = groupFunc
        def do_group(iterator):
            for key, subiter in itertools.groupby(iterator, lambda entryAndData: groupFunc(entryAndData[0])):
                yield key, concat_maybe(data for entry, data in subiter)

        self._chain.append(do_group)
        return self

    def __getattr__(self, attr):
        def do_getattr(iterator):
            for entry, data in iterator:
                try:
                    yield entry, getattr(data, attr)
                except GeneratorExit:
                    raise GeneratorExit
                except:
                    self._progbar.write('Error while processing "{}":'.format(str(entry)))
                    self._progbar.write( traceback.format_exc() )
                    self._progbar.write("Data was:")
                    self._progbar.write(repr(data))

        self._chain.append(do_getattr)
        return self

    def __getitem__(self, *index):
        def do_getitem(iterator):
            for entry, data in iterator:
                try:
                    yield entry, data.__getitem__(*index)
                except GeneratorExit:
                    raise GeneratorExit
                except:
                    self._progbar.write('Error while processing "{}":'.format(str(entry)))
                    self._progbar.write( traceback.format_exc() )
                    self._progbar.write("Data was:")
                    self._progbar.write(repr(data))

        self._chain.append(do_getitem)
        return self

    def __call__(self, *args, **kwargs):
        def do_call(iterator):
            for entry, data in iterator:
                try:
                    yield (entry, data(*args, **kwargs))
                except GeneratorExit:
                    raise GeneratorExit
                except:
                    self._progbar.write('Error while processing "{}":'.format(str(entry)))
                    self._progbar.write( traceback.format_exc() )
                    self._progbar.write("Data was:")
                    self._progbar.write(repr(data))

        self._chain.append(do_call)
        return self


    def __iter__(self):
        state = self.prepareState(self._endpoint, self._filters, **self._prepareStateParams)
        entries = self._endpoint(sort= self._sort, n= self._n, **self._filters)

        try:
            get_ipython
            inNotebook = True
        except NameError:
            inNotebook = False

        if not inNotebook:
            sys.stderr.write("Locating data...")
        entries = list(entries)
        if not inNotebook:
            sys.stderr.write("\r")

        try:
            self._progbar = tqdm_notebook(entries, unit= "entries")
        except (NameError, AttributeError, TypeError):
            self._progbar = tqdm(entries, unit= "entries")


        def iterate():
            for entry in self._progbar:
                try:
                    data = self.parse(entry, state= state) if state is not None else self.parse(entry)
                    # self._progbar.write("Processing {}".format(entry.path))
                    yield entry, data
                except GeneratorExit:
                    raise GeneratorExit
                except:
                    self._progbar.write('Error while parsing "{}":'.format(entry.path))
                    self._progbar.write( traceback.format_exc() )

        # chain the operations together
        # each function in self._chain is a generator which takes an iterator
        # (remember that you call a generator to "activate" it: calling a generator returns an iterator)
        # so end condition for the loop is that `iterate` refers to an iterator
        iterate = iterate()
        for do in self._chain:
            iterate = do(iterate)
        return iterate




