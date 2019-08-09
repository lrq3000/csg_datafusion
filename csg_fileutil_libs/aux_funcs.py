#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Auxiliary functions library for data fusion from reports extractor, dicoms handling, etc
# Copyright (C) 2017-2019 Stephen Karl Larroque
# Licensed under MIT License.
# v2.9.7
#

from __future__ import absolute_import

import ast
import chardet
import copy
import numbers
import os
import re
import shutil
import unicodecsv as csv
from collections import OrderedDict
from .dateutil import parser as dateutil_parser
from .distance import distance
from .pydicom.filereader import InvalidDicomError
from . import pydicom

import pandas as pd

try:
    from scandir import walk # use the faster scandir module if available (Python >= 3.5), see https://github.com/benhoyt/scandir
except ImportError as exc:
    from os import walk # else, default to os.walk()

try:
    # to convert unicode accentuated strings to ascii
    from .unidecode import unidecode
    _unidecode = unidecode
except ImportError as exc:
    # native alternative but may remove quotes and some characters (and be slower?)
    import unicodedata
    def _unidecode(s):
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore')
    print("Notice: for reliable ascii conversion, you should pip install unidecode. Falling back to native unicodedata lib.")

try:
    from .tqdm import tqdm
    _tqdm = tqdm
except ImportError as exc:
    def _tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

try:
    from StringIO import StringIO as _StringIO
except ImportError as exc:
    from io import StringIO as _StringIO

def _str(s):
    """Convert to str only if the object is not unicode"""
    return str(s) if not isinstance(s, unicode) else s

def save_dict_as_csv(d, output_file, fields_order=None, csv_order_by=None, verbose=False):
    """Save a dict/list of dictionaries in a csv, with each key being a column
    Note: Does NOT support unicode, nor quotes. See Python doc to get UnicodeWriter for unicode support (but quotes unsupport is due to inner working, can't fix that)"""
    # Define CSV fields order
    # If we were provided a fields_order list, we will show them first, else we create an empty fields_order
    if fields_order is None:
        fields_order = []
    # Get dict/list values
    if isinstance(d, dict):
        dvals = d.values()
    else:
        dvals = d
    # Then automatically add any other field (which order we don't care, they will be appended in alphabetical order)
    fields_order_check = set(fields_order)
    for missing_field in sorted(dvals[0]):
        if missing_field not in fields_order_check:
            fields_order.append(missing_field)
    if verbose:
        print('CSV fields order: '+str(fields_order))

    # Write the csv
    with open(output_file, 'wb') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, fields_order, delimiter=';')
        w.writeheader()
        # Reorder by name (or by any other column)
        if csv_order_by is not None:
            d_generator = sorted(dvals, key=lambda x: x[csv_order_by])
        else:
            d_generator = dvals
        # Walk the ordered list of dicts and write each as a row in the csv!
        for d_fields in d_generator:
            w.writerow(d_fields)
    return True


def save_df_as_csv(d, output_file, fields_order=None, csv_order_by=None, keep_index=False, encoding='utf-8-sig', blankna=False, excel=False, verbose=False, **kwargs):
    """Save a dataframe in a csv
    fields_order allows to precise what columns to put first (you don't need to specify all columns, only the ones you want first, the rest being alphabetically ordered). If None, alphabetical order will be used. If False, the original order will be used.
    csv_order_by allows to order rows according to the alphabetical order of the specified column(s)
    keep_index will save rows indices in the csv if True
    blankna fills None, NaN and NaT with an empty string ''. This can be useful for 2 purposes: 1) more human readability, 2) pandas will more easily understand a field is empty, even if the correct datatype is not set (eg, datetime null value is NaT, but at loading the column will be type 'object' which means NaT values won't be considered null). Note however that if you want to use date_format or float_format or decimal options of pd.to_csv(), they will not work since the columns datatypes will be converted to object/string.
    Encoding is by default 'utf-8-sig', which is UTF-8 with an encoding BOM in the file's header, this is necessary for Excel 2007 to correctly read the file (else it assumes latin-1): 
    If excel is True, will save as an excel file (which better supports accentuated/special characters).
    Combine with df_to_unicode() or df_to_unicode_fast() in case of encoding issues.
    """
    # Define CSV fields order
    # If we were provided a fields_order list, we will show them first, else we create an empty fields_order
    if fields_order is None:
        fields_order = []
    elif fields_order is False:
        fields_order = d.columns
    # Then automatically add any other field (which order we don't care, they will be appended in alphabetical order)
    fields_order_check = set(fields_order)
    for missing_field in sorted(d.columns):
        if missing_field not in fields_order_check:
            fields_order.append(missing_field)
    try:
        d = d.reindex(columns=fields_order)  # Reindex in case we supplied an empty column
    except ValueError as exc:
        if verbose:
            print('Warning: ValueError raised: %s' % exc)
        pass
    if verbose:
        print('CSV fields order: '+str(fields_order))
    # Blank none values
    if blankna:
        # Blank only non-datetime columns, else this will prevent reformatting on saving (because they will be considered as object/string and not date)
        #cols_except_date = [col for col in d.columns if d[col].dtype.name != 'datetime64[ns]']
        #d[cols_except_date] = d[cols_except_date].fillna('')
        # Blank every null values
        d = d.fillna('')
        # replace also any null value hidden as a string
        d[d.isna() | (d == 'None') | (d == 'NONE') | (d == 'none') | (d == 'NaN') | (d == 'nan') | (d == 'NaT') | (d == 'nat') | (d == 'na') | (d == 'NA') | (d == 'N/A')] = ''

    # Write the csv
    if csv_order_by is not None:
        d = d.sort_values(csv_order_by)
    else:
        d = d.sort_index()
    if not excel:
        if not output_file.endswith('.csv'):  # make sure the file extension is correct (else pandas will raise an error)
            output_file += '.csv'
        d.to_csv(output_file, sep=';', index=keep_index, columns=fields_order, encoding=encoding, **kwargs)
    else:
        if not output_file.endswith('.xls'):
            output_file += '.xls'
        d.to_excel(output_file, index=keep_index, columns=fields_order, encoding=encoding, **kwargs)
    return True


def distance_jaccard_words(seq1, seq2, partial=False, norm=False, dist=0, minlength=0):
    """Jaccard distance on two lists of words. Any permutation is tested, so the resulting distance is insensitive to words order.
    @param dist float Set to any value above 0 to get fuzzy matching (ie, a word matches if character distance < dist)
    @param partial boolean Set to True to match words if one of the two starts with the other (eg: 'allan' and 'al' will match) - combine with minlength to ensure a minimum length to match
    @param norm [True/False/None] True to get normalized result (number of equal words divided by total words count from both), False to get the number of different words, None to get the number of equal words.
    @param minlength int Minimum number of caracters to allow comparison and matching between words (else a word smaller than this will be a 'dead weight' if norm=True in the sense that it will still be accounted in the total but cannot be matched)"""
    # The goal was to have a distance on words that 1- is insensible to permutation ; 2- returns 0.2 or less if only one or two words are different, except if one of the lists has only one entry! ; 3- insensible to shortened name ; 4- allow for similar but not totally exact words.
    seq1_c = filter(None, list(seq1))
    seq2_c = filter(None, list(seq2))
    count_total = len(seq1_c) + len(seq2_c)
    count_eq = 0
    for s1 in seq1_c:
        flag_eq = False
        for skey, s2 in enumerate(seq2_c):
            if minlength and (len(s1) < minlength or len(s2) < minlength):
                continue
            if s1 == s2 or \
            (partial and (s1.startswith(s2) or s2.startswith(s1))) or \
            (dist and distance.nlevenshtein(s1, s2, method=1) <= dist):
                count_eq += 1
                del seq2_c[skey]
                break
    # Prepare the result to return
    if norm is None:
        # Just return the count of equal words
        return count_eq
    else:
        count_eq *= 2  # multiply by two because everytime we compared two items
        if norm:
            # Normalize distance
            return 1.0 - (float(count_eq) / count_total)
        else:
            # Return number of different words
            return count_total - count_eq

def distance_jaccard_words_split(s1, s2, *args, **kwargs):
    """Split sentences in words and call distance jaccard for words"""
    wordsplit_pattern = kwargs.get('wordsplit_pattern', None)
    if 'wordsplit_pattern' in kwargs:
        del kwargs['wordsplit_pattern']
    if not wordsplit_pattern:
        wordsplit_pattern = r'\s+|_+|,+|\.+|/+';  # do not split on -+| because - indicates a single word/name

    return distance_jaccard_words(re.split(wordsplit_pattern, s1), re.split(wordsplit_pattern, s2), *args, **kwargs)

def fullpath(relpath):
    '''Relative path to absolute'''
    if (type(relpath) is object or hasattr(relpath, 'read')): # relpath is either an object or file-like, try to get its name
        relpath = relpath.name
    return os.path.abspath(os.path.expanduser(relpath))

def recwalk(inputpath, sorting=True, folders=False, topdown=True, filetype=None):
    '''Recursively walk through a folder. This provides a mean to flatten out the files restitution (necessary to show a progress bar). This is a generator.'''
    noextflag = False
    if filetype and isinstance(filetype, list):
        filetype = list(filetype) # make a copy to avoid modifying the input variable (in case it gets reused externally)
        if '' in filetype:  # special case: we accept when there is no extension, then we don't supply to endswith() because it would accept any filetype then, we check this case separately
            noextflag = True
            filetype.remove('')
        filetype = tuple(filetype)  # str.endswith() only accepts a tuple, not a list
    # If it's only a single file, return this single file
    if os.path.isfile(inputpath):
        abs_path = fullpath(inputpath)
        yield os.path.dirname(abs_path), os.path.basename(abs_path)
    # Else if it's a folder, walk recursively and return every files
    else:
        for dirpath, dirs, files in walk(inputpath, topdown=topdown):	
            if sorting:
                files.sort()
                dirs.sort()  # sort directories in-place for ordered recursive walking
            # return each file
            for filename in files:
                if not filetype or filename.endswith(filetype) or (noextflag and not '.' in filename):
                    yield (dirpath, filename)  # return directory (full path) and filename
            # return each directory
            if folders:
                for folder in dirs:
                    yield (dirpath, folder)

def create_dir_if_not_exist(path):
    """Create a directory if it does not already exist, else nothing is done and no error is return"""
    if not os.path.exists(path):
        os.makedirs(path)

def real_copy(srcfile, dstfile):
    """Copy a file or a folder and keep stats"""
    shutil.copyfile(srcfile, dstfile)
    shutil.copystat(srcfile, dstfile)

def symbolic_copy(srcfile, dstfile):
    """Create a symlink (symbolic/soft link) instead of a real copy"""
    os.symlink(srcfile, dstfile)

def sort_list_a_given_list_b(list_a, list_b):
    return sorted(list_a, key=lambda x: list_b.index(x))

def replace_buggy_accents(s, encoding=None):
    """Fix weird encodings that even ftfy cannot fix"""
    # todo enhance speed? or is it the new regex on name?
    dic_replace = {
        '\xc4\x82\xc2\xa8': 'e',
        'ĂŠ': 'e',
        'Ăť': 'u',
        'â': 'a',
        'Ă´': 'o',
        'Â°': '°',
        'â': "'",
        'ĂŞ': 'e',
        'ÂŤ': '«',
        'Âť': '»',
        'Ă': 'a',
        'AŠ': 'e',
        'AŞ': 'e',
        'A¨': 'e',
        'A¨': 'e',
        'Ă': 'E',
        'â˘': '*',
        'č': 'e',
        '’': '\'',
    }
    # Convert the patterns to unicode if the input is a unicode string
    if isinstance(s, unicode):
        dic_replace = {k.decode('utf-8'): v.decode('utf-8') for k,v in dic_replace.items()}
    # Replace each pattern with its correct counterpart
    for pat, rep in dic_replace.items():
        if encoding:
            pat = pat.decode(encoding)
            rep = rep.decode(encoding)
        s = s.replace(pat, rep)
    return s

def cleanup_name(s, encoding=None, normalize=True, clean_nonletters=True):
    """Clean a name and remove accentuated characters"""
    if not isinstance(s, unicode):
        # Decode only if the input string is not already unicode (decoding is from str to unicode, encoding is from unicode to str)
        if encoding is None:
            encoding = chardet.detect(s)['encoding']
        if encoding:
            s = s.decode(encoding)
    s = _unidecode(s.replace('^', ' '))
    if normalize:
        s = s.lower().strip()
    if clean_nonletters:
        s = re.sub('\-+', '-', re.sub('\s+', ' ', re.sub('[^a-zA-Z0-9\-]', ' ', s))).strip().replace('\r', '').replace('\n', '').replace('\t', '').replace(',', ' ').replace('  ', ' ').strip()  # clean up spaces, punctuation and double dashes in name
    return s

# Compute best diagnosis for each patient
def compute_best_diag(serie, diag_order=None, persubject=True):
    """Convert a serie to a categorical type and extract the best diagnosis for each subject (patient name must be set as index level 0)
    Note: case insensitive and strip spaces automatically
    Set persubject to None if you want to do the max or min yourself (this will return the Series configured with discrete datatype)"""
    if diag_order is None:
        diag_order = ['coma', 'vs/uws', 'mcs', 'mcs-', 'mcs+', 'emcs', 'lis']  # from least to best
    # Convert to lowercase
    diag_order = [x.lower().strip() for x in diag_order]
    # Check if our list of diagnosis covers all possible in the database, else raise an error
    possible_diags = serie.str.lower().str.strip().dropna().unique()
    # If unicode, we convert the diag_order to unicode
    if isinstance(possible_diags[0].lower().strip(), unicode):
        diag_order = list_to_unicode(diag_order)
    try:
        assert not set([x.lower().strip() for x in possible_diags]) - set([x.lower().strip() for x in diag_order])
    except Exception as exc:
        raise ValueError('The provided list of diagnosis does not cover all possible diagnosis in database. Please fix the list. Here are the possible diagnosis from database: %s' % str(possible_diags))

    #for subjname, d in cf_crsr_all.groupby(level=0):
    #    print(d['CRSr::Computed Outcome'])

    if persubject:
        # Return one result per patient
        return serie.str.lower().str.strip().astype(pd.api.types.CategoricalDtype(categories=diag_order, ordered=True)).max(level=0)
    elif persubject is False:
        # Respect the original keys and return one result for each key (can be multilevel, eg subject + date)
        return serie.str.lower().str.strip().astype(pd.api.types.CategoricalDtype(categories=diag_order, ordered=True)).groupby(level=range(serie.index.nlevels)).max()
    else:
        # If None, just return the Serie as-is, and the user can do .max() or .min() or whatever
		return serie.str.lower().str.strip().astype(pd.api.types.CategoricalDtype(categories=diag_order, ordered=True))

def ordereddict_change_key(d, old, new):
    """Rename the key of an ordered dict without changing the order"""
    # from https://stackoverflow.com/a/17747040
    d2 = d.copy()
    for _ in range(len(d2)):
        k, v = d2.popitem(False)
        d2[new if old == k else k] = v
    return d2

def merge_two_df(df1, df2, col='Name', dist_threshold=0.2, dist_words_threshold=0.4, mode=0, skip_sanity=False, keep_nulls=True, returnmerged=False, keep_lastname_only=False, prependcols=None, fillna=False, fillna_exclude=None, join_on_shared_keys=True, squish=True, verbose=False, **kwargs):
    """Compute the remapping between two dataframes (or a single duplicated dataframe) based on one or multiple columns. Supports similarity matching (normalized character-wise AND words-wise levenshtein distance) for names, and date comparison using provided formatting.
    mode=0 is or test, 1 is and test. In other words, this is a join with fuzzy matching on one column (based on name/id) but supporting multiple columns with exact matching for the others.
    `keep_nulls=True` if you want to keep all records from both databases, False if you want only the ones that match both databases, 1 or 2 if you want specifically the ones that are in 1 or in 2
    `col` can either be a string for a merge based on a single column (usually name), or an OrderedDict of multiple columns names and types, following this formatting: [OrderedDict([('column1_name', 'column1_type'), ('column2_name', 'column2_type')]), OrderedDict([...])] so that you have one ordered dict for each input dataframe, and with the same number of columns (even if the names are different) and in the same order (eg, name is first column, date of acquisition as second column, etc)
    If `fillna=True`, subjects with multiple rows/sessions will be squashed and rows with missing infos will be completed from rows where the info is available (in case there are multiple information, they will all be present as a list). This is only useful when merging (and hence argument `col`) is multi-columns. The key columns are never filled, even if `fillna=True`.
    If `fillna_exclude` is specified with a list of columns, this list of columns won't be filled (particularly useful for dates).
    if `join_on_shared_keys=True`, if merging on multi-columns and not the same number of key columns are supplied, the merge will be done on only the shared keys in both dataframes: this is very convenient to allow to groupby in one dataframe according to some keys but not in the other one (eg, one is grouped by name and date so both are kept, while the other one is only grouped by name).
    if `squish=True`, the dataframes are each squished on key columns to make them unique, so that other non-key columns will get concatenated values. True by default, but if you have to non overlapping databases, then you can set this to False to keep all rows.
    """
    ### Preparing the input dataframes
    # If the key column is in fact a list of columns (so we will merge on multiple columns), we first extract and rename the id columns for ease
    if isinstance(col, list):
        # Make a backup of the columns
        keycol = list(col)
        # Make copies of each dict to avoid modifying the originals AND to disambiguate if we are working on a single dataframe and thus list (else modifying one list will also modify the 2nd one!)
        keycol[0] = keycol[0].copy()
        keycol[1] = keycol[1].copy()
        # Extract id columns (type = 'id')
        keyid1 = [(x, y) for x,y in keycol[0].items() if y == 'id'][0][0]
        keyid2 = [(x, y) for x,y in keycol[1].items() if y == 'id'][0][0]
        # Rename second dataframe to have the same id column name as the first (easier merge)
        df2.rename(columns={keyid2: keyid1}, inplace=True)
        # Use the id column of first dataframe as the main joining column
        col = keyid1
        # Rename in our keycol
        if keyid2 != keyid1:
            keycol[1][keyid1] = keycol[1][keyid2]
            del keycol[1][keyid2]
    else:
        keycol = None
    # Reset keys
    #df1.reset_index(drop=True, inplace=True)
    #df2.reset_index(drop=True, inplace=True)
    # Find and rename any column "Name" or "NAME" to lowercase "name"
    df1 = df_cols_lower(df1, col=col)
    df2 = df_cols_lower(df2, col=col)
    col = col.lower()  # lowercase also the id column
    #if keycol:  # lowercase also the other key columns
        #keycol = [OrderedDict([(k.lower(), v) for k,v in kcol.items()]) for kcol in keycol]
    # drop all rows where all cells are empty or where name is empty (necessary else this will produce an error, we expect the name to exist)
    if squish:
        df1 = df1.dropna(how='all').dropna(how='any', subset=[col])
        df2 = df2.dropna(how='all').dropna(how='any', subset=[col])
    # Rename all columns if user wants, except the key columns (else the merge would not work) - this is an alternative to automatic column renaming in case of conflict
    if prependcols is not None and len(prependcols) == 2:
        if keycol:
            # Multi-columns merging: we drop all the key columns
            df1.rename(columns={c: prependcols[0]+c for c in df1.columns.drop(keycol[0].keys())}, inplace=True)
            df2.rename(columns={c: prependcols[1]+c for c in df2.columns.drop(keycol[1].keys())}, inplace=True)
        else:
            # Single column merging: we drop the one key column
            df1.rename(columns={c: prependcols[0]+c for c in df1.columns.drop(col)}, inplace=True)
            df2.rename(columns={c: prependcols[1]+c for c in df2.columns.drop(col)}, inplace=True)
    # Check if columns are colluding (ie, apart from keys, some columns have the same name) then rename them by prepending "a." and "b."
    cols_conflict = set(df1.columns).intersection(set(df2.columns)).difference(set([col]))
    if len(cols_conflict) > 0:
        if verbose:
            print('Warning: columns naming conflicts detected: will automatically rename the following columns to avoid issues: %s' % str(cols_conflict))
        # Rename the colluding columns in the dataframes
        df1.rename(columns={c: 'a.'+c for c in cols_conflict}, inplace=True)
        df2.rename(columns={c: 'b.'+c for c in cols_conflict}, inplace=True)
        # Rename also the key columns if they are part of the colluding columns
        if keycol:
            for x,y in zip(keycol[0].keys(), keycol[1].keys()):
                if x in cols_conflict:
                    keycol[0] = ordereddict_change_key(keycol[0], x, 'a.'+x)
                if y in cols_conflict:
                    keycol[1] = ordereddict_change_key(keycol[1], y, 'b.'+y)
    # Make a backup of the original name
    df1[col+'_orig'] = df1[col]
    df2[col+'_orig2'] = df2[col]
    # if doing multiple consecutive merges, a name can in fact be a list of concatenated names, then extract the first name in the list
    # TODO: enhance this to account for all names when comparing
    df1[col] = df1[col].apply(lambda x: df_literal_eval(x)[0] if isinstance(df_literal_eval(x), list) else x)
    df2[col] = df2[col].apply(lambda x: df_literal_eval(x)[0] if isinstance(df_literal_eval(x), list) else x)
    # keep only the lastname (supposed to be first), this can ease comparison
    if keep_lastname_only:
        df1[col] = df1[col].apply(lambda x: x.split()[0])
        df2[col] = df2[col].apply(lambda x: x.split()[0])
    ### Prepare merging variables
    dmerge = []  # result of the merge mapping
    list_names1 = df1[col].unique()
    list_names2 = df2[col].unique()
    ### Merge mapping construction based on id (name) column
    # Find all similar names in df2 compared to df1 (missing names will be None)
    for c in _tqdm(list_names1, total=len(df1[col].unique()), desc='MERGE'):
        found = False
        #c = str(c)
        if dist_threshold <= 0.0 and dist_words_threshold <= 0.0:
            # No fuzzy matching, we simply compute equality
            if c in list_names2:
                dmerge.append( (c, c) )
                found = True
        else:
            # Fuzzy matching
            for cd in list_names2:
                if not cd:
                    continue
                #cd = str(cd)
                # Clean up the names
                name1 = cleanup_name(replace_buggy_accents(c))
                name2 = cleanup_name(replace_buggy_accents(cd))
                # Compute similarity
                testsim1 = distance.nlevenshtein(name1, name2, method=1) <= dist_threshold  # character-wise distance on the whole name
                testsim2 = distance_jaccard_words_split(name1, name2, partial=False, norm=True, dist=dist_threshold) <= dist_words_threshold  # word-wise distance
                if (mode==0 and (testsim1 or testsim2)) or (mode==1 and testsim1 and testsim2): # use shortest distance with normalized levenshtein
                    # Found a similar name in both df, add the names
                    dmerge.append( (c, cd) )
                    found = True
        # Did not find any similar name, add as None
        if not found:
            dmerge.append( (c, None) )
    # Find all names missing in df1 compared to df2
    missing = [(None, x) for x in list(set(list_names2) - set([y for _,y in dmerge if y is not None]))]
    dmerge.extend(missing)
    # Convert to a dataframe
    dmerge = pd.DataFrame(dmerge, columns=[col, col+'2'])
    # Sanity check
    if not skip_sanity:
        for n in list_names2:
            try:
                assert dmerge[dmerge[col+'2'] == n].count().max() <= 1
            except AssertionError as exc:
                raise AssertionError('Conflict found: a subject has more than one mapping! Subject: %s %s' % (n, dmerge[dmerge[col+'2'] == n]))
        for n in list_names1:
            try:
                assert dmerge[dmerge[col] == n].count().max() <= 1
            except AssertionError as exc:
                raise AssertionError('Conflict found: a subject has more than one mapping! Subject: %s %s' % (n, dmerge[dmerge[col] == n]))
    ### Mapping finished
    if not returnmerged:
        # Return merge mapping result!
        return dmerge
    else:
        ### Return not only the ID merge result but a unified DataFrame merging both whole databases (ie, all columns)
        if keep_nulls is True:
            # Recopy empty matchs from the other database (so that we don't lose them after the final merge)
            dmerge.loc[dmerge[col].isnull(), col] = dmerge[col+'2']
            dmerge.loc[dmerge[col].isnull(), col+'2'] = dmerge[col]
        elif keep_nulls is False:
            # Drop nulls in both dataframes (ie, drop any name that has no correspondance in both dataframes)
            dmerge = dmerge.dropna(how='any')
            df1 = df1.loc[df1[col].isin(dmerge[col]),:]
            df2 = df2.loc[df2[col].isin(dmerge[col+'2']),:]
        elif keep_nulls == 1:
            # Keep nulls only in 1st dataframe (drop in second)
            dmerge = dmerge.dropna(how='any', subset=[col+'2'])
            df2 = df2.loc[df2[col].isin(dmerge[col+'2']),:]
        elif keep_nulls == 2:
            # Keep nulls only in 2nd dataframe (drop in first)
            dmerge = dmerge.dropna(how='any', subset=[col])
            df1 = df1.loc[df1[col].isin(dmerge[col]),:]
        # Remap IDs so that 2nd dataframe has same names as in 1st dataframe (to ease comparisons)
        df2 = df_remap_names(df2, dmerge, col, col+'2', keep_nulls=keep_nulls)
        del df2['index']
        # Merging databases
        if keycol is None:
            # Simple merging on one key column (name usually)
            # Concatenate all rows into one per gupi
            if squish:
                df1 = df_concatenate_all_but(df1, col, setindex=False)
                df2 = df_concatenate_all_but(df2, col, setindex=False)
            # Final merge of all columns
            dfinal = pd.merge(df1, df2, how='outer', on=col)
        else:
            # More complex merging on multiple key columns
            # Concatenate all rows into one per key columns (ie, we aggregate on the key columns)
            if squish:
                df1 = df_concatenate_all_but(df1, keycol[0].keys(), setindex=False)
                df2 = df_concatenate_all_but(df2, keycol[1].keys(), setindex=False)
            # Preprocessing key columns with specific datatypes (eg, datetime), else they won't be comparable across the two input DataFrames
            for dfid, kcol in zip([1, 2], keycol):  # for each dataframe and the corresponding key columns list
                for colname, coltype in kcol.items():  # loop through each key column for this dataframe
                    # datetime type: will convert the column into a datetime format by interpreting it given the formatting provided by user in the following format, separated by a '|': 'datetime|%formatting%here'
                    if coltype.startswith('datetime'):
                        if dfid == 1:
                            df1 = convert_to_datetype(df1, colname, coltype.split('|')[1])
                        elif dfid == 2:
                            df2 = convert_to_datetype(df2, colname, coltype.split('|')[1])
                    # other types: we do nothing (the id type is already taken care of at the beginning of the function, the rest is undefined at the moment)
            # Final merge of all columns
            #return df1, df2  # debug: in case of bug, return df1 and df2 and try to do the subsequent commands in a notebook, easiest way to debug
            if len(keycol[0].keys()) == len(keycol[1].keys()):
                # Merge on multiple columns, we need the same number and types of key columns in both dataframes (but not necessarily same column names)
                dfinal = pd.merge(df1, df2, how='outer', left_on=keycol[0].keys(), right_on=keycol[1].keys(), **kwargs)
            elif join_on_shared_keys:
                # Find all keys that are shared, but try to preserve the order
                keycol_shared = []
                for k in keycol[0].keys():
                    for k2 in keycol[1].keys():
                        # If column name is the same, we add this column in the list of shared ones
                        if k == k2:
                            keycol_shared.append(k)
                # Merge on shared keys only
                if keycol_shared:
                    dfinal = pd.merge(df1, df2, how='outer', on=keycol_shared, **kwargs)
                else:
                    # Fallback to merge on the first x columns (positional merging)
                    # TODO: this does NOT work because higher in func we replace the name of 2nd dataframe key column in case they are mismatched. Thus, this branch is never called. The old/current solution works for one column, but not if we have multiple, positional merging is a more elegant and flexible solution, but need to fix this old code above before.
                    keycol_shared_len = min(len(keycol[0]), len(keycol[1]))
                    dfinal = pd.merge(df1, df2, how='outer', left_on=keycol[0].keys()[:keycol_shared_len], right_on=keycol[0].keys()[:keycol_shared_len], **kwargs)
            else:
                raise ValueError('To merge two dataframes on multiple keys, you need to provide the same number of keys with the same types (but not necessarily the same names) for both dataframes! Or you can enable join_on_shared_key=True to do a partial merge on the keys that are shared (then it is based on sharing the same column name - the id columns should have the same name internally in any case).')
            # Filling gaps by squashing per subject id and trying to fill the missing fields from another row of the same subject where the information is present
            if fillna:
                # Drop duplicated columns (probably the 'index_x' and 'index_y' that are produced automatically by a previous pd.merge()), this is necessary else groupby().agg() will fail on dataframes with duplicated columns: https://stackoverflow.com/questions/27719407/pandas-concat-valueerror-shape-of-passed-values-is-blah-indices-imply-blah2
                if not dfinal.columns.is_unique:
                    print('Duplicated columns found, dropping them automatically: %s' % dfinal.columns[dfinal.columns.duplicated()])
                    dfinal = dfinal.drop(columns=dfinal.columns[dfinal.columns.duplicated()])
                    if not dfinal.columns.is_unique:
                        raise ValueError('There are still duplicated columns! Please contact the developer to fix the issue!')
                # Drop duplicated rows (can only be done if columns are deduplicated)
                if not dfinal.index.is_unique:
                    print('Duplicated rows found, dropping them automatically, please contact the developer to fix this issue (as this should not happen!).')
                    dfinal = dfinal.drop_duplicates(keep='first')
                # Generate an aggregate only based on id
                if fillna_exclude is None:
                    fillna_exclude = []
                dfinal_agg = dfinal.drop(columns=[colname for kcol in keycol for colname, coltype in kcol.items() if coltype != 'id']+fillna_exclude).fillna('').groupby(by=col, sort=False).agg(concat_vals_unique)
                # Fill nan values from other rows of the same subject, by using pandas.combine_first() function
                dfinal = dfinal.set_index(col).combine_first(dfinal_agg.reset_index().set_index(col)).reset_index()
            # Create new columns merging key columns info (similarly to names), this can be useful for further merge (ie, to use one merged key column instead of two different)
            for kcol1, kcol2 in zip(keycol[0].items(), keycol[1].items()):
                col1name, col1type = kcol1
                col2name, col2type = kcol2
                if col1type != 'id':
                    colcombined = col1name+' + '+col2name
                    # Copy from first dataframe's key column values
                    dfinal[colcombined] = dfinal[col1name]
                    # Copy datatype from the original dataframe. Since pandas.merge() can lose (but not always) the datatype info of the columns after merging, we need to use the original dataframes where we already converted the key columns to the correct datatypes: https://stackoverflow.com/questions/29280393/python-pandas-merge-loses-categorical-columns
                    dfinal[colcombined].astype(df1[col1name].dtype, inplace=True)
                    # Finally, where there is an empty value, copy from the second column coming from the second dataframe (from the corresponding key column) - we copy from dfinal to ensure the index is correct (else we merge on a random index and this is bad!)
                    dfinal.loc[dfinal[col1name].isnull(), colcombined] = dfinal[col2name]
        # Keep log of original names from both databases, by creating other columns "_altx_orig", "_altx_orig2" and "_anyx" to store the name from 2nd database and create a column with any name from first db or second db
        for x in range(1000):
            # If we do multiple merge, we will have multiple name_alt columns: name_alt0, name_alt1, etc
            alt_id = col+'_alt%i_orig' % (x+1)
            alt_id2 = col+'_alt%i_orig2' % (x+1)
            alt_id3 = col+'_alt%i_cleanup' % (x+1)
            if not alt_id in dfinal.columns and not alt_id2 in dfinal.columns:
                # Rename the name column from the 2nd database
                dfinal.insert(1, alt_id, dfinal[col+'_orig']) # insert the column just after 'name' for ergonomy
                dfinal.insert(2, alt_id2, dfinal[col+'_orig2'])
                # Also add the cleaned up name as a new column, can ease search by user (no accentuated characters)
                dfinal.insert(3, alt_id3, dfinal[col])
                dfinal = cleanup_name_df(dfinal, col=alt_id3)

                # Finally delete the useless column (that we copied over to name_altx)
                del dfinal[col+'_orig']
                del dfinal[col+'_orig2']

                # Finish! We found an non existent alt%i number so we can just stop here
                break
        # Return both the merge mapping and the final merged database
        return (dmerge, dfinal)

def remove_strings_from_df(df):
    """Remove all strings from a dataframe and replace by nan and convert to float (can supply a subset of columns)"""
    # Courtesy of instant: https://stackoverflow.com/a/41941267/1121352
    def isnumber(x):
        try:
            float(x)
            return True
        except:
            return False
    return df[df.applymap(isnumber)].applymap(float)

def concat_vals(x, aggfunc=None):
    """Concatenate after a groupby values in a list, and keep the same order (except if all values are the same or null, then return a singleton). This is similar to groupby(col).agg(list) but this function returns a singleton whenever possible (for readability).
    Optionally can provide an aggfunc that will be applied to select only one value (eg, max)."""
    try:
        x = list(x)
        if len(sort_and_deduplicate(x)) == 1:
            x = x[0]
        elif len([y for y in x if ((isinstance(y, list) or not pd.isnull(y)) and (hasattr(y, '__len__') and len(y) > 0)) or isinstance(y, numbers.Number)]) == 0:
            x = None
        if aggfunc is not None and isinstance(x, list):
            x = aggfunc(x)
    except Exception as exc:
        # Warning: pd.groupby().agg(concat_vals) can drop columns without a notice if an exception happens during the execution of the function
        print('Warning: aggregation using concat_vals() met with an exception, at least one column will be dropped!')
        print(exc)
        raise
    return x

def concat_vals_unique(x, aggfunc=None):
    """Concatenate after a groupby values in a list (if not null and not unique, else return the singleton value)
    Optionally can provide an aggfunc that will be applied to select only one value (eg, max).
    Please make sure your DataFrame contains only unique columns, else you might get a weird error: https://stackoverflow.com/questions/27719407/pandas-concat-valueerror-shape-of-passed-values-is-blah-indices-imply-blah2"""
    try:
        x = list(sort_and_deduplicate([y for y in x if ((isinstance(y, list) or not pd.isnull(y)) and (hasattr(y, '__len__') and len(y) > 0)) or isinstance(y, numbers.Number)]))
        if len(x) == 1:
            x = x[0]
        elif len(x) == 0:
            x = None
        if aggfunc is not None and isinstance(x, list):
            x = aggfunc(x)
    except Exception as exc:
        # Warning: pd.groupby().agg(concat_vals) can drop columns without a notice if an exception happens during the execution of the function
        print('Warning: aggregation using concat_vals_unique() met with an exception, at least one column will be dropped!')
        print(exc)
        raise
    return x

def uniq(lst):
    # From https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function
    last = object()
    for item in lst:
        if item == last:
            continue
        yield item
        last = item

def sort_and_deduplicate(l):
    """Alternative to using set(list()) with no computational overhead, since set() is limited to hashable types (hence not lists)"""
    # From https://stackoverflow.com/questions/13464152/typeerror-unhashable-type-list-when-using-built-in-set-function
    return list(uniq(sorted(l, reverse=True)))

def concat_strings(serie, prefix='', sep=''):
    """Concatenate multiple columns as one string. Can add a prefix to make sure Pandas saves the column as a string (and does not trim leading 0)"""
    return prefix+sep.join([x if isinstance(x, (str,unicode)) else str(int(x)) if not pd.isnull(x) else '' for x in serie])

def find_columns_matching(df, L, startswith=False):
    """Find all columns of a DataFrame matching one string of the given list (case insensitive)"""
    if isinstance(L, (str,unicode)):
        L = [L]
    L = [l.lower() for l in L]
    matching_columns = []
    for col in df.columns:
        if (not startswith and any(s in col.lower() for s in L)) or \
            (startswith and any(col.lower().startswith(s) for s in L)):
            matching_columns.append(col)
    return matching_columns

def reorder_cols_df(df, cols):
    """Reorder the columns of a DataFrame to start with the provided list of columns"""
    cols2 = [c for c in cols if c in df.columns.tolist()]
    cols_without = df.columns.tolist()
    for col in cols2:
        cols_without.remove(col)
    return df[cols2 + cols_without]

def df_remap_names(df, dfremap, col='name', col2='name2', keep_nulls=False):
    """Remap names in a dataframe using a remapping list (ie, a dataframe with two names columns)"""
    def replace_nonnull_df(x, repmap):
        replacement = repmap[x]
        return replacement if replacement is not None else x
    repmap = dfremap.set_index(col2)[col].to_dict()
    df2 = df.copy().reset_index()
    if keep_nulls:
        # Much faster but if there are nulls they will be replaced
        df2[col] = df2[col].map(repmap)
    else:
        # Slower but remap only if the remap is not null
        df2[col] = df2[col].apply(lambda x: replace_nonnull_df(x, repmap))
    return df2

def convert_to_datetype(df, col, dtformat=None, **kwargs):
    """Convert a column of a dataframe to date type with the given format"""
    df2 = df.copy()
    if not df2.index.name is None:
        df2 = df2.reset_index()
    try:
        df2[col] = pd.to_datetime(df2[col], format=dtformat, **kwargs)
    except Exception as exc:
        print('Warning: cannot convert column %s as datetype using pandas.to_datetime() and formatting %s and supplied format, falling back to fuzzy matching date format (this might introduce buggy dates, you should check manually afterwards)...' % (col, dtformat))
        df2 = df_date_clean(df2, col)
    return df2

def df_drop_duplicated_index(df):
    """Drop all duplicated indices in a dataframe or series"""
    df2 = df.copy()
    return df2[~df2.index.duplicated(keep='first')]

def cleanup_name_df(df, col='name'):
    """Cleanup the name field of a dataframe"""
    df2 = df.copy()
    try:
        df2[col] = df2[col].astype('str').apply(lambda name: cleanup_name(name))
    except UnicodeEncodeError as exc:
        df2[col] = df2[col].astype('unicode').apply(lambda name: cleanup_name(name))
    return df2
    #for c in df2.itertuples():  # DEPRECATED: itertuples() is limited to 255 columns in Python < 3.7, prefer to avoid this approach
    #    try:
    #        df2.loc[c.Index, 'name'] = cleanup_name(c.name)
    #    except Exception as exc:
    #        print("An error occurred while cleaning up names in the provided dataframe, please check the following lines which might be the culprit:")
    #        print(df2[df2['name'].isnull()])
    #        raise
    #return df2

def cleanup_name_customregex(cname, customregex=None, returnmatches=False):
    """Cleanup the input name given a custom dictionary of regular expressions (format of customregex: a dict like {'regex-pattern': 'replacement'}"""
    if customregex is None:
        customregex = {'_': ' ',
                       'repos': '',
                       'ecg': '',
                       '[0-9]+': '',
                      }
    matches = set()
    # For each pattern
    for pattern, replacement in customregex.iteritems():
        # First try to see if there is a match and store it if yes
        if returnmatches:
            m = re.search(pattern, cname, flags=re.I)
            if m:
                matches.add(m.group(0))
        # Then replace the pattern found
        cname = re.sub(pattern, replacement, cname, flags=re.I)

    # Return both the cleaned name and matches
    if returnmatches:
        return (cname, matches)
    # Return just the cleaned name
    else:
        return cname

def cleanup_name_customregex_df(cf, customregex=None):
    """Cleanup the name fields of a dataframe given a custom dictionary of regular expressions (format of customregex: a dict like {'regex-pattern': 'replacement'}"""
    cf['name'] = cf['name'].apply(lambda name: cleanup_name_customregex(name, customregex))
    return cf

def compute_names_distance_matrix(list1, list2, dist_threshold_letters=0.2, dist_threshold_words=0.2, dist_threshold_words_norm=True, dist_minlength=0):
    """Find all similar items in two lists that are below a specified distance threshold (using both letters- and words- levenshtein distances). This is different from disambiguate_names() which is working on a single dataframe (trying to uniformize the names (mis)spellings).
    Note: this works less efficiently than merge_two_df(), you should use the latter."""
    dist_matches = {}
    for subj in _tqdm(list1, total=len(list1), desc='MERGE'):
        found = False
        subj = cleanup_name(replace_buggy_accents(subj))
        for c in list2:
            c = cleanup_name(replace_buggy_accents(c))
            # use shortest distance with either normalized levenshtein distance or non-normalized levenshtein distance
            if distance.nlevenshtein(subj, c, method=1) <= dist_threshold_letters or (
                (dist_threshold_words_norm is not None and distance_jaccard_words_split(subj, c, partial=False, norm=dist_threshold_words_norm, dist=dist_threshold_letters, minlength=dist_minlength) <= dist_threshold_words) or
                (dist_threshold_words_norm is None and distance_jaccard_words_split(subj, c, partial=False, norm=dist_threshold_words_norm, dist=dist_threshold_letters, minlength=dist_minlength) >= dist_threshold_words)
                ):
                if subj not in dist_matches:
                    dist_matches[subj] = []
                dist_matches[subj].append(c)
                found = True
        if not found:
            dist_matches[subj] = None
    # Remove duplicate values (ie, csv names)
    dist_matches = {k: (list(set(v)) if v else v) for k, v in dist_matches.items()}
    return dist_matches

def disambiguate_names(df, dist_threshold=0.2, col='name', verbose=False): # TODO: replace by the updated compute_names_distance_matrix() function?
    """Disambiguate names in a single dataframe, in other words finds all the different (mis)spellings of the same person's name and uniformize them all, so that we can easily find all the records pertaining to a single subject. In other words, this is like a join with fuzzy matching on a single column and dataframe. This is different from compute_names_distance_matrix() function which works on two different dataframes."""
    df2 = df.copy()
    df2 = df2.assign(**{col+'_alt': ''})  # create new column name_alt with empty values by default
    for cindex, c in _tqdm(df2.iterrows(), total=len(df2)): # Updated to use iterrows() to support more than 255 columns (because itertuples() is limited by Python 2 limit of 255 max items in a tuple, this might have changed in Python 3!)
        for c2index, c2 in df2.ix[cindex+1:,:].iterrows():
            if c[col] != c2[col] and \
            (distance.nlevenshtein(c[col], c2[col], method=1) <= dist_threshold or distance_jaccard_words_split(c2[col], c[col], partial=False, norm=True, dist=dist_threshold) <= dist_threshold): # use shortest distance with normalized levenshtein
                if verbose:
                    print(c[col], c2[col], c2index, distance.nlevenshtein(c[col], c2[col], method=1))
                # Replace the name of the second entry with the name of the first entry
                df2.loc[c2index, col] = c[col]
                # Add the other name as an alternative name, just in case we did a mistake for example
                df2.loc[cindex, col+'_alt'] = df2.loc[cindex, col+'_alt'] + '/' + c2[col] if df2.loc[cindex, col+'_alt'] else c2[col]
    return df2

def df_concatenate_all_but(df, col, setindex=False):
    """Make sure each id (in col) is unique, else concatenate all other rows for each id into one row.
    col can either be a string for a single column name, or a list of column names to use for the aggregation."""
    df2 = df.copy()
    df2.loc[:,col] = df2.loc[:,col].fillna(value='')  # fill nan values with placeholder to avoid losing these rows, particularly with multiple columns as keys of groupby, this can lead to mysterious loss of rows. Indeed, pandas drops any row where the groupby key columns have a nan or nat value (in any of the key columns! Even if other key columns are filled!). There is currently no option to disable this behavior. See https://github.com/pandas-dev/pandas/issues/3729 and https://stackoverflow.com/questions/18429491/groupby-columns-with-nan-missing-values for more info.
    df2 = df2.reset_index().groupby(col, sort=False).agg(concat_vals)  # groupby the key columns and aggregate by concatenating duplicated values (using our custom function concat_vals)
    df2.reset_index(inplace=True)
    if setindex:
        df2.set_index(col, inplace=True)
    return df2

def dict_to_unicode(d):
    """Convert a dict of dict to unicode in order to be able to save via csv.DictWriter()
    See also: https://stackoverflow.com/questions/5838605/python-dictwriter-writing-utf-8-encoded-csv-files
    TODO: fix naming, it's not decoding to unicode, it's encoding to utf-8 (thus it's converting to a str)
    """
    d2 = copy.deepcopy(d)
    for k, v in d2.items():
        if isinstance(v, dict):
            d2[k] = {k2:v2.encode('utf8') for k2,v2 in v.items()}
        else:
            d2[k] = v.encode('utf8')
    return d2

def list_to_unicode(l):
    """Convert the items of a list of str string into unicode"""
    return [unicode(x.decode(encoding=chardet.detect(x)['encoding'])) if not isinstance(x, unicode) else x for x in l]

def string_to_unicode(s, failsafe_encoding='iso-8859-1', skip_errors=False):
    """Ensure unicode encoding for one string.
    If failing to convert to unicode, will use failsafe_encoding to attempt to decode.
    If skip_errors=True, the unicode encoding will be forced by skipping undecodable characters (errors='ignore').
    If failing, try df_to_unicode_fast(), which will strip out special characters and try to replace them with the closest ascii character.
    """
    try:
        # Try default unidecode
        s = unicode(s)
    except UnicodeDecodeError as exc:
        # If fail, we use the failsafe encoding
        try:
            s = s.decode(failsafe_encoding)
        except UnicodeDecodeError as exc2:
            # At worst, we can just skip errors (and unrecognized characters)
            if skip_errors:
                s = unicode(s, errors='ignore')
            else:
                raise
    return s

def df_to_unicode(df_in, cols=None, failsafe_encoding='iso-8859-1', skip_errors=False, progress_bar=False):
    """Ensure unicode encoding for all strings in the specified columns of a dataframe.
    If cols=None, will walk through all columns.
    If failing to convert to unicode, will use failsafe_encoding to attempt to decode.
    If skip_errors=True, the unicode encoding will be forced by skipping undecodable characters (errors='ignore').
    If failing, try df_to_unicode_fast(), which will strip out special characters and try to replace them with the closest ascii character.
    """
    if len(df_in) == 0:  # if empty, this will produce an error (when trying to reset the index)
        return df_in
    # Make a copy to avoid tampering the original
    df = df_in.copy()
    # If there is a complex index, it might contain strings, so we reset it as columns so that we can unidecode indices too, and we will restore the indices at the end
    if df.index.dtype.name == 'object' or isinstance(df.index, pd.core.indexes.multi.MultiIndex):
        idxbak = df.index.names
        df.reset_index(inplace=True)
    else:
        idxbak = None
    # Which columns do we have to unidecode?
    if cols is None:  # by default, all!
        # Ensure column names are unicode
        df.columns = [unicode(cleanup_name(x, normalize=False, clean_nonletters=False), errors='ignore') for x in df.columns]
        # By default, take all columns
        cols = df.columns
    # Calculate total number of items
    if progress_bar:
        pbar_total = 0
        for col in cols:
            pbar_total += len(df[col].index)
    # Main loop
    if progress_bar:
        pbar = _tqdm(total=pbar_total, desc='UNICODE', disable=(not progress_bar))
    for col in cols:
        for idx in df[col].index:
            # Unidecode if value is a string
            if isinstance(df.loc[idx,col], str):  # if item is already unicode, we don't need to do anything (also if item is null or another type, we do not need to decode)
                df.loc[idx,col] = string_to_unicode(df.loc[idx,col], failsafe_encoding=failsafe_encoding, skip_errors=skip_errors)
            if progress_bar:
                pbar.update()
    # Restore index
    if idxbak:
        df.set_index(idxbak, inplace=True)
    return df

def df_to_unicode_fast(df_in, cols=None, replace_ascii=False, skip_errors=False, progress_bar=False):
    """Ensure unicode encoding for all strings in the specified columns of a dataframe in a fast way, and optionally by replacing non recognized characters by ascii equivalents. Also ensures that columns names are correctly decodable as unicode if cols=None.
    If cols=None, will walk through all columns.
    If replace_ascii, will replace special characters with the closest ASCII counterpart (using unidecode) if the conversion to unicode fails
    If skip_errors=True, the unicode encoding will be forced by skipping undecodable characters (errors='ignore').
    The main difference with df_to_unicode() is that the former tries to maintain special characters (instead of replacing them with their closest ascii counterpart) and it is slower (but more thorough, it should not miss any field, whereas the fast version will work column by column and thus might miss a column of mixed types).
    """
    # Make a copy to avoid tampering the original
    df = df_in.copy()
    # If there is a complex index, it might contain strings, so we reset it as columns so that we can unidecode indices too, and we will restore the indices at the end
    if df.index.dtype.name == 'object' or isinstance(df.index, pd.core.indexes.multi.MultiIndex):
        idxbak = df.index.names
        df.reset_index(inplace=True)
    else:
        idxbak = None
    # Which columns do we have to unidecode?
    if cols is None:  # by default, all!
        # Ensure column names are unicode
        df.columns = [unicode(cleanup_name(x, normalize=False, clean_nonletters=False), errors='ignore') for x in df.columns]
        # Use the new column names
        cols = df.columns
    if skip_errors:
        serrors = 'ignore'
    else:
        serrors = 'strict'
    for col in _tqdm(cols, desc='UNICODE', disable=(not progress_bar)):
        # Verify that the column is of type object, else for sure it is not a string
        # also if there are duplicate names, just skip these columns
        # TODO: try to process columns with duplicate names
        if (len(df.loc[:, col].shape) > 1 and df.loc[:, col].shape[1] > 1) or df.loc[:, col].dtype.name != 'object':
            continue
        try:
            # First try a decoding by detecting the correct encoding
            #encoding = chardet.detect(''.join(df.loc[:, col]))['encoding']
            allvals = (x if isinstance(x, basestring) else str(x) for _, x in df.loc[:, col].items())
            allvalsjoined = ''.join(allvals)
            if isinstance(allvalsjoined, unicode):  # if unicode, skip decoding
                encoding = None
            else:
                encoding = chardet.detect(allvalsjoined)['encoding']
            if encoding:
                df.loc[:, col] = df.loc[:, col].apply(lambda x: x.decode(encoding, errors=serrors) if isinstance(x, str) else x)
            #df.loc[:, col] = df.loc[:, col].astype('unicode')  # DEPRECATED: works but if we do this, all null values (nan, nat, etc) will be converted to strings and become very difficult to process (eg, not detectable using pd.isnull())!
            #df[col] = df[col].map(lambda x: x.encode('unicode-escape').decode('utf-8'))
        except Exception as exc:
            try:
                # If decoding failed, we can try to replace the special characters with their closest ASCII counterpart (via unidecode)
                if replace_ascii:
                    df.loc[:, col] = df.loc[:, col].apply(lambda x: unicode(cleanup_name(x, normalize=False, clean_nonletters=False), errors=serrors) if isinstance(x, str) else x)
                    #df.loc[:, col] = df.loc[:, col].astype('unicode')  # DEPRECATED: works but if we do this, all null values (nan, nat, etc) will be converted to strings and become very difficult to process (eg, not detectable using pd.isnull())!
                else:
                    raise
            except Exception as exc:
                # Else everything failed!
                if skip_errors:
                    pass
                else:
                    print('Failed with column: %s' % col)
                    raise
    # Restore index
    if idxbak:
        df.set_index(idxbak, inplace=True)
    return df

def df_encode(df_in, cols=None, encoding='utf-8', skip_errors=False, decode_if_errors=False, progress_bar=False):
    """Encode all unicode strings in a dataframe into a string of the chosen encoding.
    When decode_if_errors is True, if a string (str) is found, an attempt will be made to decode it using an autodetection of the encoding, to make a unicode sandwich."""
    # Make a copy to avoid tampering the original
    df = df_in.copy()
    # If there is a complex index, it might contain strings, so we reset it as columns so that we can unidecode indices too, and we will restore the indices at the end
    if df.index.dtype.name == 'object' or isinstance(df.index, pd.core.indexes.multi.MultiIndex):
        idxbak = df.index.names
        df.reset_index(inplace=True)
    else:
        idxbak = None
    # Which columns do we have to unidecode?
    if cols is None:  # by default, all!
        # Ensure column names are encoded
        df.columns = [x.encode(encoding) for x in df.columns]
        # Use all columns, and the new column names encoded
        cols = df.columns
    # Calculate total number of items
    if progress_bar:
        pbar_total = 0
        for col in cols:
            pbar_total += len(df[col].index)
    # Main loop
    if progress_bar:
        pbar = _tqdm(total=pbar_total, desc='UNICODE', disable=(not progress_bar))
    for col in cols:
        for idx in df[col].index:
            # Unidecode if value is a string
            if isinstance(df.loc[idx,col], (basestring, unicode)):
                try:
                    # Try to encode to utf-8, but only if it is unicode
                    if isinstance(df.loc[idx,col], unicode):
                        df.loc[idx,col] = df.loc[idx,col].encode(encoding)
                    elif decode_if_errors:
                        df.loc[idx,col] = string_to_unicode(df.loc[idx,col]).encode(encoding)
                    else:
                        raise ValueError('Error at column "%s" index %s: not unicode!')
                except UnicodeDecodeError as exc:
                    # At worst, try unidecode
                    if skip_errors:
                        df.loc[idx,col] = _unidecode(df.loc[idx,col]).encode(encoding)
                    else:
                        print('Error at column "%s" index %s' % (col, str(idx)))
                        raise
            if progress_bar:
                pbar.update()
    # Restore index
    if idxbak:
        df.set_index(idxbak, inplace=True)
    return df

def df_literal_eval(x):
    """Evaluate each string cell of a DataFrame as if it was a Python object, and return the Python object"""
    try:
        # Try to evaluate using ast
        return(ast.literal_eval(x))
    except (SyntaxError, ValueError):
        try:
            # Else evaluate as a list without quotes
            if not ((x.startswith('[') or x.startswith('{')) and (x.endswith(']') or x.endswith('}'))):
                raise Exception()
            return re.split(',\s*u?', re.sub('[\[\]\{\}]', '', x))
            # TODO: implement a real parser using pyparser: https://stackoverflow.com/a/1894785
        except Exception as exc:
            # Else simply return the item as-is
            return x

def df_cols_lower(df_in, col='name'):
    """Find in a DataFrame any column matching the col argument in lowercase and rename all found columns to lowercase"""
    # Make a copy to avoid tampering the original
    df = df_in.copy()
    # Find and rename any column "Name" or "NAME" to lowercase "name"
    namecols = df.columns[[True if x.lower() == col.lower() else False for x in df.columns]]
    if len(namecols) > 0:
        df = df.rename(columns={x: x.lower() for x in namecols})
    return df

def date_fr2en(s):
    """Convert french month names into english so that dateutil.parse works"""
    if isinstance(s, basestring):
        s = s.lower()
        rep = {
            'jan\w+': 'jan',
            'fe\w+': 'feb',
            'mar\w+': 'march',
            'av\w+': 'april',
            'mai\w+': 'may',
            'juin\w+': 'june',
            'juil\w+': 'july',
            'ao\w+': 'august',
            'se\w+': 'september',
            'oc\w+': 'october',
            'no\w+': 'november',
            'de\w+': 'december',
        }
        for m, r in rep.items():
            s = re.sub(m, r, s)
    return s

def date_cleanchar(s):
    """Clean a date from any non useful character (else dateutil_parser will fail, eg with a '?')"""
    if isinstance(s, basestring):
        s = s.lower()
        res = re.findall('[\d/-:\s]+', s, re.UNICODE)
        if res:
            return '-'.join(res)
        else:
            return None
    else:
        return s

def date_clean(s):
    """Clean the provided string and parse as a date using dateutil.parser fuzzy matching (alternative to pd.to_datetime()). Should be used with df[col].apply(df_date_clean)."""
    if pd.isnull(s):
        return None
    else:
        # Clean non date characters (might choke the date parser)
        cleaned_date = date_fr2en(date_cleanchar(str(s))) # convert to str so that if we get a datetime object, we do not get an error
        if not cleaned_date:
            return None
        else:
            try:
                # First try an ISO date parsing, this is to circumvent bad decoding when month is in the middle, see: https://github.com/dateutil/dateutil/pull/340
                return dateutil_parser.isoparse(cleaned_date)
            except ValueError as exc:
                try:
                    # Try to parse with formatting with day first
                    return dateutil_parser.parse(cleaned_date, dayfirst=True, fuzzy=True)
                except ValueError as exc:
                    # If failed, try with year first
                    try:
                        m = re.search('\d+', s, re.UNICODE)
                        if len(m.group(0)) == 4:
                            return dateutil_parser.parse(cleaned_date, yearfirst=True, fuzzy=True)
                        else:
                            raise
                    except Exception as exc:
                        # Else we print an error but we pass (we just don't use this date)
                        print('Warning: Failed parsing this date: %s' % s)
                        return None

def df_date_clean(df_in, col):
    """Apply fuzzy date cleaning (and datetype conversion) to a dataframe's column"""
    # Make a copy to avoid tampering the original
    df = df_in.copy()
    df[col] = df[col].apply(date_clean).astype('datetime64')
    return df

def clean_integer_score(x):
    """Converts x from potentially a float or string into a clean integer, and replace NA and NP values with one string character"""
    try:
        x = str(int(float(x)))
    except Exception as exc:
        if isinstance(x, basestring):
            pass
        else:
            raise
    x = x.lower().strip()
    return 'A' if x == 'na (not assesible)' else 'P' if x == 'np (not performed)' else x

def df_subscores_concat(df, cols=None, col_out='subscore'):
    """Create CRS-R subscores summary (eg, S123456)"""
    if cols is None:
        raise ValueError('Must specify which columns we take the CRS-R subscores from!')
    if len(cols) != 6:
        raise ValueError('The number of columns specified does not correspond to CRS-R, we need 6 columns (6 categories of items)!')
    # We extract only the subscores in the correct order and we clean up the special values (not assesible NA and not performed NP) to replace by a single character (A and P respectively) and we replace NAN by X
    df_subscores = df[cols].fillna('X').applymap(clean_integer_score)
    # Make a copy to avoid side effects
    df2 = df.copy()
    # Then we concatenate all subscores in one string, prefixing 'S' before, and add the result as a new column
    df2[col_out] = df_subscores.apply(lambda x: concat_strings(x, 'S'), axis=1)
    # Return!
    return df2

def df_unify(df, cols, unicol):
    """Unify multiple columns by recursively filling blanks from the specified list of columns.
    cols is the list of columns from which to copy (the first will be the most copied, the last the least and only if other columns did not contain any info)
    unicol is the name of the new unified column"""
    # Make a copy to avoid side effects
    df2 = df.copy()
    # Copy the first column as reference
    df2[unicol] = df2[cols[0]]
    # Then each subsequent column will be used to fill the blank parts
    for col in cols[1:]:
        df2.loc[df2[unicol].isnull() | (df2[unicol] == ''), unicol] = df2[col]
    # Return the Dataframe with the new unified column
    return df2

def df_replace_nonnull(x, repmap, cleanup=False):
    if cleanup and isinstance(x, str):
        x = cleanup_name(replace_buggy_accents(x))
    if x in repmap:
        replacement = repmap[x]
        return replacement if replacement is not None else x
    else:
        return x

def df_translate(df, col, mapping, cleanup=False, partial=False):
    """Translate a list of strings on a Dataframe's column
    This can for example be used to simplify all the values to a reduced set, for more comprehensible graphs or easier data organization
    If partial=True, if a cell contains the matching pattern, it will be replaced by the target. Use an OrderedDict to take advantage of the sequential order (to make sure that the first pattern is replaced first, then the rest)."""
    df2 = df.copy()
    if not partial:
        # Exact match required
        df2[col] = df2[col].apply(lambda x: df_replace_nonnull(x, mapping, cleanup=cleanup))
    else:
        # Partial match OK
        for pattern, replacement in mapping.items():
            df2.loc[df2[col].str.contains(pattern), col] = replacement
    return df2

def filter_nan_str(x):
    """Filter 'nan' values as string from a list of strings"""
    if not isinstance(x, list):
        return x
    else:
        return [y for y in x if y.lower() != 'nan']

def df_filter_nan_str(df_col):
    """Filter all 'nan' values as strings from a Dataframe column containing lists"""
    return df_col.apply(df_literal_eval).apply(filter_nan_str).astype('str')


######################## DICOMS #############################

def generate_path_from_dicom_fields(output_dir, dcmdata, key_dicom_fields, cleanup_dicom_fields=True, placeholder_value='unknown'):
    pathparts = []
    # For each outer list elements (will be concatenated with a directory separator like '/')
    for dfields in key_dicom_fields:
        if not isinstance(dfields, list):
            dfields = [dfields]
        innerpathparts = []
        # For each inner list elements (will be concatenated with '_')
        for dfield in dfields:
            # Extract the dicom field's value
            if dfield in dcmdata:
                if isinstance(dfield, str):
                    # If string (a named field)
                    dcmfieldval = dcmdata[dcmdata.data_element(dfield).tag].value
                else:
                    # Else it's a coordinate field (no name, like (0010, 2020))
                    dcmfieldval = dcmdata[dfield].value
            else:
                dcmfieldval = placeholder_value
            # Cleanup the dicom field is enabled (this will replace accentuated characters, most english softwares do not support those)
            if cleanup_dicom_fields:
                dcmfieldval = cleanup_name(dcmfieldval)
            # Add the path parts to the list
            innerpathparts.append(dcmfieldval)
        # Concatenate the inner path parts and add to the outer path parts list
        pathparts.append('_'.join(innerpathparts))
    # Build the full path from the outer path parts list
    pathpartsassembled = os.path.join(*pathparts)
    # Replace all spaces by dashes (so that programs that do not support spaces well won't be bothered)
    pathpartsassembled = re.sub(r'\s+', r'-', pathpartsassembled, count=0)
    # Join with output dir to get final path
    finalpathdir = os.path.join(output_dir, pathpartsassembled)
    return finalpathdir

def recwalk_dcm(*args, **kwargs):
    """Recursive DICOM metadata reader, supporting zipfiles.
    Yields for each dicom file (whether normal or inside a zipfile) a dictionary filled with DICOM file metadata, path and zip handler if it is inside a zipfile.
    Comes with an integrated progress bar."""
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']
        del kwargs['verbose']
    else:
        verbose = False
    if 'nobar' in kwargs:
        nobar = kwargs['nobar']
        del kwargs['nobar']
    else:
        nobar = False
    if not 'filetype' in kwargs:
        kwargs['filetype'] = ['.dcm', '', '.zip']

    # Make list of filetypes for zipfile
    # Process no extension separately (because else endswith() will accept any extension if we supply '')
    noextflag = False
    filetypes = list(kwargs['filetype'])  # make a copy
    if '' in filetypes:
        filetypes.remove('')
        filetypes = tuple(filetypes)  # endswith() only supports tuples
        noextflag = True

    # Counting total number of files (to show a progress bar)
    filescount = 0
    if not nobar:
        for dirpath, filename in _tqdm(recwalk(*args, **kwargs), desc='PRECOMP', unit='files'):
            if not filename.endswith('.zip'):
                filescount +=1
            else:
                try:
                    zfilepath = os.path.join(dirpath, filename)
                    with zipfile.ZipFile(zfilepath, 'r') as zipfh:
                        zfilescount = sum(1 for item in zipfh.namelist() if not item.endswith('/'))
                    filescount += zfilescount
                except zipfile.BadZipfile as exc:
                    # If the zipfile is unreadable, just pass
                    if verbose:
                        print('Error: Bad zip file: %s' % os.path.join(dirpath, filename))
                    pass

    pbar = _tqdm(total=filescount, desc='REORG', unit='files', disable=nobar)
    for dirpath, filename in recwalk(*args, **kwargs):
        try:
            if not filename.endswith('.zip'):
                if filename.lower() == 'dicomdir':  # pass DICOMDIR files
                    continue
                try:
                    if verbose:
                        print('* Try to read fields from dicom file: %s' % os.path.join(dirpath, filename))
                    # Update progress bar
                    pbar.update()
                    # Read the dicom data in memory (via StringIO)
                    dcmdata = pydicom.read_file(os.path.join(dirpath, filename), stop_before_pixels=True, defer_size="512 KB", force=True)  # stop_before_pixels allow for faster processing since we do not read the full dicom data, and here we can use it because we do not modify the dicom, we only read it to extract the dicom patient name. defer_size avoids reading everything into memory, which workarounds issues with some malformatted fields that are too long (OverflowError: Python int too large to convert to C long)
                    yield {'data': dcmdata, 'dirpath': dirpath, 'filename': filename}
                except (InvalidDicomError, AttributeError, OverflowError) as exc:
                    pass
            else:
                try:
                    zfilepath = os.path.join(dirpath, filename)
                    with zipfile.ZipFile(zfilepath, 'r') as zipfh:
                        #zfolders = (item for item in zipfh.namelist() if item.endswith('/'))
                        zfiles = ( item for item in zipfh.infolist() if (not item.filename.endswith('/') and (item.filename.endswith(filetypes) or (noextflag and not '.' in item.filename))) )  # infolist() is better than namelist() because it will also work in case of duplicate filenames
                        for zfile in zfiles:
                            # Update progress bar
                            pbar.update()
                            # Need to extract because pydicom does not support not having seek() (and zipfile in-memory does not provide seek())
                            zf = zfile.filename
                            if zf.lower().endswith('dicomdir'):  # pass DICOMDIR files
                                continue
                            z = _StringIO(zipfh.read(zf)) # do not use .extract(), the path can be anything and it does not support unicode (so it can easily extract to the root instead of target folder!)
                            # Try to open the extracted dicom
                            try:
                                if verbose:
                                    print('* Try to decode dicom fields with zipfile member %s' % zf)
                                # Read the dicom data in memory (via StringIO)
                                dcmdata = pydicom.read_file(z, stop_before_pixels=True, defer_size="512 KB", force=True)  # stop_before_pixels allow for faster processing since we do not read the full dicom data, and here we can use it because we do not modify the dicom, we only read it to extract the dicom patient name. defer_size avoids reading everything into memory, which workarounds issues with some malformatted fields that are too long (OverflowError: Python int too large to convert to C long)
                                yield {'data': dcmdata, 'dirpath': dirpath, 'filename': filename, 'ziphandle': zipfh, 'zipfilemember': zfile}
                            except (InvalidDicomError, AttributeError, OverflowError) as exc:
                                pass
                            except IOError as exc:
                                if 'no tag to read' in str(exc).lower():
                                    pass
                                else:
                                    raise
                except zipfile.BadZipfile as exc:
                    # If the zipfile is unreadable, just pass
                    if verbose:
                        print('Error: Bad zip file: %s' % os.path.join(dirpath, filename))
                    pass
        except Exception as exc:
            print('ERROR: chocked on file %s' % os.path.join(dirpath, filename))
            import traceback
            print(traceback.format_exc())
            raise(exc)

def remove_if_exist(path):  # pragma: no cover
    """Delete a file or a directory recursively if it exists, else no exception is raised"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            return True
        elif os.path.isfile(path):
            os.remove(path)
            return True
    return False

def copy_any(src, dst, only_missing=False, symlink=False):  # pragma: no cover
    """Copy a file or a directory tree, deleting the destination before processing.
    If symlink, then the copy will only create symbolic links to the original files."""
    def real_copy(srcfile, dstfile):
        """Copy a file or a folder and keep stats"""
        shutil.copyfile(srcfile, dstfile)
        shutil.copystat(srcfile, dstfile)
    def symbolic_copy(srcfile, dstfile):
        """Create a symlink (symbolic/soft link) instead of a real copy"""
        os.symlink(srcfile, dstfile)

    # Delete destination folder/file if it exists
    if not only_missing:
        remove_if_exist(dst)
    # Continue only if source exists
    if os.path.exists(src):
        # If it's a folder, recursively copy its content
        if os.path.isdir(src):
            # If we copy everything, we already removed the destination folder, so we can just copy it all
            if not only_missing and not symlink:
                shutil.copytree(src, dst, symlinks=False, ignore=None)
            # Else we will check each file and add only new ones (present in source but absent from destination)
            # Also if we want to only symlink all files, shutil.copytree() does not support that, so we do it here
            else:
                for dirpath, filepath in recwalk(src):
                    srcfile = os.path.join(dirpath, filepath)
                    relpath = os.path.relpath(srcfile, src)
                    dstfile = os.path.join(dst, relpath)
                    if not only_missing or not os.path.exists(dstfile):  # only_missing -> dstfile must not exist
                        create_dir_if_not_exist(os.path.dirname(dstfile))
                        if symlink:
                            symbolic_copy(srcfile, dstfile)
                        else:
                            real_copy(srcfile, dstfile)
            return True
        # Else it is a single file, copy the file
        elif os.path.isfile(src) and (not only_missing or not os.path.exists(dst)):
            create_dir_if_not_exist(os.path.dirname(dst))
            if symlink:
                symbolic_copy(src, dst)
            else:
                real_copy(src, dst)
            return True
    return False

def get_list_of_folders(rootpath):
    return [item for item in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, item))]

def get_list_of_zip(rootpath):
    return [item for item in os.listdir(rootpath) if os.path.isfile(os.path.join(rootpath, item)) and item.endswith('.zip')]
