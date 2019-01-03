# CSG Data Fusion by Stephen Karl Larroque, 2016-2019

This toolset provide several tools to unify databases from multiple sources (data fusion).

Here are the descriptions of a few tools included (not exhaustive - there are descriptions and readme included at the top of each script):
* PDF/DOC reports extractor to csv database.
* CSV databases comparison and merge (to merge reports database with fmp database, but can be used with any two csv files as long as they have two columns: name and final_diagnosis).
* Dicom and csv anonymizer (Note: this is an old version, the latest anonymizer is now standalone in its own script and with a GUI! There is another folder or it is also on github: lrq3000/csg_dicoms_anonymizer).

To use them, you need to pip install pandas (if you have installed Anaconda, you don't need to install pandas).

For the reports extractor specifically, you will also need to pip install textract. If you want to use OCR (to extract a few PDF documents that cannot be extracted otherwise), you will need also install tesseract ocr v3 (this is not a Python library, so you will have to grab the installer for your platform). Tesseract ocr v4 might also work but at the time of this writing it is still in alpha so it was not possible to test.

You might need additional libraries depending on the script, but the author tried his best to maintain the number of dependencies to the minimum (to avoid complicating usage and scripts breaking when the dependencies are updated).

## Usage and steps order
Please place all original databases in a folder "databases_original". After running each step, please move all generated databases into a folder "databases_output".

Here is the advised steps order for generating a clean unified database from Coma Science Group multiple databases:

1- extract_fields_from_reports
2- dicoms_extract
3- fmp_db_cleaner
4- sarah_db_cleaner
5- stats_analysis_fmp_dicoms_db
6- stats_analysis_fmp_dicoms_db_acute
7- db_merger (repeat this to merge any database you want, particularly those that were cleaned by previous steps)
8- optional: ecg_db_generator

Tip: when you convert an input database from .xls or .xlsx (Excel) to csv format to be able to use these notebooks, please make sure to convert with a UTF-8 encoding and a CSV format that can store the BOM (eg, in Office Excel, choose "CSV (Comma delimited)" format when saving, and not "CSV (MS-DOS)" nor "CSV (Macintosh)").

Tip2: the CSV files must use the semicolon ";" as a separator (and not the comma ",").

## Requirements
For all notebooks:
* Anaconda 64bits Python >= 2.7.15
* pandas 0.23.4

For reports extractor notebook:
* tesseract-ocr 3.05.02-20180621 (must be added to the PATH)
* textract (via pip install)
* chardet >= 3.0.4 (via pip install, at least v3 for better recognition of latin-1 == ISO-8859-1)
* antiword 0.37 (so that C:\antiword\antiword.exe or ~/antiword/antiword exists)
* pypdf2 (via pip install)
* pdftoppm via Xpdf tools v4.00 http://www.xpdfreader.com/download.html (needs to be added to the PATH)
* pdftotext via Xpdf tools v4.00
* optionally: pdfminer 20140328 or pdfminer.six (via pip install, and associate .py files with python binary, for Windows do the following in a terminal with Admin power: ftype pyfile="C:\Users\Your_Username\Anaconda2\python.exe" "%1"; assoc .py=pyfile )

Other necessary libraries such as pydicom are included with some modifications to fix bugs and issues.

## License
MIT License (see file LICENSE) for the notebooks and csg_fileutil_libs/aux_funcs.py, the rest in csg_fileutil_libs subfolders have their own licenses, compatible with the MIT.
