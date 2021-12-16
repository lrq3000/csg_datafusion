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

1. extract_fields_from_reports
2. optional: dicoms_reorganizer (to ensure there is no duplicates, else the dicom_to_nifti conversion and modular_reorganizer steps might become a headache + it will speed up calculations by removing duplicates)
3. dicoms_extract
4. fmp_db_cleaner
5. sarah_db_cleaner
6. stats_analysis_fmp_dicoms_db
7. stats_analysis_fmp_dicoms_db_acute
8. optional: ecg_db_generator
9. db_merger (repeat this to merge any database you want, particularly those that were cleaned by previous steps)
10. finaldbunification
11. dicoms_to_nifti
12. optional: manual preprocess your data (eg, using [reorientation_registration_helper.py](https://github.com/lrq3000/csg_mri_pipelines/blob/master/utils/pathmatcher/reorientation_registration_helper.py) for fmri data)
13. modular_reorganizer
14. db_merger again to merge in the post-nifti conversion additional infos (movement parameters, any quality assurance demographics file you made, etc)

Bonus: dbconcat allows to concatenate (ie, append) 2 csv databases together, which circumvents the buggy concatenation in Microsoft Excel (which can lose the separator if the csv file is too long).

Tip: when you convert an input database from .xls or .xlsx (Excel) to csv format to be able to use these notebooks, please make sure to convert with a UTF-8 encoding and a CSV format that can store the BOM (eg, in Office Excel, choose "CSV (Comma delimited)" format when saving, and not "CSV (MS-DOS)" nor "CSV (Macintosh)").

Tip2: the CSV files must use the semicolon ";" as a separator (and not the comma ",").

This whole process might seem overwhelming, but a BIG advantage lies in its modularity: if you need to update a specific part (let's say add more calculations on diagnoses dates, or rewrite how final diagnoses are computed, or add a new demographics file source, etc), you can do so without having to recompute everything from the start, you mostly need to update the appropriate notebook or create a new one to add the data you want if it's a calculation (else you don't need to), and then use db_merger to unify into a single database file. Thus, this project was built in such a way as to allow selective updating of precise parts of the pipeline and of the final database, without having to recompute everything, which saves a substantial amount of time (both in calculation and in development).

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

Other necessary libraries such as pydicom (git revision 356a51ab4bc54fd18950041ebc44dbfa1a425a10) are included with some modifications to fix bugs and issues.

## License
MIT License (see file LICENSE) for the notebooks and csg_fileutil_libs/aux_funcs.py, the rest in csg_fileutil_libs subfolders have their own licenses, compatible with the MIT.

## Similar projects

This is an addendum on December 2021. It appears what this set of notebooks does is more formally called deduplication by record linkage (comparison of fields between the same on between documents) and fuzzy matching (name deduplication by a modified Hamming distance metric).

There are now other projects that are available to achieve a similar goal, including merging different files together, although they rely on a different method, leveraging machine learning, which confers them more flexibility at the expense of more calibration required beforehand and a higher risk of uncontrolled false positives/negatives:

* [Dedupe Python package](https://github.com/dedupeio/dedupe). An already trained machine learning model is available on dedupe.io for a fee, a [python module](https://github.com/Lyonk71/pandas-dedupe) exists to use this online model. See this [talk](https://www.youtube.com/watch?v=McsTWXeURhA) for more details on how deduplication by record linkage via machine learning works.
