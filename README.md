# CSG Data Fusion by Stephen Karl Larroque, 2016-2019

This toolset provide several tools to unify databases from multiple sources (data fusion).

What this toolset can do:
* **Data extraction** from multiple unformatted and formatted heterogenous data sources (pdf, excel, doc, docx, MRI DICOMs).
* **Data cleaning**, such as unreadable characters and redundant fields, and detection and removal of impossible data such as [impossible or improbable CRS-R scores](https://pubmed.ncbi.nlm.nih.gov/26944708/).
* **Data disambiguation and deduplication** using **fuzzy name matching** with custom distance metrics.
* **Databases realignment** and joining using relational databases rules (the toolset assumes only two fields are present: name and final_diagnosis, all other fields are dynamically accounted for). This is similar to the rationale that motivated the [tidy data paper by Wickham](https://doi.org/10.18637/jss.v059.i10), although the author did not know of it at the time, so this toolset likely is tidy compatible, accepts untidy data as input and then outputs tidy-like data.
* A Dicoms to Nifti batch convertor, using external tools. This allows to convert in a standardardized layout, then you can do manual preprocessing once and for all on the whole NIFTI files database, and finally you can use the data selector to extract sub-populations anytime you want later, without having to redo the manual preprocessing steps.
* **Data selector** with freeform rules, similarly to SQL in relational databases but using dataframes and Python.

## Usage and steps order
Please place all original databases in a folder "databases_original". After running each step, please move all generated databases into a folder "databases_output".

Here is the advised steps order for generating a clean unified database from Coma Science Group multiple databases, organized by the type of step:

Inputs extractor and tidying:
1. extract_fields_from_reports. Extracts a CSV database out of PDF files. The rules are very specific to our own set of PDFs, but the base functions to loop read digitalized and non-digitalized (ie, via OCR) PDF files can easily be re-used.
2. optional: dicoms_reorganizer (to ensure there is no duplicates, else the dicom_to_nifti conversion and modular_reorganizer steps might become a headache + it will speed up calculations by removing duplicates)
3. dicoms_extract. Extracts a CSV database of meta-data from DICOMs (ie, names, date of birth, sessions names).
4. fmp_db_cleaner. FileMakerPro database cleaner, already exported as a CSV file.
5. sarah_db_cleaner. A specific excel file cleaner.
7. stats_analysis_fmp_dicoms_db. Statistical analyses on the above databases, and also some data cleaning.
8. stats_analysis_fmp_dicoms_db_acute. Statistical analyses on the above databases, and also some data cleaning.
9. optional: ecg_db_generator. Extracts a database of meta-data from ECG filenames.

Demographics databases realignment and merging, with data disambiguation and deduplication using fuzzy and exact matchings:

10. db_merger (repeat this to merge any database you want, particularly those that were cleaned by previous steps). Databases realignment and joining with fuzzy matching. It can be used with any two csv files as long as they have two columns: name and final_diagnosis.
11. finaldbunification. Final disambiguation and clean-up operations.

Neuroimaging data conversion and relational-like data selector:

12. dicoms_to_nifti. DICOMs to NIFTI batch convertor using dcm2niix, while generating a database of meta-data such as files location (crucial for the data selector to work).
13. optional: manual preprocess your data (eg, using [reorientation_registration_helper.py](https://github.com/lrq3000/csg_mri_pipelines/blob/master/utils/pathmatcher/reorientation_registration_helper.py) for fmri data). Then, you can use the data selector below to select your already preprocessed data to extract subpopulations with the rules you want. Before this tool, we used to redo the preprocessing every time we extracted a subpopulation from raw data, since DICOMs cannot be reoriented, and there was no single data warehouse, hence original data had to be used as the "source of groundtruth".
14. modular_reorganizer. This is the data selector, you can make SQL-like requests on dataframes using Python, and it will essentially extract dicoms or nifti files. This generates clean self-contained subdatabases, with everything needed: the subset of demographics data and the neuroimaging data.
15. db_merger again to merge in the post-nifti conversion additional infos (movement parameters, any quality assurance demographics file you made, etc)

Bonus: dbconcat allows to concatenate (ie, append) 2 csv databases together, which circumvents the buggy concatenation in Microsoft Excel (which can lose separator characters if the csv file is too long).

Tip: when you convert an input database from .xls or .xlsx (Excel) to csv format to be able to use these notebooks, please make sure to convert with a UTF-8 encoding and a CSV format that can store the BOM (eg, in Office Excel, choose "CSV (Comma delimited)" format when saving, and not "CSV (MS-DOS)" nor "CSV (Macintosh)").

Tip2: CSV files must use the semicolon ";" as a separator (and not the comma ",").

This whole process might seem overwhelming, but a BIG advantage lies in its modularity: if you need to update a specific part (let's say add more calculations on diagnoses dates, or rewrite how final diagnoses are computed, or add a new demographics file source, etc), you can do so without having to recompute everything from the start, you mostly need to update the appropriate notebook or create a new one to add the data you want if it's a calculation (else you don't need to), and then use db_merger to unify into a single database file. Thus, this project was built in such a way as to allow selective updating of precise parts of the pipeline and of the final database, without having to recompute everything, which saves a substantial amount of time (both in calculation and in development).

Note: a DICOM and CSV anonymizer is also present, but this is an old version, the latest anonymizer is now standalone in its own script and with a GUI: lrq3000/csg_dicoms_anonymizer

## Requirements
To use the toolset, you need to pip install pandas (if you have installed Anaconda, you don't need to install pandas).

For the reports extractor specifically, you will also need to pip install textract. If you want to use OCR (to extract a few PDF documents that cannot be extracted otherwise), you will need also install tesseract ocr v3 (this is not a Python library, so you will have to grab the installer for your platform). Tesseract ocr v4 might also work but at the time of this writing it is still in alpha so it was not possible to test.

You might need additional libraries depending on the script, but the author tried his best to maintain the number of dependencies to the minimum (to avoid complicating usage and scripts breaking when the dependencies are updated).

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
