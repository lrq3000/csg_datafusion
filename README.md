# CSG Data Fusion by Stephen Larroque, 2016-2019

This toolset provide several tools to unify databases from multiple sources (data fusion).

Here are the descriptions of a few tools included (not exhaustive - there are descriptions and readme included at the top of each script):
* PDF/DOC reports extractor to csv database.
* CSV databases comparison and merge (to merge reports database with fmp database, but can be used with any two csv files as long as they have two columns: name and final_diagnosis).
* Dicom and csv anonymizer (Note: this is an old version, the latest anonymizer is now standalone in its own script and with a GUI! There is another folder or it is also on github: lrq3000/csg_dicoms_anonymizer).

To use them, you need to pip install pandas (if you have installed Anaconda, you don't need to install pandas).

For the reports extractor specifically, you will also need to pip install textract. If you want to use OCR (to extract a few PDF documents that cannot be extracted otherwise), you will need also install tesseract ocr v3 (this is not a Python library, so you will have to grab the installer for your platform). Tesseract ocr v4 might also work but at the time of this writing it is still in alpha so it was not possible to test.

You might need additional libraries depending on the script, but the author tried his best to maintain the number of dependencies to the minimum (to avoid complicating usage and scripts breaking when the dependencies are updated).
