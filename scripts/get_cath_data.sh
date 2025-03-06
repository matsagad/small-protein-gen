#!/bin/bash

CLEAN=true
CATH_ENDPOINT="ftp://orengoftp.biochem.ucl.ac.uk/cath/releases/latest-release/non-redundant-data-sets/cath-dataset-nonredundant-S40.pdb.tgz"
DATA_PATH="data"
UNZIP_FOLDER=$DATA_PATH/dompdb
FILTERED_FOLDER=$DATA_PATH/dompdb-filtered
TAR_FILE=$DATA_PATH/$(basename $CATH_ENDPOINT)

if [ ! -d $DATA_PATH ]; then mkdir $DATA_PATH; fi

if [ -d $UNZIP_FOLDER ]; then echo "Data folder already exists."; exit 0; fi

# Download and extract tar file.
if [ ! -f $TAR_FILE ]; then echo "Downloading tar file..."; curl -o $TAR_FILE $CATH_ENDPOINT; else echo "Existing tar file found."; fi
echo "Extracting tar file..."; tar -xzf $TAR_FILE --directory $DATA_PATH && $CLEAN && rm $TAR_FILE
for FILE in $UNZIP_FOLDER/*; do mv $FILE $FILE.pdb; done

# Filter proteins to have length between 40 and 100.
python3 scripts/filter_cath_data.py --data_folder $UNZIP_FOLDER --out_folder $FILTERED_FOLDER