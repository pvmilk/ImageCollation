#!/bin/bash
#
# HOWTO CALL:
# 	$ source PATH_TO_FILE/prepare.sh
#


BASEDIR=$(dirname "${BASH_SOURCE[0]}")
echo $BASEDIR


#
# Reading many files from google drive is slow
# - Choice 1: Extract to local file
# - Choice 2: Preprocess and store a large file (h5, lmdb) in google drive
#
# REF: https://medium.com/@selvam85/how-to-work-with-large-training-dataset-in-google-colab-platform-c3499fc10c24
#

PATH_DATASET='/content/img-collation/'

# if dataset not existed
if [[ ! -d "$PATH_DATASET" ]]
then
	#
	# UNZIP DataSet if not exist
	#
	PATH_DATAZIP='/content/drive/MyDrive/data/img-collation/img-collation.tar.xz'
	mkdir -p $PATH_DATASET
	tar -xvf $PATH_DATAZIP -C $PATH_DATASET
fi

#
# Install requirements
#
# If TPU, also run another command.

pip3 uninstall -r $BASEDIR/norequirements.txt -y
pip3 install -r $BASEDIR/requirements.txt
# pip3 install -r $BASEDIR/requirements-tpu.txt

export PATH_DATASET
