export OUTPUT_DIR='../data/dataset512'
mkdir siim-acr 
cd siim-acr 
mkdir data 
cd data
kaggle datasets download -d seesee/siim-train-test
unzip siim-train-test.zip 
mv siim/* . 
rmdir siim
mkdir ../src/ 
cd ../src
git clone https://github.com/sneddy/pneumothorax-segmentation
python pneumothorax-segmentation/unet_pipeline/utils/prepare_png.py -train_path ../data/dicom-images-train/ -test_path ../data/dicom-images-test/ -out_path $OUTPUT_DIR -img_size 512 -rle_path ../data/train-rle.csv