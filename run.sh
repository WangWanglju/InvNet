#!/etc/bash

# echo "TRAIN IP"
# python Preprocess.py -d 'IP'
# python CreateDataset.py -d 'IP' -r 0.1
# python train.py -d 'IP'
# python test.py -d 'IP'
# rm -rf pickle valset trainset

echo "TRAIN PU"
python Preprocess.py -d 'PU'
python CreateDataset.py -d 'PU' -r 0.025
python train.py -d 'PU'
python test.py -d 'PU'
python test_grade.py -d 'PU'

rm -rf pickle valset trainset

# echo "TRAIN PA"
# python Preprocess.py -d 'PA'
# python CreateDataset.py -d 'PA' -r 0.01
# python train.py -d 'PA'
# python test.py -d 'PA'
# rm -rf pickle valset trainset

echo "TRAIN SA"
python Preprocess.py -d 'SA'
python CreateDataset.py -d 'SA' -r 0.02
python train.py -d 'SA'
python test.py -d 'SA'
python test_grade.py -d 'SA'
rm -rf pickle valset trainset
