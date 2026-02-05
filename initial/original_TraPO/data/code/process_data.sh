
#!/bin/bash
set -e  

mkdir ../ID_domain

python ./get_data.py ../openr1.parquet ../ID_domain/openr1_1024_fixed.parquet --start 0 --rows 1024
python ./get_data.py ../openr1.parquet ../ID_domain/openr1_4096_fixed.parquet --start 0 --rows 4096
python ./get_data.py ../openr1.parquet ../ID_domain/openr1_1024_1024_fixed.parquet --start 1024 --rows 1024
python ./get_data.py ../openr1.parquet ../ID_domain/openr1_1024_3072_fixed.parquet --start 1024 --rows 3072
python ./get_data.py ../openr1.parquet ../ID_domain/openr1_4096_12288_fixed.parquet --start 4096 --rows 12288

mkdir ../OOD_domain

python ./get_data.py ../valid.mmlu_pro.parquet ../OOD_domain/train.mmlu_pro-1024_fixed.parquet --start 0 --rows 1024


echo "All data processed successfully!"