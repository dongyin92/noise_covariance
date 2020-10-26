set -e
set -x

virtualenv -p python3 .
source ./bin/activate

pip install tensorflow
pip install tensorflow-datasets
pip install -r noise_covariance/requirements.txt
python -m noise_covariance.train
