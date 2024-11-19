sudo apt-get update
sudo apt-get install -y python3 python3-pip python3.11-venv git
sudo mkdir -p /home/bank-marketing-prediction-mlops
sudo chmod -R 777 /home/bank-marketing-prediction-mlops
cd /home/bank-marketing-prediction-mlops
git clone https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git
cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_Mlops/gcpdeploy
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt