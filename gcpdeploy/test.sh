#!/bin/bash

# Step 1: Create a VM instance
gcloud compute instances create imdb-sentiment-analysis-vm \
    --zone=us-central1-a \
    --image=debian-10-buster-v20201014 \
    --image-project=debian-cloud \
    --machine-type=e2-micro \
    --boot-disk-size=10GB \
    --tags=http-server,https-server \
    --metadata startup-script='#!/bin/bash
                                  # Update package lists
                                  sudo apt-get update

                                  # Install required packages
                                  sudo apt-get install -y python3 python3-pip python3.11-venv git

                                  # Create directory and set permissions
                                  sudo mkdir -p /home/bank-marketing-prediction-mlops
                                  sudo chmod -R 777 /home/bank-marketing-prediction-mlops

                                  # Change to the project directory
                                  cd /home/bank-marketing-prediction-mlops

                                  # Clone the repository
                                  git clone https://github.com/Saitej0211/Bank_Marketing_Prediction_MLops.git

                                  # Navigate to the gcpdeploy directory
                                  cd Bank_Marketing_Prediction_MLops/gcpdeploy

                                  # Create virtual environment
                                  python3 -m venv env

                                  # Activate virtual environment
                                  . env/bin/activate

                                  # Install requirements (removed --user flag)
                                  pip install -r requirements.txt

                                  echo "Setup completed successfully"' \
    --async

# Wait for the VM to be up and running (can be done in your script or manually)
echo "VM created and environment is being set up..."

# Step 2: Create a snapshot of the VM
gcloud compute instances stop imdb-sentiment-analysis-vm --zone=us-central1-a
gcloud compute disks snapshot imdb-sentiment-analysis-vm --snapshot-names=imdb-sentiment-analysis-vm-snapshot --zone=us-central1-a

# Step 3: Create a custom VM image
gcloud compute images create imdb-sentiment-analysis-image \
    --source-disk=imdb-sentiment-analysis-vm --source-disk-zone=us-central1-a

# Step 4: Create VPC Network and Firewall Rules
gcloud compute networks create imdb-sentiment-analysis-vpc --subnet-mode=custom
gcloud compute networks subnets create imdb-sentiment-analysis-vpc-subnet \
    --network=imdb-sentiment-analysis-vpc \
    --region=us-central1 \
    --range=10.0.0.0/24
gcloud compute firewall-rules create imdb-sentiment-analysis-vpc-allow-custom \
    --network=imdb-sentiment-analysis-vpc \
    --allow=tcp:22,tcp:80,tcp:8000

# Step 5: Create an Instance Template and Managed Instance Group (MIG)
cat <<EOF > startup-script.sh
#!/bin/bash

# Navigate to the project directory and gcpdeploy folder
cd /home/bank-marketing-prediction-mlops/Bank_Marketing_Prediction_MLops/gcpdeploy

# Activate the virtual environment
. env/bin/activate

# Run the application in the background using nohup
nohup python3 app.py & 
EOF

gcloud compute instance-templates create imdb-sentiment-analysis-template \
    --machine-type=e2-micro \
    --image=imdb-sentiment-analysis-image \
    --metadata-from-file=startup-script=startup-script.sh \
    --network=imdb-sentiment-analysis-vpc \
    --subnet=imdb-sentiment-analysis-vpc-subnet \
    --region=us-central1  # Add the region flag here

gcloud compute instance-groups managed create imdb-mig \
    --base-instance-name=imdb-instance \
    --template=imdb-sentiment-analysis-template \
    --size=1 \
    --zone=us-central1-c

# Step 6: Set Up Load Balancer
gcloud compute health-checks create http imdb-health-check \
    --port 8000 \
    --request-path /health

gcloud compute backend-services create imdb-backend-service \
    --protocol=HTTP \
    --health-checks=imdb-health-check \
    --global

gcloud compute backend-services add-backend imdb-backend-service \
    --instance-group=imdb-mig --global

gcloud compute url-maps create imdb-url-map \
    --default-service=imdb-backend-service

gcloud compute target-http-proxies create imdb-http-proxy \
    --url-map=imdb-url-map

gcloud compute forwarding-rules create imdb-http-forwarding-rule \
    --global \
    --target-http-proxy=imdb-http-proxy \
    --ports=80

# Step 7: Final Confirmation
echo "VM, Snapshot, Custom Image, VPC, Firewall, MIG, and Load Balancer have been set up successfully!"
