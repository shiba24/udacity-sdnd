scp -i ~/.ssh/aws-machine-learning-g.pem ~/work/udacity-sdnd/data/behavioral-cloning-data/driving_log.csv ubuntu@ec2-34-228-69-36.compute-1.amazonaws.com:/home/ubuntu/udacity-sdnd/data
scp -i ~/.ssh/aws-machine-learning-g.pem ~/work/udacity-sdnd/data/behavioral-cloning-data/driving_log_additional.csv ubuntu@ec2-34-228-69-36.compute-1.amazonaws.com:/home/ubuntu/udacity-sdnd/data

scp -i ~/.ssh/aws-machine-learning-g.pem ~/work/udacity-sdnd/data/behavioral-cloning-data/IMG.zip ubuntu@ec2-34-228-69-36.compute-1.amazonaws.com:/home/ubuntu/udacity-sdnd/data/behavioral-cloning-data/
scp -i ~/.ssh/aws-machine-learning-g.pem ~/work/udacity-sdnd/data/behavioral-cloning-data/data/IMG.zip ubuntu@ec2-34-228-69-36.compute-1.amazonaws.com:/home/ubuntu/udacity-sdnd/data/behavioral-cloning-data/data/


scp -i ~/.ssh/aws-machine-learning-g.pem ubuntu@ec2-34-228-69-36.compute-1.amazonaws.com:/home/ubuntu/udacity-sdnd/CarND-Behavioral-Cloning-P3-master/model.h5 ~/work/udacity-sdnd/CarND-Behavioral-Cloning-P3-master/
