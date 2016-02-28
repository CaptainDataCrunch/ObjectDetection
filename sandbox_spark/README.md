### Instructions on Running on AWS [EMR](https://us-west-2.console.aws.amazon.com/elasticmapreduce)

Latest version of implementing spark is in spark_test3.py. Currently, we are using Spark to output the integral images dictionary. 
In order to run the file, you would need to run pickle_grey_images.py to get a json data of gray images. A sample is provided in test.json.

Run these command to get the required files onto the EMR. Note (Don't forget the colon at the end):

    scp -i ~/.ssh/<YOUR PEM KEY>.pem spark_test3.py hadoop@ec2-xxxxxxxxx.us-west-2.compute.amazonaws.com: 
    scp -i ~/.ssh/<YOUR PEM KEY>.pem adaboost* hadoop@ec2-xxxxxxxxx.us-west-2.compute.amazonaws.com: 
    scp -i ~/.ssh/<YOUR PEM KEY>.pem test.txt hadoop@ec2-xxxxxxxxx.us-west-2.compute.amazonaws.com: 

Now ssh onto your cluster: 

    ssh -i ~/.ssh/<YOUR PEM KEY>.pem hadoop@ec2-xxxxxxxxx.us-west-2.compute.amazonaws.com

and run the file:

    pyspark spark_test3.py

