aws s3 cp s3://emr-demo-2024/code/AWS_EMR_program.py ./

spark-submit --executor-memory 1g AWS_EMR_program.py
