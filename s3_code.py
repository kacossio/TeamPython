import os
import boto3

#********READ THROUGH COMMENTS BEFORE RUNNING******

#set working directory if necessary
#path=r'/Users/nins/Desktop'
#os.chdir(path)


#Add in your keys you copied from the AWS CLI. They expire after you logout so you will need to update them everytime
session = boto3.Session(
        aws_access_key_id='AWS_ACCESS_KEY_ID',
        aws_secret_access_key='AWS_SECRET_ACCESS_KEY',
        aws_session_token='AWS_SESSION_TOKEN',
        region_name='us-east-1'
        )

s3 = session.resource('s3')


#Choose a bucket
Ruijia_bucket='daen690twitter'
Nina_bucket='teampython690'   #Only Ruijia has access to mine


#upload test 
#Instructions:  Choose a small file to test. If it's in your working directory you just need the file name
#When ready to run uncomment code. Don't run multiple times as it will continue to upload.

#example for Ruijia's bucket
s3.meta.client.upload_file(Filename= 'INSERT FILE NAME', Bucket=Ruijia_bucket)


#download test
## Instructions: Change the location_path to where you want the file saved on your computer
#s3.meta.client.download_file('bucket', 'name of file in bucket', 'location_path')

#example for Nina's bucket
s3.meta.client.download_file(Nina_bucket, 'test', 'INSERT PATH/practice.txt')


#example for Ruijia's bucket
s3.meta.client.download_file(Ruijia_bucket, 'DAEN.json', 'INSERT PATH/practice.json')


#list files in bucket. You can't see it in your S3 dashboard
bucket = s3.Bucket(Ruijia_bucket)

for obj in bucket.objects.all():
    print(obj.key)
    
    

