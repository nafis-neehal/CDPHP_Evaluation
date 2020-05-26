import boto3
from botocore.exceptions import NoCredentialsError

'''
bucketname=‘cdphp-rpi’
directory = data/predictions
directory = data/referrals
directory = cdphp-rpi/results
'''

bucket= 'cdphp-rpi'

prediction_dir = './data/predictions/'
referral_dir = './data/referrals/'
result_dir = './results/'

local_file_name = 'config.yaml'
s3_file_name = 'config_s3.yaml'

#ACCESS_KEY = <add_key>
#SECRET_KEY = <add_secret_key>

s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

def upload_to_aws(s3, local_file, bucket, s3_file):

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
def download_from_aws(s3, bucket, s3_file, local_file):
    
    try:
        s3.download_file(bucket, s3_file, local_file)
        print("Download Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False

uploaded = upload_to_aws(s3, local_file_name, bucket, result_dir + s3_file_name)
downloaded = download_from_aws(s3, bucket, result_dir + s3_file_name, s3_file_name)

#show objects in S3 bucket with keys with certain prefix
response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix ='./data/',
        MaxKeys=100 )

print(response)