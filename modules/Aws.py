import boto3
from botocore.exceptions import NoCredentialsError, ClientError, ConnectionError

#--------------------------------------------------
#add the access keys here (Jason's S3 - RPI Org)
#ACCESS_KEY = 
#SECRET_KEY = 
#--------------------------------------------------

def upload_to_aws(local_file, bucket, s3_file):

    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.upload_file(local_file, bucket, s3_file)
        print("Upload Successful")
        return True
    except FileNotFoundError:
        print("The file was not found")
        return False
    except ClientError:
        print("Client Error")
        return False
    except ConnectionError:
        print("Connection Error")
        return False
    except NoCredentialsError:
        print("Credentials not available")
        return False
    
def download_from_aws(bucket, s3_file, local_file):
    
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY,
                      aws_secret_access_key=SECRET_KEY)

    try:
        s3.download_file(bucket, s3_file, local_file)
        print("Download Successful")
        return True
    except ClientError:
        print("Client Error")
        return False
    except ConnectionError:
        print("Connection Error")
        return False
    except FileNotFoundError:
        print("The file was not found")
        return False
