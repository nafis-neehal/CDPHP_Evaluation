import boto3
from botocore.exceptions import NoCredentialsError, ClientError, ConnectionError
import pandas as pd
import pickle

#--------------------------------------------------
#add the access keys here (Jason's S3 - RPI Org)
ACCESS_KEY = 'AKIA2W36TZO4GGFOD2GS'
SECRET_KEY = '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'
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

    
def load_from_aws(bucket, directory, file):
    s3 = boto3.resource('s3')
    try:
        print("Loading from Pickle file")
        my_object = pickle.loads(s3.Object(bucket_name= bucket, key=directory+file).get()['Body'].read())
        
    except:
        try:
            print("Pickle loading failed, loading CSV")
            f_name = file

            a = s3.Object(bucket_name = bucket, key = directory+file).download_file(
                file)
            
            my_object = pd.read_csv(file)
        
        except ClientError:
            print("Client Error")
            return False
        except ConnectionError:
            print("Connection Error")
            return False
        except FileNotFoundError:
            print("The file was not found")
            return False
    
    #returns dataframe
    return my_object
    
def download_from_aws(bucket, s3_file, local_file):
    
    s3 = boto3.client('s3')

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
