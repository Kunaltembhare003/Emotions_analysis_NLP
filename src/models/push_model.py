import boto3
import shutil
import pathlib
from botocore.exceptions import NoCredentialsError, ClientError

## push model to s3 bucket
def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    # Create an S3 client
    s3 = boto3.client('s3')
     # Check if the bucket exists
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"The bucket '{bucket_name}' does not exist.")
            return
        else:
            raise
    # upload the file
    try:
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File uploaded successfully to {bucket_name}/{s3_file_path}")
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")

# Example usage
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
local_model_path = home_dir.as_posix() + '/models/model.h5'
s3_bucket_name = 'sentiment-analysis-bucket-003'
s3_file_path = 'models/model.h5' 

upload_to_s3(local_model_path, s3_bucket_name, s3_file_path)
shutil.copy(local_model_path, home_dir.as_posix()+'/model.h5')

## push tokennizer.json file to s3 bucket
def upload_to_s3_token(local_file_path, bucket_name, s3_file_path):
    # Create an S3 client
    s3 = boto3.client('s3')

    try:
        # Upload the file
        s3.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"File uploaded successfully to {bucket_name}/{s3_file_path}")
    except FileNotFoundError:
        print(f"The file {local_file_path} was not found.")
    except NoCredentialsError:
        print("Credentials not available.")

# Example usage


local_token_path = home_dir.as_posix() + '/data/interim/tokenizer.json'
s3_bucket_name = 'sentiment-analysis-bucket-003'
s3_token_path = 'models/tokenizer.json'

upload_to_s3(local_token_path , s3_bucket_name, s3_token_path)
shutil.copy(local_token_path , home_dir.as_posix()+'/tokenizer.json')
