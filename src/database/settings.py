import os

class Personal:
    db_name = os.getenv('DB_NAME', 'name')
    host = os.getenv('DB_HOST', 'host')
    password = os.getenv('DB_PASSWORD', 'password')
    user = os.getenv('DB_USER', 'user')
    dataset_path = os.getenv('DATASET_PATH', 'dataset')
