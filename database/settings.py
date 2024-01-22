import os

class Personal:
    db_name = os.getenv('DB_NAME', 'default_db_name')
    host = os.getenv('DB_HOST', 'default_host')
    password = os.getenv('DB_PASSWORD', 'default_password')
    user = os.getenv('DB_USER', 'default_user')
