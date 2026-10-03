import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DATABASE_URL = (
    "postgresql://neondb_owner:npg_EH5qZ6ABMjJs@ep-twilight-tree-b5ftwb7x-pooler.c-7.us-east-2.aws.neon.tech/"
    "neondb?sslmode=require&channel_binding=require"
)

DATABASE_URL = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
