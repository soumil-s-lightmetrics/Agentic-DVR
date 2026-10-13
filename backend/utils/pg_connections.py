import os
import psycopg2
import psycopg2.pool
from logger import debug_logger

debug_logger = debug_logger()

if (
    os.environ.get("LLM_POSTGRES_DATABASENAME") is None
    or os.environ.get("LLM_POSTGRES_DATABASENAME") == ""
):
    raise Exception("Invalid LLM_POSTGRES_DATABASENAME in env")
databasename = os.environ["LLM_POSTGRES_DATABASENAME"]
if (
    os.environ.get("LLM_POSTGRES_USERNAME") is None
    or os.environ.get("LLM_POSTGRES_USERNAME") == ""
):
    raise Exception("Invalid LLM_POSTGRES_USERNAME in env")
username = os.environ["LLM_POSTGRES_USERNAME"]
if (
    os.environ.get("LLM_POSTGRES_PASSWORD") is None
    or os.environ.get("LLM_POSTGRES_PASSWORD") == ""
):
    raise Exception("Invalid LLM_POSTGRES_PASSWORD in env")
password = os.environ["LLM_POSTGRES_PASSWORD"]
if (
    os.environ.get("LLM_POSTGRES_HOST") is None
    or os.environ.get("LLM_POSTGRES_HOST") == ""
):
    raise Exception("Invalid LLM_POSTGRES_HOST in env")
host = os.environ["LLM_POSTGRES_HOST"]
port = None
if (
    os.environ.get("LLM_POSTGRES_PORT") is None
    or os.environ.get("LLM_POSTGRES_PORT") == ""
):
    port = "5432"


def get_new_connection():
    debug_logger.info("Creating new PG connection...")
    conn = psycopg2.connect(
        database=os.environ["LLM_POSTGRES_DATABASENAME"],
        user=os.environ["LLM_POSTGRES_USERNAME"],
        password=os.environ["LLM_POSTGRES_PASSWORD"],
        host=os.environ["LLM_POSTGRES_HOST"],
        port=os.environ["LLM_POSTGRES_PORT"],
    )
    debug_logger.info("Connection established", extra={
        "database": os.environ["LLM_POSTGRES_DATABASENAME"],
        "host": os.environ["LLM_POSTGRES_HOST"],
        "port": os.environ["LLM_POSTGRES_PORT"],
    })
    return conn


def get_new_connection_pool():
    debug_logger.info("Creating PostgreSQL connection pool...")
    pool = psycopg2.pool.ThreadedConnectionPool(
        5,
        50,
        database=os.environ["LLM_POSTGRES_DATABASENAME"],
        user=os.environ["LLM_POSTGRES_USERNAME"],
        password=os.environ["LLM_POSTGRES_PASSWORD"],
        host=os.environ["LLM_POSTGRES_HOST"],
        port=os.environ["LLM_POSTGRES_PORT"],
    )
    debug_logger.info("PostgreSQL connection pool created", extra={
        "database": os.environ["LLM_POSTGRES_DATABASENAME"],
        "host": os.environ["LLM_POSTGRES_HOST"],
        "port": os.environ["LLM_POSTGRES_PORT"],
    })
    return pool


def get_connection_url_psycopg2():
    return f'postgresql+psycopg2://{os.environ.get("LLM_POSTGRES_USERNAME")}:{os.environ.get("LLM_POSTGRES_PASSWORD")}@{os.environ.get("LLM_POSTGRES_HOST")}:{os.environ.get("LLM_POSTGRES_PORT")}/{os.environ.get("LLM_POSTGRES_DATABASENAME")}'


def get_connection_url_pg():
    return f'postgresql://{os.environ.get("LLM_POSTGRES_USERNAME")}:{os.environ.get("LLM_POSTGRES_PASSWORD")}@{os.environ.get("LLM_POSTGRES_HOST")}:{os.environ.get("LLM_POSTGRES_PORT")}/{os.environ.get("LLM_POSTGRES_DATABASENAME")}'


db_con_pool = get_new_connection_pool()
