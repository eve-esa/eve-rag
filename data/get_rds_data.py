import pandas as pd
import pymysql

def get_rds_metadata(DB_HOST=None, DB_PORT=None, DB_USER=None, DB_PASSWORD=None, DB_NAME=None, num_of_docs=None):
    """
    Connects to a MySQL RDS database and retrieves cleaned metadata from the 'metadata' table.

    Args:
        DB_HOST (str): Hostname or IP address of the RDS database.
        port (int): Port number to connect to the database.
        DB_USER (str): Username for database authentication.
        DB_PASSWORD (str): Password for database authentication.
        DB_NAME (str): Name of the database to connect to.
        num_of_docs (int, optional): Number of documents (rows) to retrieve. 
                                     If None, returns all rows. 
                                     If a positive integer, returns only that many rows.

    Returns:
        pd.DataFrame: A cleaned DataFrame containing metadata records.
    """
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

    # Fetch all metadata
    meta_data_rds = pd.read_sql("SELECT * FROM metadata;", conn)

    # Drop unnecessary columns and rows with missing values
    meta_data_rds = meta_data_rds.drop(columns=[
        'created_at', 'updated_at', 'scraper', 'authors', 'volume', 'issue', 'page', 'month'
    ])
    meta_data_rds = meta_data_rds.dropna()
    meta_data_rds.rename(columns={'citation_count': 'n_citations'}, inplace=True)

    cols_to_convert = ['n_citations', 'reference_count', 'influential_citation_count']
    meta_data_rds[cols_to_convert] = meta_data_rds[cols_to_convert].astype(int)

    # Slice to limit number of documents, if specified
    if num_of_docs is not None:
        meta_data_rds = meta_data_rds[:num_of_docs]

    conn.close()
    return meta_data_rds
