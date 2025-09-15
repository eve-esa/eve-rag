import pandas as pd
from sqlalchemy import create_engine

def get_rds_metadata(DB_HOST=None, DB_PORT=None, DB_USER=None, DB_PASSWORD=None, DB_NAME=None, num_of_docs=None):
    """
    Connects to a MySQL RDS database and retrieves cleaned metadata from the 'metadata' table.

    Args:
        DB_HOST (str): Hostname or IP address of the RDS database.
        DB_PORT (int): Port number to connect to the database.
        DB_USER (str): Username for database authentication.
        DB_PASSWORD (str): Password for database authentication.
        DB_NAME (str): Name of the database to connect to.
        num_of_docs (int, optional): Number of documents (rows) to retrieve. 
                                     If None, returns all rows. 

    Returns:
        pd.DataFrame: A cleaned DataFrame containing metadata records.
    """
    engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    query = "SELECT * FROM metadata"
    if num_of_docs is not None and isinstance(num_of_docs, int) and num_of_docs > 0:
        query += f" LIMIT {num_of_docs}"

    # Fetch all metadata
    meta_data_rds = pd.read_sql(query, engine)

    # Drop unnecessary columns and rows with missing values
    drop_cols = ['created_at', 'updated_at', 'scraper', 'authors', 'volume', 'issue', 'page', 'month']
    meta_data_rds = meta_data_rds.drop(columns=[col for col in drop_cols if col in meta_data_rds.columns])

    cols_to_fill = ['citation_count', 'reference_count', 'influential_citation_count']
    for col in cols_to_fill:
        if col in meta_data_rds.columns:
            meta_data_rds[col] = meta_data_rds[col].fillna(0).astype(int)

    meta_data_rds = meta_data_rds.dropna()

    if 'citation_count' in meta_data_rds.columns:
        meta_data_rds.rename(columns={'citation_count': 'n_citations'}, inplace=True)

    if num_of_docs is not None:
        meta_data_rds = meta_data_rds[:num_of_docs]

    return meta_data_rds
