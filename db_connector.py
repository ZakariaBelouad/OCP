import mysql.connector
import pandas as pd
from mysql.connector import connect

db_config = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "1234",
    "database": "satisfaction_ai"
}

def fetch_evaluation_data():
    query = """
    SELECT 
        e.id_evaluation,
        e.date,
        e.time,
        e.avis,
        e.type,
        e.id_centre,
        v.code_centre,
        v.nom_centre
    FROM evaluation e
    JOIN ville_centre v ON e.id_centre = v.id_centre
    """
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    return df
