"""
Database connection and utility functions
Handles SQLite for development and PostgreSQL for production
"""
import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.config import DATABASE_URL, DATA_DIR
from src.config.logging_config import setup_logger

logger = setup_logger(__name__)

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, use_postgres: bool = False):
        """
        Initialize database manager
        
        Args:
            use_postgres: If True, use PostgreSQL; otherwise use SQLite
        """
        self.use_postgres = use_postgres
        
        if use_postgres:
            self.engine = create_engine(DATABASE_URL)
            logger.info(f"Connected to PostgreSQL database")
        else:
            # Use SQLite for development
            db_path = DATA_DIR / "ecommerce.db"
            self.engine = create_engine(f"sqlite:///{db_path}")
            logger.info(f"Using SQLite database at {db_path}")
        
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self):
        """Get a new database session"""
        return self.Session()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            Query results as pandas DataFrame
        """
        try:
            df = pd.read_sql_query(query, self.engine, params=params)
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str, 
                      if_exists: str = 'replace', index: bool = False):
        """
        Save DataFrame to database table
        
        Args:
            df: DataFrame to save
            table_name: Name of the table
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            index: Whether to save index
        """
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
            logger.info(f"Saved {len(df)} rows to table '{table_name}'")
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names in the database"""
        query = """
        SELECT name FROM sqlite_master 
        WHERE type='table'
        ORDER BY name;
        """ if not self.use_postgres else """
        SELECT tablename FROM pg_tables
        WHERE schemaname='public'
        ORDER BY tablename;
        """
        df = self.execute_query(query)
        return df.iloc[:, 0].tolist()
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        return table_name in self.get_table_names()

# Create default database manager (SQLite for development)
db_manager = DatabaseManager(use_postgres=False)