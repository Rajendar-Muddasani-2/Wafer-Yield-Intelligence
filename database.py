"""
Database Connection Module

Provides connection management, connection pooling, and common database operations.
"""

import cx_Oracle
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.pool import NullPool
from contextlib import contextmanager
from typing import Optional, Generator, Any, Dict, List
import pandas as pd
import logging

from config_loader import get_config

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when database operations fail"""
    pass


class DatabaseConnection:
    """
    Database connection manager with connection pooling and error handling.
    """
    
    def __init__(self, config=None):
        """
        Initialize database connection manager
        
        Args:
            config: Optional Config instance. If None, uses global config.
        """
        self.config = config or get_config()
        self._engine = None
        self._init_oracle_client()
        
    def _init_oracle_client(self):
        """Initialize Oracle Instant Client"""
        try:
            oracle_client_path = self.config.oracle_client_path
            if oracle_client_path:
                cx_Oracle.init_oracle_client(lib_dir=oracle_client_path)
                logger.info(f"Oracle client initialized: {oracle_client_path}")
        except Exception as e:
            # Client might already be initialized
            logger.debug(f"Oracle client initialization: {e}")
    
    @property
    def engine(self) -> sqlalchemy.engine.Engine:
        """Get or create SQLAlchemy engine with connection pooling"""
        if self._engine is None:
            try:
                connection_string = self.config.db_connection_string
                
                # Create engine with connection pooling
                self._engine = create_engine(
                    connection_string,
                    pool_size=self.config.db_pool_size,
                    max_overflow=self.config.db_max_overflow,
                    pool_pre_ping=True,  # Verify connections before using
                    echo=self.config.debug_mode,  # Log SQL in debug mode
                )
                
                logger.info("Database engine created successfully")
                
            except Exception as e:
                logger.error(f"Failed to create database engine: {e}")
                raise DatabaseError(f"Database connection failed: {e}")
                
        return self._engine
    
    @contextmanager
    def get_connection(self) -> Generator[sqlalchemy.engine.Connection, None, None]:
        """
        Context manager for database connections
        
        Usage:
            with db.get_connection() as conn:
                result = conn.execute(query)
        
        Yields:
            SQLAlchemy Connection object
        """
        conn = None
        try:
            conn = self.engine.connect()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute a SQL query and return results
        
        Args:
            query: SQL query string
            params: Optional dictionary of query parameters
            
        Returns:
            Query result
        """
        with self.get_connection() as conn:
            result = conn.execute(text(query), params or {})
            return result
    
    def read_sql(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            params: Optional dictionary of query parameters
            
        Returns:
            pandas DataFrame with query results
        """
        try:
            with self.get_connection() as conn:
                df = pd.read_sql(text(query), conn, params=params)
                logger.info(f"Query returned {len(df)} rows")
                return df
        except Exception as e:
            logger.error(f"Failed to execute query: {e}")
            raise DatabaseError(f"Query failed: {e}")
    
    def to_sql(self, df: pd.DataFrame, table_name: str, 
               if_exists: str = 'append', 
               chunksize: Optional[int] = None) -> int:
        """
        Write DataFrame to database table
        
        Args:
            df: pandas DataFrame to write
            table_name: Target table name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            chunksize: Number of rows to write at a time
            
        Returns:
            Number of rows written
        """
        try:
            chunksize = chunksize or self.config.batch_size
            
            with self.get_connection() as conn:
                rows_written = df.to_sql(
                    table_name,
                    conn,
                    if_exists=if_exists,
                    index=False,
                    chunksize=chunksize,
                    method='multi'
                )
                
                logger.info(f"Wrote {len(df)} rows to {table_name}")
                return rows_written
                
        except Exception as e:
            logger.error(f"Failed to write to table {table_name}: {e}")
            raise DatabaseError(f"Write failed: {e}")
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database
        
        Args:
            table_name: Name of table to check
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            query = f"""
                SELECT COUNT(*) as cnt
                FROM user_tables
                WHERE table_name = :table_name
            """
            result = self.execute_query(query, {'table_name': table_name.upper()})
            count = result.fetchone()[0]
            return count > 0
        except Exception as e:
            logger.error(f"Failed to check table existence: {e}")
            return False
    
    def get_table_row_count(self, table_name: str) -> int:
        """
        Get row count for table
        
        Args:
            table_name: Name of table
            
        Returns:
            Number of rows in table
        """
        try:
            query = f"SELECT COUNT(*) as cnt FROM {table_name}"
            result = self.execute_query(query)
            count = result.fetchone()[0]
            return count
        except Exception as e:
            logger.error(f"Failed to get row count for {table_name}: {e}")
            return 0
    
    def truncate_table(self, table_name: str):
        """
        Truncate table (delete all rows)
        
        Args:
            table_name: Name of table to truncate
        """
        try:
            with self.get_connection() as conn:
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                logger.info(f"Truncated table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to truncate {table_name}: {e}")
            raise DatabaseError(f"Truncate failed: {e}")
    
    def get_max_value(self, table_name: str, column_name: str) -> Optional[Any]:
        """
        Get maximum value from a column
        
        Args:
            table_name: Name of table
            column_name: Name of column
            
        Returns:
            Maximum value or None if table is empty
        """
        try:
            query = f"SELECT MAX({column_name}) as max_val FROM {table_name}"
            result = self.execute_query(query)
            max_val = result.fetchone()[0]
            return max_val
        except Exception as e:
            logger.error(f"Failed to get max value: {e}")
            return None
    
    def execute_procedure(self, procedure_name: str, params: Optional[List] = None):
        """
        Execute stored procedure
        
        Args:
            procedure_name: Name of stored procedure
            params: Optional list of parameters
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.connection.cursor()
                cursor.callproc(procedure_name, params or [])
                logger.info(f"Executed procedure: {procedure_name}")
        except Exception as e:
            logger.error(f"Failed to execute procedure {procedure_name}: {e}")
            raise DatabaseError(f"Procedure execution failed: {e}")
    
    def get_column_names(self, table_name: str) -> List[str]:
        """
        Get column names for a table
        
        Args:
            table_name: Name of table
            
        Returns:
            List of column names
        """
        try:
            query = f"""
                SELECT column_name
                FROM user_tab_columns
                WHERE table_name = :table_name
                ORDER BY column_id
            """
            result = self.execute_query(query, {'table_name': table_name.upper()})
            columns = [row[0] for row in result.fetchall()]
            return columns
        except Exception as e:
            logger.error(f"Failed to get columns for {table_name}: {e}")
            return []
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute(text("SELECT 1 FROM DUAL"))
                result.fetchone()
                logger.info("Database connection test: SUCCESS")
                return True
        except Exception as e:
            logger.error(f"Database connection test: FAILED - {e}")
            return False
    
    def close(self):
        """Close database engine and cleanup connections"""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            logger.info("Database connections closed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Global database connection instance
_db_connection: Optional[DatabaseConnection] = None


def get_db_connection(config=None) -> DatabaseConnection:
    """
    Get or create global database connection instance
    
    Args:
        config: Optional Config instance
        
    Returns:
        DatabaseConnection instance
    """
    global _db_connection
    if _db_connection is None:
        _db_connection = DatabaseConnection(config)
    return _db_connection


if __name__ == "__main__":
    # Test database connection
    try:
        db = get_db_connection()
        
        print("Testing database connection...")
        if db.test_connection():
            print("✅ Database connection successful!")
            
            # Test basic query
            print("\nTesting basic query...")
            result = db.read_sql("SELECT SYSDATE as current_time FROM DUAL")
            print(f"Current database time: {result.iloc[0, 0]}")
            
        else:
            print("❌ Database connection failed!")
            
    except DatabaseError as e:
        print(f"❌ Database Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
