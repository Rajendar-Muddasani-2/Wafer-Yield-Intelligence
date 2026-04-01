"""
Database Connection Test Script

Tests database connectivity and validates environment configuration.
"""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from utils.config_loader import get_config, ConfigurationError
from utils.database import get_db_connection, DatabaseError


def test_configuration():
    """Test configuration loading"""
    print("=" * 60)
    print("Testing Configuration")
    print("=" * 60)
    
    try:
        config = get_config()
        print("✅ Configuration loaded successfully!")
        print(f"\n📊 Configuration Summary:")
        print(f"  Database TNS: {config.db_tns}")
        print(f"  Database Port: {config.db_port}")
        print(f"  Product Family: {config.product_family}")
        print(f"  Product Type: {config.product_type}")
        print(f"  TP Version: {config.tp_version}")
        print(f"  Default Retest Model: {config.default_retest_model}")
        print(f"  Default Pattern Model: {config.default_pattern_model}")
        print(f"  Log Level: {config.log_level}")
        print(f"  Debug Mode: {config.debug_mode}")
        return True
    except ConfigurationError as e:
        print(f"❌ Configuration Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def test_database_connection():
    """Test database connection"""
    print("\n" + "=" * 60)
    print("Testing Database Connection")
    print("=" * 60)
    
    try:
        db = get_db_connection()
        
        print("🔌 Connecting to database...")
        if db.test_connection():
            print("✅ Database connection successful!")
            
            # Test basic query
            print("\n📝 Running test query...")
            result = db.read_sql("SELECT SYSDATE as current_time FROM DUAL")
            print(f"  Current database time: {result.iloc[0, 0]}")
            
            return True
        else:
            print("❌ Database connection failed!")
            return False
            
    except DatabaseError as e:
        print(f"❌ Database Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def test_table_access():
    """Test access to required tables"""
    print("\n" + "=" * 60)
    print("Testing Table Access")
    print("=" * 60)
    
    try:
        db = get_db_connection()
        
        required_tables = [
            'lot_info',
            'lot',
            'lot_wafer',
            'lot_wafer_site',
            'lot_stats',
            'lot_analysis',
            'lot_ml',
        ]
        
        print("\n📋 Checking required tables:")
        all_exist = True
        for table in required_tables:
            exists = db.table_exists(table)
            status = "✅" if exists else "❌"
            print(f"  {status} {table}")
            
            if exists:
                count = db.get_table_row_count(table)
                print(f"      Row count: {count:,}")
            
            all_exist = all_exist and exists
        
        if all_exist:
            print("\n✅ All required tables exist!")
        else:
            print("\n⚠️  Some required tables are missing!")
            print("    Please create missing tables before running pipeline.")
        
        return all_exist
        
    except DatabaseError as e:
        print(f"❌ Database Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def test_model_access():
    """Test access to ML models"""
    print("\n" + "=" * 60)
    print("Testing ML Model Access")
    print("=" * 60)
    
    try:
        db = get_db_connection()
        config = get_config()
        
        # Check for models in database
        query = """
            SELECT model_type, accuracy, training_date
            FROM lot_ml
            WHERE product_type = :product_type
            AND tp_version = :tp_version
            ORDER BY training_date DESC
        """
        
        print(f"\n🤖 Checking models for {config.product_type}_{config.tp_version}:")
        models_df = db.read_sql(query, {
            'product_type': config.product_type,
            'tp_version': config.tp_version
        })
        
        if len(models_df) > 0:
            print("✅ Found models in database:")
            for _, row in models_df.iterrows():
                print(f"  • {row['model_type']}: "
                      f"Accuracy={row['accuracy']:.2%}, "
                      f"Trained={row['training_date']}")
            return True
        else:
            print("⚠️  No models found in database!")
            print("    Please train models before running predictions.")
            return False
            
    except DatabaseError as e:
        print(f"❌ Database Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Wafer Yield Intelligence System" + " " * 16 + "║")
    print("║" + " " * 15 + "Database Connection Test" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        'Configuration': test_configuration(),
        'Database Connection': test_database_connection(),
        'Table Access': test_table_access(),
        'Model Access': test_model_access(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:.<40} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! System is ready to use.")
    else:
        print("⚠️  Some tests failed. Please address the issues above.")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
