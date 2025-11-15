import db_setup
import process_data

if __name__ == "__main__":
    db_setup.create_database()
    process_data.start_processing()