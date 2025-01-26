import sys
from os.path import dirname, abspath
import logging

# Add the directory of the 'treat' folder to the system path
sys.path.append(abspath(dirname(__file__)) + "/treat")

# Import the Flask app from the app module
from app import app

if __name__ == '__main__':
    # Run the Flask app with debug mode enabled
    app.run(debug=True)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
