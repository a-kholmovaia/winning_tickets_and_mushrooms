# creates a venv folder that contains a copy of python3 packages to isolate any further changes in packages from the system installation
python3 -m venv venv
 # tell the shell to use the created virtual environment
source venv/bin/activate
# install requirements
pip3 install -r requirements.txt
# run the script
python3 main.py
# deactivate the virtual environment
deactivate
# remove the venv
rm -rf venv
