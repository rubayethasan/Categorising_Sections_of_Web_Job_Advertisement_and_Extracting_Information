if [ -d "../venv" ]
then
    printf "\n### Virtual environment already existing ###\n"
else
    printf "\n##### Creating new virtual environment #####\n\n"
    virtualenv ../venv
fi

printf "\n###### Activating virtual environment ######\n"
source ../venv/bin/activate

printf "\n####### Installing required packages #######\n\n"
pip3 install -r requirements.txt

printf "\n########### Running setup script ###########\n\n"
python3 setup.py