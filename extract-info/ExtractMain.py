from curses_display import Display
from ClassifyUsingSavedModel import ClassifyUsingSavedModel
from ExtractInfo import ExtractInfo
from ProcessAndMergeInfo import ProcessAndMergeInfo
from MergeAndExportInfo import MergeAndExportInfo
from extractor import run_extraction
from GenerateDataFromHtml import GenerateDataFromHtml


PARAMETERS = {
        'in_path_a': '../data/parsed-data/euro-jobs-parsed.csv',
        'out_path_a': '../data/extracted-data/merged/result-andr.csv',
        'out_path_r': '../data/extracted-data/merged/final-data-rub.csv',
        'downloads_path': '../data/downloads',
        'degree_list_csv': {
            'primary': '../data/essential/elementary_pr.csv',
            'secondary': '../data/essential/secondary_program.csv',
            'diploma': '../data/essential/diploma_program.csv',
            'bachelor': '../data/essential/bachelor_program.csv',
            'masters': '../data/essential/master_program.csv',
            'phd': '../data/essential/phd_program.csv',
        },  # degree list are provided as per priority. minimum degree have more priority than the heigher one. priority list diploma > bachelor > masters > phd
        'preprocessor_mode': 2,
        'num_postings': 0,      # change the number of postings to run the extraction for (0 -> all)
        'execute_from': 3,      # specify from what checkpoint the code should be executed
        'execute_to':   5,      # specify at what checkpoint the execution should terminate
        'execute_skip': [],     # specify if there are any checkpoints that should be skipped
        'to_calculate': [
            'job_title',
            'skills',
            'sector',
            'estimated_salary',
            'education_requirements',
            'employment_type',
            'job_location',
            'work_hours',
            'base_salary',
            'currency_info',
        ]
    }


def to_execute(params, checkpoint):
    return params['execute_from'] <= checkpoint <= params['execute_to'] and checkpoint not in params['execute_skip']


def extract_main():
    display = Display(dummy=True)

    if to_execute(PARAMETERS, 0):
        print("Checkpoint 0")
        GenerateDataFromHtml()._init_(display)
    if to_execute(PARAMETERS, 1):
        print("Checkpoint 1")
        ClassifyUsingSavedModel()._init_(display)
    if to_execute(PARAMETERS, 2):
        print("Checkpoint 2")
        ExtractInfo()._init_(PARAMETERS, display)
    if to_execute(PARAMETERS, 3):
        print("Checkpoint 3")
        run_extraction(PARAMETERS, display)
    if to_execute(PARAMETERS, 4):
        print("Checkpoint 4")
        ProcessAndMergeInfo()._init_(PARAMETERS, display)
    if to_execute(PARAMETERS, 5):
        print("Checkpoint 5")
        MergeAndExportInfo()._init_(display)


if __name__ == '__main__':
    print("Starting up...")
    extract_main()
