# Title 
Categorising Sections of Web Job Advertisement and Extracting Information.
## Goal
Classifying different sections of HTML job advertisements and extracting specific useful information for populating a Job Knowledge Base.

## Abstract
The number of job advertisements on different web platforms has been increasing at a high rate. Each job advertisement contains a vast amount of information about the occupation at hand (e.g. work hours, base salary, currency, job location, educational degrees, skills etc.). Analyzing this information is very important to get more insight into the labour market. Before exploring this domain’s information, we need to find an efficient extraction process to collect specific information from job advertisements. To extract relevant information using the manual strategy with human intervention is time-consuming, costly and inefficient. Several data extraction approaches from web pages have been developed in recent years, which need manually developed ground-truth dataset.
This research work has created an automatic process for generating ground-truth data with a manually annotated dataset. We have used a similarity-based comparison approach between annotated data and text chunks parsed from HTML job advertisements to label the text chunks. This process is evaluated with a manual approach where we have achieved 80% f1-score for the dataset prepared using 2-gram features and 0.35 cosine similarity threshold. From this labelled data, we have extracted different features (e.g. N-gram, Parts of Speech, Named Entity Relation, Numerical Token and Number of Word). We evaluated different state-of-the-art machine learning algorithms (e.g. logistic regression, support vector machine, multinomial naive bays and decision tree) and an ensemble vote classifier to label text chunks. We have observed the support vector machine model performed better than other models and achieved an f1-score of 80% and recall of 81%. Finally, specific information is extracted from these labelled text chunks using a combination of rule-based and pre-trained Named Entity Relation model. The extracted information is found significantly accurate.

## Class Description:
    1. extract-info/GenerateDataFromHtml.py : Class for creating text chunks and plain text from html documents.
    2. extract-info/ClassifyUsingSavedModel.py : Class for performing classification task to genarate labeled chunks.
    3. extract-info/ExtractInfo.py : Class for extracting information from text chunks and palin text.
    4. extract-info/ProcessAndMergeInfo.py : Class for processingand cleannig extracted data as merged data.
    5. extract-info/MergeAndExportInfo.py : Class for merging and exporting final data.
    6. extract-info/ExtractMain.py : Class for invoking all process.

## How to Run the process:

    1.Run the run.bash file. 
    2.Running the run.bash file will initiatea vertual invironment and will install all dependencies and download required data files. And also willinvoke the extract-info/ExtractMain.py file which contains all other classes for performing the wholeoperation.
    2.Alternatively (From genarating the text chunk from html till exporting final data) execute the command in the terminal
    'python extract-info/ExtractMain.py'. In this process input is the file contains all html documents and output is the file 
    contains all extracted information.
    
## Used Technologies
1. Algorithms: DecisionTrees,LogisticRegression,SupportVectorMachine,MultinomialNaiveBays,EnsembleVoteClassifier.
2. Platforms: Python,Scikit-learn,MySQL,NLTK,Spacy,Matplotlib,Scrapy, Translation API.
