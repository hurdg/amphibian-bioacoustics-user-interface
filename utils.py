import streamlit as st
import opensoundscape
import pandas as pd
import os
from opensoundscape.ml.cnn import load_model
from opensoundscape import Audio, Spectrogram
from opensoundscape.preprocess.preprocessors import SpectrogramPreprocessor
from opensoundscape.ml.datasets import AudioSplittingDataset
from opensoundscape.ml.utils import collate_audio_samples_to_tensors
import torch.nn as nn
from torchvision import models
import tqdm

import pickle
import glob
import re

import torch
import sqlite3

def get_filenames(user_input):
    audio_filepaths = glob.glob(os.path.join(user_input, '*.wav'))
    filename_toreview = [os.path.basename(filepath) for filepath in audio_filepaths]
    return(filename_toreview)

#Check to see if scores.json is present within folder
#Presence indicates  that classifier model has already been run on folder files
def get_scores(user_input, audio_filenames):
    if os.path.isfile(os.path.join(user_input, 'amphib_db.db')):
        pass

    else:
        #If scores.json does not exist (i.e. audio files have not been automatically classified) then load and run the classifier
        ###################################

        #Initalize audio preprocessor for spectrogram conversion
        preprocessor = SpectrogramPreprocessor(sample_duration=3, height=224, width=224)

        #Spectrogram properties
        preprocessor.pipeline.to_spec.params.window_type = 'hann' 
        preprocessor.pipeline.to_spec.params.window_samples = 256
        preprocessor.pipeline.to_spec.params.overlap_fraction = 0.5
        preprocessor.pipeline.bandpass.bypass= True
        preprocessor.pipeline.load_audio.params.sample_rate = 16000

        #Initiate resnet models
        weto_model = ConfigureResnet(architecture = 'resnet34', dropout = False,  dropout_rate=0.5)
        print('using weto_state_dict...')
        weto_model.load_state_dict(torch.load('weto_state_dict.pth', map_location=torch.device('cpu') ))
        wofr_model = ConfigureResnet(architecture = 'resnet34', dropout = False,  dropout_rate=0.5)
        print('using wofr_state_dict...')
        wofr_model.load_state_dict(torch.load('wofr_state_dict.pth', map_location=torch.device('cpu') ))

        #Create object to store audio filepaths
        filepaths = [os.path.join(user_input, filename) for filename in audio_filenames]
        filepath_presence_dict = {"filepath":filepaths}
        sample_df = pd.DataFrame(filepath_presence_dict).set_index('filepath')

        #Create dataloader - remove any augmentations
        dataset = AudioSplittingDataset(sample_df, preprocessor)
        dataset.bypass_augmentations = True # Remove augmentations
        pred_dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=60, 
                                            shuffle=False,
                                            collate_fn = collate_audio_samples_to_tensors)  

        #Run audio files through recognizers to obtain prediction scores
        df_list = []
        weto_model.eval()
        wofr_model.eval()
        with torch.no_grad():
            for i, (batch, _) in enumerate(tqdm.tqdm(pred_dataloader)):

                #Weto 
                weto_output = torch.sigmoid(weto_model(batch))
                weto_nested_list = weto_output.tolist()
                weto_pos = [inner_item for outer_item in weto_nested_list for inner_item in outer_item]
                #weto_neg = [1-value for value in weto_pos]

                #Wofr
                wofr_output = torch.sigmoid(wofr_model(batch))
                wofr_nested_list = wofr_output.tolist()
                wofr_pos = [inner_item for outer_item in wofr_nested_list for inner_item in outer_item]
                #wofr_neg = [1-value for value in wofr_pos]

                #Append results
                df = pred_dataloader.dataset.label_df.reset_index().iloc[60*i:60*(i+1)]
                df['weto_pos'] = weto_pos
                #df['weto_neg'] = weto_neg
                df['wofr_pos'] = wofr_pos
                #df['wofr_neg'] = wofr_neg
                df_list.append(df)

        scores = pd.concat(df_list)  

        #Write json of scores for future use
        #writes to same folder where audio files are stored
        scores[['transcriber_weto','transcriber_wofr', 'weto', 'wofr']] = None
        scores['filename'] = scores['file'].apply(lambda x: os.path.basename(x))
        try:
            print("Opening SQLite connection to create database")
            sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
            cursor = sqlite_connection.cursor()

            for filename_i in audio_filenames:
                df_i = scores.loc[scores['filename'] == filename_i]
                df_i.to_sql(filename_i, sqlite_connection, if_exists='replace', dtype={'file': 'TEXT', 
                                                                                       'start_time': 'INTEGER',
                                                                                       'end_time': 'INTEGER',
                                                                                       'weto_pos': 'REAL',
                                                                                       'wofr_pos': 'REAL',
                                                                                       'transcriber_weto': 'TEXT',
                                                                                       'transcriber_wofr': 'TEXT',
                                                                                       'filename': 'TEXT'})

            sqlite_connection.commit()
            print("Database created")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS summary_table (
                    filename TEXT PRIMARY KEY,
                    transcriber_weto TEXT,
                    transcriber_wofr TEXT,
                    weto INTEGER,
                    wofr INTEGER
                );
                """)
            
            sqlite_connection.commit()
            print("Summary table created")

            sqlite_connection.close()


        # Handle errors
        except sqlite3.Error as error:
            print('Error occurred - ', error)

        # Close DB Connection irrespective of success
        # or failure
        finally:
        
            if sqlite_connection:
                sqlite_connection.close()
                print('SQLite Connection closed')

    return()

class CompareAggregate:
    def __init__(self):
        self.priority = 0   # Start with False, will set to True if condition is met
    # This function will be called for each row

    def step(self, spp, transcriber):
        # Example: Define column comparison conditions dynamically
        comparisons = [
            lambda spp, transcriber: 5 if spp == str(1) and (transcriber != 'eim_ai' and transcriber is not None) else 0, 
            lambda spp, transcriber: 4    if spp == str(1) and (transcriber == 'eim_ai' or transcriber is None) else 0,
            lambda spp, transcriber: 3    if spp == None else 0,
            lambda spp, transcriber: 2 if spp != str(1) and (transcriber != 'eim_ai' and transcriber is not None) else 0,
            lambda spp, transcriber: 1    if spp != str(1) and (transcriber == 'eim_ai' or transcriber is None) else 0,
        ]
        for comparison in comparisons:
            priority_tmp = comparison(spp, transcriber)  # Apply condition on column values

            if priority_tmp > self.priority:
                self.priority = priority_tmp
                break
            
        
    # This function will be called once all rows are processed
    def finalize(self):
        return (self.priority)


def get_ai_classification(audio_filenames, ai_range, spp_code, user_input, user_name):
    lower = ai_range[0]/100
    upper = ai_range[1]/100

    try:
        print("Opening SQLite connection to automatically classify recordings")
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
        cursor = sqlite_connection.cursor()
        sqlite_connection.create_aggregate("compare_columns_aggregate", 2, CompareAggregate)  # Accepts 2 column values

        # WHERE CLAUSE TO UPDATE DATA
        for filename_i in audio_filenames:
            for spp_code in ['weto', 'wofr']:
                #Record samples that are classified by AI - manual classification has priority
                #Set samples with scores above upper threshold
                cursor.execute(f"""
                            UPDATE '{filename_i}' SET 
                            {'transcriber_'+spp_code} ='eim_ai',
                            {spp_code} = 1
                                WHERE    ({spp_code + '_pos'} > {upper})  AND 
                                        ({'transcriber_'+spp_code} = 'eim_ai' OR {'transcriber_'+spp_code} IS NULL)
                            """)
                #Set samples with scores below lower threshold
                cursor.execute(f"""
                            UPDATE '{filename_i}' SET 
                            {'transcriber_'+spp_code} ='eim_ai',
                            {spp_code} = 0
                                WHERE    ({spp_code + '_pos'} < {lower})  AND 
                                        ({'transcriber_'+spp_code} = 'eim_ai' OR {'transcriber_'+spp_code} IS NULL)
                            """)

                #Overwrite samples that are not classified by AI, while respecting those that have been manually classified
                cursor.execute(f"""
                                UPDATE '{filename_i}' SET 
                                {'transcriber_'+spp_code} =NULL, 
                                {spp_code} = NULL
                                    WHERE   ({spp_code + '_pos'} >= {lower} AND {spp_code + '_pos'} <= {upper})  AND 
                                            ({'transcriber_'+spp_code} = 'eim_ai' OR {'transcriber_'+spp_code} IS NULL)
                                    """)

                cursor.execute(f"""
    WITH    priority_weto AS (SELECT filename, compare_columns_aggregate(weto, transcriber_weto) as num
                                FROM '{filename_i}'),
            priority_wofr AS (SELECT filename, compare_columns_aggregate(wofr, transcriber_wofr) as num
                            FROM '{filename_i}')

    INSERT OR REPLACE INTO summary_table (filename, transcriber_weto, transcriber_wofr, weto, wofr) 

    SELECT 
        priority_weto.filename,
        CASE
            WHEN priority_weto.num = 5 THEN '{user_name}'
            WHEN priority_weto.num = 4 THEN 'eim_ai'
            WHEN priority_weto.num = 3 THEN NULL
            WHEN priority_weto.num = 2 THEN '{user_name}'
            WHEN priority_weto.num = 1 THEN 'eim_ai'
        END AS transcriber_weto,
        CASE
            WHEN priority_wofr.num = 5 THEN '{user_name}'
            WHEN priority_wofr.num = 4 THEN 'eim_ai'
            WHEN priority_wofr.num = 3 THEN NULL
            WHEN priority_wofr.num = 2 THEN '{user_name}'
            WHEN priority_wofr.num = 1 THEN 'eim_ai'
        END AS transcriber_wofr,
        CASE
            WHEN priority_weto.num > 3 THEN 1
            WHEN priority_weto.num = 3 THEN NULL
            WHEN priority_weto.num < 3 THEN 0
        END AS weto,
        CASE
            WHEN priority_wofr.num > 3 THEN 1
            WHEN priority_wofr.num = 3 THEN NULL
            WHEN priority_wofr.num < 3 THEN 0
        END AS wofr
    FROM priority_weto LEFT JOIN priority_wofr
                                    """)

        sqlite_connection.commit()  # Commit only if all statements succeed
        sqlite_connection.close()

    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)
        sqlite_connection.rollback()  # Undo all changes if an error occurs

    # Close DB Connection irrespective of success
    # or failure
    finally:
    
        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')

    return()

def get_sidebar_table(audio_filenames, spp_code, key, user_input, selection, user_name, auto_filer):
# Define the custom aggregate class
    # class CompareAggregate:
    #     def __init__(self):
    #         self.priority = 0   # Start with False, will set to True if condition is met
    #     # This function will be called for each row

    #     def step(self, spp, transcriber):
    #         # Example: Define column comparison conditions dynamically
    #         comparisons = [
    #             lambda spp, transcriber: 5 if spp == str(1) and (transcriber != 'eim_ai' and transcriber is not None) else 0, 
    #             lambda spp, transcriber: 4    if spp == str(1) and (transcriber == 'eim_ai' or transcriber is None) else 0,
    #             lambda spp, transcriber: 3    if spp == None else 0,
    #             lambda spp, transcriber: 2 if spp != str(1) and (transcriber != 'eim_ai' and transcriber is not None) else 0,
    #             lambda spp, transcriber: 1    if spp != str(1) and (transcriber == 'eim_ai' or transcriber is None) else 0,
    #         ]
    #         for comparison in comparisons:
    #             priority_tmp = comparison(spp, transcriber)  # Apply condition on column values

    #             if priority_tmp > self.priority:
    #                 self.priority = priority_tmp
    #                 break
                
            
    #     # This function will be called once all rows are processed
    #     def finalize(self):
    #         return (self.priority)
    try:
        print("Opening SQLite connection to extract and update summary table")
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
    #     sqlite_connection.create_aggregate("compare_columns_aggregate", 2, CompareAggregate)  # Accepts 2 column values

    #     cursor = sqlite_connection.cursor()
    #     for filename_i in audio_filenames:
    #         cursor.execute(f"""
                        
    # -- create and insert a summary row of each table in the database        

       
    # WITH    priority_weto AS (SELECT filename, compare_columns_aggregate(weto, transcriber_weto) as num
    #                             FROM '{filename_i}'),
    #         priority_wofr AS (SELECT filename, compare_columns_aggregate(wofr, transcriber_wofr) as num
    #                         FROM '{filename_i}')

    # INSERT OR REPLACE INTO summary_table (filename, transcriber_weto, transcriber_wofr, weto, wofr) 

    # SELECT 
    #     priority_weto.filename,
    #     CASE
    #         WHEN priority_weto.num = 5 THEN '{user_name}'
    #         WHEN priority_weto.num = 4 THEN 'eim_ai'
    #         WHEN priority_weto.num = 3 THEN NULL
    #         WHEN priority_weto.num = 2 THEN '{user_name}'
    #         WHEN priority_weto.num = 1 THEN 'eim_ai'
    #     END AS transcriber_weto,
    #     CASE
    #         WHEN priority_wofr.num = 5 THEN '{user_name}'
    #         WHEN priority_wofr.num = 4 THEN 'eim_ai'
    #         WHEN priority_wofr.num = 3 THEN NULL
    #         WHEN priority_wofr.num = 2 THEN '{user_name}'
    #         WHEN priority_wofr.num = 1 THEN 'eim_ai'
    #     END AS transcriber_wofr,
    #     CASE
    #         WHEN priority_weto.num > 3 THEN 1
    #         WHEN priority_weto.num = 3 THEN NULL
    #         WHEN priority_weto.num < 3 THEN 0
    #     END AS weto,
    #     CASE
    #         WHEN priority_wofr.num > 3 THEN 1
    #         WHEN priority_wofr.num = 3 THEN NULL
    #         WHEN priority_wofr.num < 3 THEN 0
    #     END AS wofr
    # FROM priority_weto LEFT JOIN priority_wofr

    # ;""")
        
    #     sqlite_connection.commit()  # Commit only if all statements succeed
        df = pd.read_sql("SELECT * FROM summary_table", sqlite_connection)
        if auto_filer:
            df.sort_values(by=f'{spp_code}', inplace = True, ascending = False, na_position='first')
        sqlite_connection.close()

    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)
        sqlite_connection.rollback()  # Undo all changes if an error occurs

    # Close DB Connection irrespective of success
    # or failure
    finally:

        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')

    


    column_configuration = {
    "filename": st.column_config.TextColumn(
        "Filename", help="The name of the file within the file folder", max_chars=100, width=None
    ),
    'transcriber_'+spp_code: st.column_config.TextColumn(
        "Transcriber", help="The name of the transcriber", max_chars=100, width=None
    ),

    "weto": st.column_config.TextColumn(
        "Western Toad",
        help="Western toad detection status",
        width=None,
    ),
    "wofr": st.column_config.TextColumn(
        "Wood Frog",
        help="Wood frog detection status",
        width=None,
    ),
    }

    #Inner conversion removes decimals, outer conversion formats for streamlit
    df = df.astype({'weto': 'Int32','wofr': 'Int32'}).astype({'weto': 'object','wofr': 'object'})

    def color_coding(row):
        if (str(row[spp_code]) == '<NA>'):
            result = ['background-color:yellow'] * len(row)
        else:
            result = [''] * len(row)
        return(result)

            
    sidebar_df = st.dataframe(
        df[['filename', f'transcriber_{spp_code}', f'{spp_code}']].style.apply(color_coding, axis=1),
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key = key
        #height = 550
    )

    return(df, sidebar_df)


def no_update_json(filename, spp_code, user_name, yesno_key, user_input):

    try:
        print("Opening SQLite connection to insert a negative manual classification")
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
        cursor = sqlite_connection.cursor()
        cursor.execute(f"""
        UPDATE '{filename}'
        SET {'transcriber_'+spp_code} = '{user_name}',
            {spp_code} = 0
            WHERE start_time = 3 * {yesno_key};
        """)

        sqlite_connection.commit()
        print("Manual selection updated")

        sqlite_connection.close()


    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # Close DB Connection irrespective of success
    # or failure
    finally:

        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')
        st.session_state.no_button = False

def yes_update_json(filename, spp_code, user_name, yesno_key, user_input):

    try:
        print("Opening SQLite connection to insert a positive manual classification")
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
        cursor = sqlite_connection.cursor()
        cursor.execute(f"""
        UPDATE '{filename}'
        SET {'transcriber_'+spp_code} = '{user_name}',
            {spp_code} = 1
            WHERE start_time = 3 * {yesno_key};
        """)

        sqlite_connection.commit()
        print("Manual selection updated")

        sqlite_connection.close()


    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # Close DB Connection irrespective of success
    # or failure
    finally:

        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')
        st.session_state.yes_button = False


def make_occupancy_df(audio_filenames):

    weto_df_list = []
    wofr_df_list = []
    for filename in audio_filenames:
        df = pd.read_sql(f"SELECT * FROM '{filename}'")
        filename_re = re.compile(r'^(.*)_(\d{8}_\d{6})')
        location = filename_re.match(filename).group(1)
        survey = filename_re.match(filename).group(2)
        
        weto_presence = int(any(df['weto']==1))
        wofr_presence = int(any(df['wofr']==1))
        
        weto_df = pd.DataFrame({'location': [location], 'survey':  [survey], 'presence': [weto_presence]}) 
        wofr_df = pd.DataFrame({'location': [location], 'survey':  [survey], 'presence': [wofr_presence]})

        weto_df_list.append(weto_df)
        wofr_df_list.append(wofr_df)

    weto_occ_df = pd.concat(weto_df_list).pivot(index='location', columns='survey', values='presence')
    wofr_occ_df = pd.concat(wofr_df_list).pivot(index='location', columns='survey', values='presence')

    return(weto_occ_df, wofr_occ_df)


def get_sorted_keys(filename, spp_code, user_input):
    try:
        print(f"Opening SQLite connection to extract {filename} data")
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
        cursor = sqlite_connection.cursor()
        cursor.execute(f"""
    SELECT start_time / 3 as key
        FROM '{filename}'
        WHERE {'transcriber_'+spp_code} IS NULL
        ORDER BY {spp_code+'_pos'} DESC
        ;
            """)

        # Fetch all results as a list of tuples (one tuple per row)
        result = cursor.fetchall()
        # Convert to a list and flattern
        sorted_keys_unclassified = [int(row[0]) for row in result]

        file_dict = pd.read_sql(f"SELECT * FROM '{filename}'", sqlite_connection).to_dict()

        sqlite_connection.close()

    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # Close DB Connection irrespective of success
    # or failure
    finally:

        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')
    return(sorted_keys_unclassified, file_dict)



def reset_file_classifications(spp_code, filename, user_input):
    print(f"Opening SQLite connection to reset {filename}")
    try:
        print(f'resetiing {filename}')
        sqlite_connection = sqlite3.connect(os.path.join(user_input, 'amphib_db.db'))
        cursor = sqlite_connection.cursor()
        cursor.execute(f"""
        UPDATE '{filename}'
        SET {'transcriber_'+spp_code} = NULL,
            {spp_code} = NULL
            ;""")
        sqlite_connection.commit()
        sqlite_connection.close()

    # Handle errors
    except sqlite3.Error as error:
        print('Error occurred - ', error)

    # Close DB Connection irrespective of success
    # or failure
    finally:

        if sqlite_connection:
            sqlite_connection.close()
            print('SQLite Connection closed')


##### Weto Model Architecture #######

class ConfigureResnet(nn.Module):
    def __init__(self, architecture, dropout:bool, dropout_rate=0.5):
        super(ConfigureResnet, self).__init__()
        # Load a pretrained ResNet model
        self.resnet = getattr(models, architecture)(weights=None)
        
        num_features = self.resnet.fc.in_features
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if dropout:
        # Replace the fully connected layer with a new one that includes dropout
            self.resnet.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(num_features, out_features =1)
            )
        else:
            self.resnet.fc = nn.Linear(num_features, out_features =1) 

    def forward(self, x):
        return self.resnet(x)


##### Functions #######
#Load the classifier model (cache (?))
@st.cache_data
def loading_model():
    weto_model_path = os.path.join(os.getcwd(), 'weto978.model')
    #weto_model_path = os.path.join(os.getcwd(), 'final.model')
    print("Using 'final.model'!!")
    wofr_model_path = os.path.join(os.getcwd(), 'wofr997.model')
    
    weto_model = opensoundscape.ml.cnn.load_model(weto_model_path)
    wofr_model = opensoundscape.ml.cnn.load_model(wofr_model_path)
    return weto_model, wofr_model


def get_predictions(_model, user_input, filename_toreview):
    filepaths = [os.path.join(user_input, filename) for filename in filename_toreview]
    output = _model.predict(filepaths, activation_layer = 'sigmoid', batch_size  = 60, num_workers = 0)
    output.reset_index(inplace=True)
    return output


def get_file_attr(df, user_input, selection):
    filename = df['filename'].iloc[selection[0]]
    filepath = os.path.join(user_input, filename)
    return(filename, filepath)

##### Functions #######
def get_df(audio_filenames, spp_code):
    df = pd.DataFrame(columns = ['filename', 'transcriber_'+spp_code, spp_code+'_start_time',  spp_code])
    df['filename'] = audio_filenames
    return(df)


def make_wildtrax_df(output):

    df_list = []

    for filename in output.keys():
        df_raw = pd.DataFrame(output[filename])
        df_weto = df_raw[['transcriber_weto', 'start_time']][(df_raw['weto']==1)].rename(columns = {'start_time': 'startTime'})
        df_weto['species'] = 'WETO'
        df_wofr = df_raw[['transcriber_wofr', 'start_time']][(df_raw['wofr']==1)].rename(columns = {'start_time': 'startTime'})
        df_wofr['species'] = 'WOFR'
        df = pd.concat([df_weto.rename(columns={'transcriber_weto':'transcriber'}), df_wofr.rename(columns={'transcriber_wofr':'transcriber'})])
        try:
            #filename_location_re = re.compile(r'[Aa]-\d+?(?=_)')
            filename_location_re = re.compile(r'^(.*)_\d{8}_\d{6}')
            location = filename_location_re.match(filename).group(1)
        except:
            try:
                filepath_location_re = re.compile(r'[Aa]-\d+')
                location = filepath_location_re.match(filename)
            except: location = "NA"
        
        df['location'] = location

        try:
            datetime_re = re.compile(r'_(\d{8})_(\d{6})')
            datetime_raw = datetime_re.search(filename)
            date = '-'.join([datetime_raw.group(1)[:4], datetime_raw.group(1)[4:6], datetime_raw.group(1)[6:]])
            time = ':'.join([datetime_raw.group(2)[:2], datetime_raw.group(2)[2:4], datetime_raw.group(2)[4:]])
            datetime_parsed = date +" "+ time        
        except:
            datetime_parsed = "NA"
        
        df['recordingDate'] = datetime_parsed
        df['method'] = 'NONE'
        df['taskLength'] = 180
        df['speciesIndividualNumber'] = 1
        df['vocalization'] = 'call'
        df['abundance'] = None
        df['tagLength'] = 3
        df['minFreq'] = 0
        df['maxFreq'] = 12000
        df['speciesIndividualComment'] = None
        df['internal_tag_id'] = None

        df_list.append(df)

    wildtrax_df = pd.concat(df_list)
    wildtrax_df = wildtrax_df[['location','recordingDate','method', 
                            'taskLength', 'transcriber', 'species', 
                            'speciesIndividualNumber', 'vocalization', 'abundance', 
                            'startTime', 'tagLength', 'minFreq', 
                            'maxFreq', 'speciesIndividualComment', 'internal_tag_id']]

    return(wildtrax_df)







#create spectrogram image from audio data
#use same spectrogram creation parameters that were used in the classification model
def get_spectrogram(filepath, start_time, end_time):
    with open('model_config.pkl', 'rb') as inp:
        spec_params = pickle.load(inp)

    audio_segment = Audio.from_file(
                                filepath,
                                offset=start_time,
                                duration=end_time-start_time).resample(16000)

    spectrogram_object = Spectrogram.from_audio(
                                                audio_segment,
                                                window_type=spec_params['window_type'],
                                                window_samples=spec_params['window_samples'],
                                                window_length_sec=spec_params['window_length_sec'],
                                                overlap_samples=spec_params['overlap_samples'],
                                                overlap_fraction=spec_params['overlap_fraction'],
                                                fft_size=spec_params['fft_size'],
                                                dB_scale=spec_params['dB_scale'],
                                                scaling=spec_params['scaling'],
    )
    return(spectrogram_object)



def get_classificationDF(user_input, audio_filenames, spp_code):
    #Check to see if classified.csv is present within folder
    #Presence indeciates that classification has already been initated on the audio files (may be completed)
    classified_filepath = os.path.join(user_input, 'ai_tags.csv')

    if os.path.exists(classified_filepath):
        df = pd.read_csv(classified_filepath, index_col = False)
        if spp_code not in df.columns:
            df[spp_code] = pd.Series(dtype='int')
            df[spp_code+'_start_time'] = pd.Series(dtype='int')

    else:
        #Create csv with wildtrax tag formatting
        #To be used to store classification decisions
        #writes to same folder where audio files are stored
        df = get_df(audio_filenames, spp_code)
        df.to_csv(classified_filepath, index = False)
    return(df, classified_filepath)


def get_event_index(grid_event):
    x_event = grid_event.selection['points'][0]['x']
    y_event = grid_event.selection['points'][0]['y']
    event_index = x_event + (11-y_event)*5
    samp_key = str(event_index)
    return(samp_key)

def get_title_markdown(filename):
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<p class="big-font">Filename:  {filename}</p>', unsafe_allow_html=True)

