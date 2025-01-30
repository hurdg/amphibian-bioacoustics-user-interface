import streamlit as st
import pandas as pd
import os
from opensoundscape.ml.cnn import load_model
from opensoundscape import Audio, Spectrogram
import pickle
import glob
import re
import json

##### Functions #######
#Load the classifier model (cache (?))
@st.cache_data
def loading_model():
    weto_model_path = os.path.join(os.getcwd(), 'weto978.model')
    wofr_model_path = os.path.join(os.getcwd(), 'wofr997.model')
    weto_model = load_model(weto_model_path)
    wofr_model = load_model(wofr_model_path)
    return weto_model, wofr_model


def get_predictions(_model, user_input, filename_toreview):
    filepaths = [os.path.join(user_input, filename) for filename in filename_toreview]
    output = _model.predict(filepaths, activation_layer = 'sigmoid', batch_size  = 60, num_workers = 0)
    output.reset_index(inplace=True)
    return output

#Check to see if scores.json is present within folder
#Presence indicates  that classifier model has already been run on folder files
def get_scores(user_input, audio_filenames):
    output_filepath = os.path.join(user_input, 'eim_amphibian_ouput.json')

    if os.path.exists(output_filepath):
        with open(output_filepath) as f:
            output_json = json.load(f)


    else:
        #If scores.json does not exist (i.e. audio files have not been automatically classified) then load and run the classifier
        weto_model, wofr_model = loading_model()
        weto_scores = get_predictions(weto_model, user_input, audio_filenames)
        weto_scores.rename(columns={'negative': 'weto_neg', 'positive': 'weto_pos'}, inplace=True)

        wofr_scores = get_predictions(wofr_model, user_input, audio_filenames)
        wofr_scores.rename(columns={'negative': 'wofr_neg', 'positive': 'wofr_pos'}, inplace=True)

        #Write json of scores for future use
        #writes to same folder where audio files are stored
        output = pd.merge(weto_scores, wofr_scores,  how='left', left_on=['file','start_time', 'end_time'], right_on = ['file','start_time', 'end_time'])
        output[['transcriber_weto','transcriber_wofr', 'weto', 'wofr']] = None
        output['filename'] = output['file'].apply(lambda x: os.path.basename(x))
        output_json = output[['filename','transcriber_weto','transcriber_wofr','start_time','weto_pos', 'weto', 'wofr_pos', 'wofr']].reset_index().groupby('filename')[['transcriber_weto','transcriber_wofr','start_time','weto_pos', 'weto', 'wofr_pos', 'wofr']].apply(
                            lambda x: x.reset_index()[['transcriber_weto','transcriber_wofr','start_time','weto_pos', 'weto', 'wofr_pos', 'wofr']].to_dict()).reset_index().set_index('filename').to_dict()[0]

        with open(output_filepath, 'w') as f:
            json.dump(output_json, f)
        with open(output_filepath) as f:
            output_json = json.load(f)

    return(output_json, output_filepath)

def get_ai_classification(output, output_filepath, ai_range):
    for filename in output.keys():
        for spp_code in ['weto', 'wofr']:
            infile = output.get(filename)
            manual_classified = [k for k,v in infile['transcriber_'+spp_code].items() if v != None and v != 'EIM_AI']
            below_slider = [k for k,v in infile[spp_code+'_pos'].items() if (v < (ai_range[0]-0.01)/100) and (k not in manual_classified)]
            above_slider = [k for k,v in infile[spp_code+'_pos'].items() if (v > (ai_range[1]+0.01)/100) and (k not in manual_classified)]
            between_slider = [k for k in infile[spp_code+'_pos'].keys() if (k not in manual_classified) and (k not in below_slider) and (k not in above_slider)]
            infile['transcriber_'+spp_code].update({k: "EIM_AI" for k in below_slider + above_slider})
            infile[spp_code].update({k: 0 for k in below_slider})
            infile[spp_code].update({k: 1 for k in above_slider})
            infile[spp_code].update({k: None for k in between_slider})
            infile['transcriber_'+spp_code].update({k: None for k in between_slider})
            output[filename].update(infile)

        with open(output_filepath, 'w') as fp:
            json.dump(output, fp)

    return(output)

def get_file_attr(df, user_input, selection):
    filename = df.iloc[selection[0]]['filename']
    filepath = os.path.join(user_input, filename)
    return(filename, filepath)

##### Functions #######
def get_df(audio_filenames, spp_code):
    df = pd.DataFrame(columns = ['filename', 'transcriber_'+spp_code, spp_code+'_start_time',  spp_code])
    df['filename'] = audio_filenames
    return(df)


def get_sidebar_table(output, spp_code, key):

    filenames_list = []
    transcriber_list = []
    spp_list = []

    for filename in output.keys():
        filenames_list.append(filename)

    #Spp occupancy val (1 or 0)
        if None in list(output[filename][spp_code].values()):
            spp_val = None
        else:
            spp_val = max(list(output[filename][spp_code].values()))

        if 1 in list(output[filename][spp_code].values()):
            spp_val = 1
        spp_list.append(spp_val)

    #Transcriber val (user, eim_ai, or none)
        if None in list(output[filename]['transcriber_'+spp_code].values()):
            transcriber_val = None   

        user_bool = any([isinstance(v, str) for v in list(output[filename]['transcriber_'+spp_code].values()) if v != 'EIM_AI'])
        if user_bool:
            transcriber_val = 'user'

        if (spp_val == 0 or spp_val ==1) and not user_bool:
            transcriber_val = 'EIM_AI'

        transcriber_list.append(transcriber_val)



    df=pd.DataFrame({'filename':filenames_list,
                    'transcriber_'+spp_code:transcriber_list,
                     spp_code:spp_list})

    column_configuration = {
    "filename": st.column_config.TextColumn(
        "Filename", help="The name of the file within the file folder", max_chars=100, width=None
    ),
    'transcriber_'+spp_code: st.column_config.TextColumn(
        "Transcriber", help="The name of the transcriber", max_chars=100, width=None
    ),

    "weto": st.column_config.NumberColumn(
        "Western Toad",
        help="weto",
        width=None,
    ),
    "wofr": st.column_config.NumberColumn(
        "Wood Frog",
        help="wofr",
        width=None,
    ),
    }


    sidebar_df = st.dataframe(
        df,
        column_config=column_configuration,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key = key
        #height = 550
    )

    return(df, sidebar_df)

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
            filename_location_re = re.compile(r'[Aa]-\d+?(?=_)')
            location = filename_location_re.match(filename).group(0)
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

#spec_params = model.preprocessor.pipeline.to_spec.params

st.cache_data

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

def get_filenames(user_input):
    audio_filepaths = glob.glob(os.path.join(user_input, '*.wav'))
    filename_toreview = [os.path.basename(filepath) for filepath in audio_filepaths]
    return(filename_toreview)


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


def get_sorted_keys(output, filename, spp_code):
    file_dict = dict(output[filename])
    sorted_file_scores = dict(sorted(dict(file_dict[spp_code+'_pos']).items(), key = lambda item: item[1], reverse = True))
    unclassified_samples = [key for key, value in dict(file_dict['transcriber_'+spp_code]).items() if value == None]
    sorted_keys_unclassified = [key for key in sorted_file_scores if key in unclassified_samples]
    return(sorted_keys_unclassified, file_dict)

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


def no_update_json(output, output_filepath, filename, spp_code, user_name, yesno_key):
    infile = output.get(filename)
    infile['transcriber_'+spp_code].update({str(yesno_key):user_name})
    infile[spp_code].update({str(yesno_key):0})
    output[filename].update(infile)
    with open(output_filepath, 'w') as fp:
        json.dump(output, fp)
    st.session_state.no_button = False

def yes_update_json(output, output_filepath, filename, spp_code, user_name, yesno_key):
    infile = output.get(filename)
    infile['transcriber_'+spp_code].update({str(yesno_key):user_name})
    infile[spp_code].update({str(yesno_key):1})
    output[filename].update(infile)
    with open(output_filepath, 'w') as fp:
        json.dump(output, fp)
    st.session_state.yes_button = False