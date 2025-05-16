#######################################################################################
######################## Initiate Environment #########################################
#######################################################################################
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

import os
import re
import json

#CUstom functions

from utils import get_filenames, get_sidebar_table, get_spectrogram, get_scores, reset_file_classifications, make_occupancy_df, get_ai_classification, get_file_attr, get_sorted_keys, get_event_index, get_title_markdown, no_update_json, yes_update_json
from sidebar_functions import ai_slider, text_input, toggle_button, text_input_name
from plotly_grid import plotly_grid


#######################################################################################
######################## Inital Config (Prompts) ######################################
#######################################################################################

#Create streamlit app
st.set_page_config(layout = 'wide', 
                   page_title = 'Amphibian Classifier', 
                   page_icon=":frog:", 
                   menu_items={'Report a bug': "mailto:gavin.hurd@pc.gc.ca"})

#Define species indexing/labeling variable
spp_iswofr_dict = {False:{"common name":"Western Toad", 'code': 'weto'},
                    True: {"common name": 'Wood Frog', 'code': 'wofr'}}

if 'samp_key' not in st.session_state: st.session_state.samp_key = "0"
if 'plotly_grid' not in st.session_state: st.session_state.plotly_grid = {}
if 'old_selection' not in st.session_state: st.session_state.old_selection = [0]
if 'selection' not in st.session_state: st.session_state.selection = [0]
if 'ai_range' not in st.session_state: st.session_state.ai_range = [60,100]
if 'no_button' not in st.session_state: st.session_state.no_button = False
if 'yes_button' not in st.session_state: st.session_state.yes_button = False
if 'auto_filer' not in st.session_state: st.session_state.auto_filer = False

with st.sidebar:
    top_left, top_right = st.columns([3, 1.2], vertical_alignment = 'bottom')
    #Toggle button for spp., create objects
    with top_right:
        is_wofr, spp_name, spp_code = toggle_button(spp_iswofr_dict)
        user_name = text_input_name()

    #Classifier title on left
    with top_left:
        st.header(f"{spp_name} Classifier", divider = True)
        user_input = text_input()

    #Stop execution until folder pathway has been specified
if user_input == "" and user_name == "":
    st.info('Please input a filepath and your Wildtrax username')
    st.stop()

if user_input == "" and user_name != "":
    st.info('Please input a filepath in the sidebar')
    st.stop()

if user_input != "" and user_name == "":
    st.info('Please input your Wildtrax username in the sidebar')
    st.stop()


#######################################################################################
######################## Get Base Data ################################################

#Create list of audio filepaths
audio_filenames = get_filenames(user_input)

#get ai classification scores
get_scores(user_input, audio_filenames)

#Auto classify the values beyond this range
get_ai_classification(audio_filenames, st.session_state.ai_range, spp_code, user_input, user_name) #Does not write over manually classified files


#######################################################################################
######################## Audio FILE Config ############################################

#Sidebar df
with st.sidebar:
    df, sidebar_df = get_sidebar_table(audio_filenames, spp_code, key = 'sidebar_df', user_input = user_input, selection = st.session_state.selection, user_name = user_name, auto_filer = st.session_state.auto_filer)

#Store the index of the row that has been selected, or, if no selection, default to the first row
if bool(sidebar_df.selection.rows): #i.e. if a seletion has been made in the sidebar
    selection = sidebar_df.selection.rows
    st.session_state.selection = selection
else: 
    selection = st.session_state.selection #default is zero at initialization (i.e. first row)

#If a new selection is made, reset all intrafile loggers
if selection != st.session_state.old_selection:
    st.session_state.old_selection = selection
    st.session_state.auto_samp_i = 0

#Extract filename of selected row, and add user_input to get filepath
filename, filepath = get_file_attr(df, user_input, selection)

#Display filename in app
null, title_col, null = st.columns([3,5,1], vertical_alignment = 'center')

with title_col:
    get_title_markdown(filename)

#######################################################################################
######################## File SAMPLES Config ##########################################

#Extract dict of all 3 second clips within selected file ('file_dict')
#Sort scores of unclassified samples descendingly (i.e. most likely clip is at top;'sorted_keys_unclassified)

sorted_keys_unclassified, file_dict = get_sorted_keys(filename, spp_code, user_input)

#If there has been a selection in the plotly grid (sample navigator) then update the samp_key and the yesno_key with the selection value
    #Samp_key is used to draw the black bounding box around the selected cell in the sample navigator
    #yesno_key is used to update the json file with any user classification decisions
        #NOTE although the samp_key may be altered when a non-selection prompts an auto-update, 
        # the yesno_key will not be re-aligned until AFTER the JSON has been written 
    #The auto_samp_i is used to keep track of the iteration number of sequential auto-updates of the samp_key value
    #The auto_samp_i is reset to 0 whenever a selection is made in the sample navigator, or when a new FILE is selected

try: #Proceeds if a selection has been made in the sample navigator 
    st.session_state.yesno_key = st.session_state.samp_key = str(get_event_index(st.session_state.plotly_grid))
    

#If there has not been a selection in the sample navigator, then the samp_key (e.g. the sample that is selected) will: 
    #1) try to default to the UNCLASSIFED sample with the HIGHEST score
    #2) If no unclassified samples ramin, the samp key will be set at a value of zero (e.g. the chronologically first sample)

except:
    if sorted_keys_unclassified:
        st.session_state.samp_key = sorted_keys_unclassified[0]
    else:
        st.session_state.samp_key = 0

#Create object for simplicity
samp_key = st.session_state.samp_key


#######################################################################################
######################## Figure Config ################################################


#Specify layout - column spacing
null, middle_left,  null, middle_right = st.columns([0.5,9,0.5, 6], vertical_alignment = 'top')


#Create slider to store A.I. confidence bounds
with middle_right:
    left, mid, right = st.columns([28,2,54], vertical_alignment = 'center')
    with left:
        st.markdown('A.I. classifier Confidence Range')
    with mid:
        st.markdown('→')
    with right:
        slider = ai_slider()


    #Employ function to create sample navigator grid
    #Function internally determines AI classification using the slider values (simpler than pulling values from JSON?)
    grid = plotly_grid(file_dict, spp_code, spp_name, st.session_state.ai_range, samp_key)

    #Upon a user selection, the app is re-run and the user selection is stored in the 'plotly_grid' session_state object
    grid_event = st.plotly_chart(grid, key="plotly_grid", on_select="rerun", selection_mode = 'points', config = {'displayModeBar': False})

    #Display selection info below sample navigator
    null, bot_left,  bot_right = st.columns([0.5,2,3], vertical_alignment = 'top')
    with bot_left: st.text(f"Sample: {samp_key}")
    with bot_right:
        samp_score = dict(file_dict[spp_code + '_pos'])[int(samp_key)]
        st.text(f"A.I. Confidence: {round(samp_score*100)}")

 
#Create spectorgram
start_time = file_dict['start_time'][int(samp_key)]
end_time = start_time + 3
spectrogram_object = get_spectrogram(filepath, start_time, end_time)

#plot spectrogram and audio
with middle_left:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(spectrogram_object.plot())
    st.audio(filepath,start_time=start_time,end_time = end_time, loop=True, autoplay = True)


#Create buttons for 'yes' and 'no' manual classifications
#Upon the user clicking the yes or no button, a corresponding boolean session_state object is set to True, which triggers the above pathway that writes to a JSON
    null, col1, mid, col2 = st.columns([1,1.5,3,2], vertical_alignment = 'center')

    def noClickFunction():
        no_update_json(filename, spp_code, user_name, st.session_state.yesno_key, user_input)
        st.session_state.yesno_key = samp_key

    def yesClickFunction():
        yes_update_json(filename, spp_code, user_name, st.session_state.yesno_key, user_input)
        st.session_state.yesno_key = samp_key

    with col1:
        with stylable_container(
            "red",
            css_styles="""
            button {
                background-color: #FF0000;
                color: black;
            }""",
        ):
            no_button = st.button("No", key="No", on_click = noClickFunction)

    with mid:
        st.write(f"**{spp_name}** detected?")
    with col2:
        with stylable_container(
            "green",
            css_styles="""
            button {
                background-color: #00FF00;
                color: black;
            }""",
        ):
            yes_button = st.button("Yes", key="Yes", on_click = yesClickFunction)

    #Update classified csv with manual selection

st.session_state.yesno_key = samp_key

with st.sidebar:
    left, mid, right = st.columns([1,1,1], vertical_alignment = 'bottom')
    with left:
        st.session_state.auto_filer = st.toggle(label = 'auto file', 
                    value = False, 
                    help = 'Click to enable auto transitions after a file is fully classified',
                    label_visibility = 'collapsed')
        with stylable_container(
        "grey",
        css_styles="""
        button {
            background-color: #b3afa6;
            color: black;
        }""",
        ):
            wildtrax_button = st.button("Export Wildtrax CSV", key="ExportWildtrax")
    
    with mid:
        with stylable_container(
        "lightgrey",
        css_styles="""
        button {
            background-color: #b3afa6;
            color: black;
        }""",
        ):
            occupancy_button = st.button("Export Occupancy CSV", key="ExportOccupancy")
    if occupancy_button:
        weto_df, wofr_df = make_occupancy_df(audio_filenames)
        weto_df.to_csv(os.path.join(user_input, 'weto_occupancy.csv'), index = True, na_rep='NA')
        wofr_df.to_csv(os.path.join(user_input, 'wofr_occupancy.csv'), index = True, na_rep='NA')

    with right:
        def reset_button():
            st.session_state.reset_table = False
            reset_file_classifications(spp_code, filename, user_input)

        with stylable_container(
        "lightgrey",
        css_styles="""
        button {
            background-color: #b3afa6;
            color: black;
        }""",
        ):
            st.session_state.reset_table = st.button("Reset Classifications", 
                        key="ResetClass", on_click = reset_button)


    