################################################################################
#                            1. INITIATE ENVIRONMENT                           #
################################################################################
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os, re, json

# Custom functions
from utils import (
    get_filenames, get_scores, get_ai_classification, get_sidebar_table, 
    no_update_json, yes_update_json, get_sorted_keys, 
    reset_file_classifications, make_occupancy_df, make_wildtrax_df,
    get_file_attr, get_spectrogram, get_event_index, get_title_markdown
)
from sidebar_functions import (
    weto_ai_slider, wofr_ai_slider, text_input, toggle_button, text_input_name
)
from plotly_grid import plotly_grid


################################################################################
#                             2. INITIAL CONFIGURATION                         #
################################################################################

st.set_page_config(
    layout='wide',
    page_title='Amphibian Classifier',
    page_icon=":frog:",
    menu_items={'Report a bug': "mailto:gavin.hurd@pc.gc.ca"}
)

# Species label map (used for toggling between species)
spp_iswofr_dict = {
    False: {"common name": "Western Toad", 'code': 'weto'},
    True:  {"common name": 'Wood Frog', 'code': 'wofr'}
}

# Initialize session state
default_states = {
    'samp_key': "0",                          # Current selected sample key (string)
    'plotly_grid': {},                        # Stores plotly grid selection
    'old_selection': [0],                     # Previous file selection
    'selection': [0],                         # Current file selection
    'weto_ai_range': [60, 100],               # Confidence range for Western Toad
    'wofr_ai_range': [60, 100],               # Confidence range for Wood Frog
    'no_button': False,                       # Tracks "No" classification button click
    'yes_button': False,                      # Tracks "Yes" classification button click
    'auto_filer': False                       # Enables auto-switching after file fully classified
}
for key, val in default_states.items():
    if key not in st.session_state:
        st.session_state[key] = val

ai_range_dict = {
    'weto': st.session_state.weto_ai_range,
    'wofr': st.session_state.wofr_ai_range
}

# Sidebar input
with st.sidebar:
    top_left, top_right = st.columns([3, 1.2], vertical_alignment='bottom')
    with top_right:
        is_wofr, spp_name, spp_code = toggle_button(spp_iswofr_dict)
        user_name = text_input_name()
    with top_left:
        st.header(f"{spp_name} Classifier", divider=True)
        user_input = text_input()

# Basic validation
# Require both folder path and user name before proceeding
if not user_input and not user_name:
    st.info('Please input a filepath and your Wildtrax username')
    st.stop()
elif not user_input:
    st.info('Please input a filepath in the sidebar')
    st.stop()
elif not user_name:
    st.info('Please input your Wildtrax username in the sidebar')
    st.stop()


################################################################################
#                            3. LOAD & CLASSIFY AUDIO                          #
################################################################################

audio_filenames = get_filenames(user_input)
get_scores(user_input, audio_filenames, ai_range_dict)
get_ai_classification(audio_filenames, ai_range_dict, spp_code, user_input, user_name)

################################################################################
#                            4. FILE SELECTION LOGIC                           #
################################################################################

with st.sidebar:
    df, sidebar_df = get_sidebar_table(
        audio_filenames, spp_code, key='sidebar_df',
        user_input=user_input,
        selection=st.session_state.selection,
        user_name=user_name,
        auto_filer=st.session_state.auto_filer
    )

if sidebar_df.selection.rows:
    selection = sidebar_df.selection.rows
    st.session_state.selection = selection
else:
    selection = st.session_state.selection

if selection != st.session_state.old_selection:
    st.session_state.old_selection = selection
    st.session_state.auto_samp_i = 0

filename, filepath = get_file_attr(df, user_input, selection)

# Title display
_, title_col, _ = st.columns([3, 5, 1])
with title_col:
    get_title_markdown(filename)

################################################################################
#                        5. GET SAMPLE (CLIP) SELECTION                        #
################################################################################

# Get all 3s samples and sort unclassified ones by AI confidence
sorted_keys_unclassified, file_dict = get_sorted_keys(filename, spp_code, user_input)

# Try to update selected sample from user plotly interaction
try:
    st.session_state.yesno_key = st.session_state.samp_key = str(
        get_event_index(st.session_state.plotly_grid)
    )
except:
    # If no user click, default to top unclassified or first sample
    if sorted_keys_unclassified:
        st.session_state.samp_key = sorted_keys_unclassified[0]
    else:
        st.session_state.samp_key = 0

# Store selected sample key
samp_key = st.session_state.samp_key


################################################################################
#                      6. PLOTLY NAVIGATOR + AI SLIDER                         #
################################################################################

# Layout for classifier controls
_, middle_left, _, middle_right = st.columns([0.5, 9, 0.5, 6], vertical_alignment='top')

with middle_right:
    left, mid, right = st.columns([28,2,54], vertical_alignment = 'center')
    with left:
        st.markdown('A.I. classifier Confidence Range')
    with mid:
        st.markdown('→')
    with right:
        print(f"weto = {st.session_state.weto_ai_range} / wofr = {st.session_state.wofr_ai_range}")
        if spp_code == 'weto':
            weto_slider = weto_ai_slider(st.session_state.weto_ai_range)
            st.session_state.wofr_ai_range = st.session_state.wofr_ai_range
        else:
            wofr_slider = wofr_ai_slider(st.session_state.wofr_ai_range)
            st.session_state.weto_ai_range = st.session_state.weto_ai_range


    # Plot grid of sample predictions
    grid = plotly_grid(file_dict, spp_code, spp_name, ai_range_dict, samp_key)

    # Rerun app on click; stores selection in session state
    st.plotly_chart(grid, key="plotly_grid", on_select="rerun", selection_mode='points', config={'displayModeBar': False})

    # Show current sample and AI score
    _, bot_left, bot_right = st.columns([0.5, 2, 3])
    with bot_left:
        st.text(f"Sample: {samp_key}")
    with bot_right:
        samp_score = dict(file_dict[spp_code + '_pos'])[int(samp_key)]
        st.text(f"A.I. Confidence: {round(samp_score * 100)}")

 
################################################################################
#                          7. SPECTROGRAM + AUDIO                              #
################################################################################

# Define 3s window for selected sample
start_time = file_dict['start_time'][int(samp_key)]
end_time = start_time + 3

# Generate spectrogram object for current clip
spectrogram_object = get_spectrogram(filepath, start_time, end_time)

# Show spectrogram and audio playback
with middle_left:
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(spectrogram_object.plot())  # Plot waveform/spectrogram
    st.audio(filepath, start_time=start_time, end_time=end_time, loop=True, autoplay=True)


################################################################################
#                     8. YES / NO MANUAL CLASSIFICATION                        #
################################################################################

    # Horizontal layout for classification buttons
    null, col1, mid, col2 = st.columns([1,1.5,3,2], vertical_alignment = 'center')

    # Manual classification logic
    def noClickFunction():
        no_update_json(filename, spp_code, user_name, st.session_state.yesno_key, user_input)
        st.session_state.yesno_key = samp_key

    def yesClickFunction():
        yes_update_json(filename, spp_code, user_name, st.session_state.yesno_key, user_input)
        st.session_state.yesno_key = samp_key

    # Render NO and YES buttons
    with col1:
        with stylable_container("red", css_styles="button { background-color: #FF0000; color: black; }"):
            st.button("No", key="No", on_click=noClickFunction)
    with mid:
        st.write(f"**{spp_name}** detected?")
    with col2:
        with stylable_container("green", css_styles="button { background-color: #00FF00; color: black; }"):
            st.button("Yes", key="Yes", on_click=yesClickFunction)

################################################################################
#                        9. SIDEBAR: EXPORT + RESET                            #
################################################################################

# Always keep current sample key aligned
st.session_state.yesno_key = samp_key

with st.sidebar:
    # Toggle to auto-sort when file is fully reviewed
    st.session_state.auto_filer = st.toggle(
        label='Sort unclassified files',
        value=False,
        help='Click to enable auto transitions after a file is fully classified'
    )

    def reset_button():
        st.session_state.reset_table = False
        reset_file_classifications(spp_code, filename, user_input, user_name)

    left, mid, right = st.columns([1, 1, 1])

    # Reset button to clear classifications for current file
    with left:
        with stylable_container("lightgrey", css_styles="button { background-color: #b3afa6; color: black; }"):
            st.session_state.reset_table = st.button("Reset Classifications", key="ResetClass", on_click=reset_button)

    # Export occupancy CSVs
    with mid:
        with stylable_container("lightgrey", css_styles="button { background-color: #b3afa6; color: black; }"):
            occupancy_button = st.button("Export Occupancy CSV", key="ExportOccupancy")
    if occupancy_button:
        weto_df, wofr_df = make_occupancy_df(audio_filenames, user_input)
        weto_df.to_csv(os.path.join(user_input, 'weto_occupancy.csv'), index=True, na_rep='NA')
        wofr_df.to_csv(os.path.join(user_input, 'wofr_occupancy.csv'), index=True, na_rep='NA')

    # Export Wildtrax-formatted tag file
    with right:
        with stylable_container("grey", css_styles="button { background-color: #b3afa6; color: black; }"):
            wildtrax_button = st.button("Export Wildtrax \n\n CSV", key="ExportWildtrax")
    if wildtrax_button:
        wildtrax_df = make_wildtrax_df(audio_filenames, user_input)
        wildtrax_df.to_csv(os.path.join(user_input, 'wildtrax_tags.csv'), index=True, na_rep='NA')