import streamlit as st

def ai_slider():
    ai_slider = st.slider(label = "slider",
                        help = """
                        Set the confidence range for the A.I. Classifier\n\nThe 
                        default confidence range is from 60 to 100.\n\n \n\nSamples will be classified as _absent_ by the A.I. if 
                        their probability score is below the lower bound of the confidence range. If the score is greater than the upper bound of the confidence range, 
                        samples will be 
                        classified as _present_. \n\n \n\n**Samples with scores that are between the upper and lower confidence bounds will not be classified by the A.I.**
                        """, 
                        min_value = 0,
                        max_value = 100,
                        key = 'ai_range',
                        value =  (60, 99),
                        label_visibility = 'collapsed')

    return(ai_slider)

def text_input():
    text_input = st.text_input(label = "user_name",
                               placeholder  = "Copy and paste the folder pathway here", 
                               value = "", 
                               key = 'user_input',
                               label_visibility = 'collapsed')
    return(text_input)


def text_input_name():
    text_input = st.text_input(label = "file_path",
                               placeholder  = "Your username", 
                               value = "", 
                               key = 'user_name',
                               label_visibility = 'collapsed')
    return(text_input)


def toggle_button(spp_iswofr_dict):
        is_wofr = st.toggle(label = 'Spp.', 
                            value = False, 
                            help = 'Click to toggle between western toad and wood frog.')
        
        spp_name = spp_iswofr_dict[is_wofr]['common name']
        spp_code = spp_iswofr_dict[is_wofr]['code']
        return(is_wofr, spp_name, spp_code)