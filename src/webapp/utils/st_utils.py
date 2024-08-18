import streamlit as st
import base64
import json

def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.
    """
    if isinstance(object_to_download, dict):
        object_to_download = json.dumps(object_to_download, indent=2)
    
    b64 = base64.b64encode(object_to_download.encode()).decode()
    
    button_uuid = st.button(button_text)
    if button_uuid:
        href = f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{button_text}</a>'
        st.markdown(href, unsafe_allow_html=True)

def display_dataframe(df, use_container_width=True):
    """
    Displays a dataframe with improved formatting.
    """
    st.dataframe(df.style.highlight_max(axis=0), use_container_width=use_container_width)

def plot_altair_chart(chart):
    """
    Plots an Altair chart with improved sizing.
    """
    st.altair_chart(chart, use_container_width=True)