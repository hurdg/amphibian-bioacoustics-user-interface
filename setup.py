SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

import streamlit as streamlit

import streamlit.web.cli as stcli
import os, sys

# Import the other libraries you need here
import streamlit_extras as streamlit_extras
import pandas as pandas
import opensoundscape as opensoundscape
import plotly as plotly

import re
import json
import pickle
import glob


def resolve_path(path):
    resolved_path = os.path.abspath(os.path.join(os.getcwd(), path))
    return resolved_path

if __name__ == "__main__":
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("amphib_classifier.py"),
        "--global.developmentMode=false",
    ]
    sys.exit(stcli.main())
