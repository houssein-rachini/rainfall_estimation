import ee
import streamlit as st
from google.oauth2 import service_account
from streamlit.errors import StreamlitSecretNotFoundError


@st.cache_resource(show_spinner=False)
def initialize_earth_engine():
    """Initialize Earth Engine for Streamlit Cloud (secrets) or local auth."""
    try:
        service_account_info = dict(st.secrets["google_ee"])
    except (StreamlitSecretNotFoundError, KeyError, FileNotFoundError):
        service_account_info = None

    if service_account_info:
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/earthengine"],
        )
        ee.Initialize(credentials)
        return

    ee.Initialize()
