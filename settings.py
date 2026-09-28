from dataclasses import dataclass
import os

@dataclass
class Settings:
    fred_key: str = ""
    ecos_key: str = ""

def load_settings(fred_key="", ecos_key=""):
    try:
        import streamlit as st
        fred_secret = st.secrets.get("FRED_API_KEY", "")
        ecos_secret = st.secrets.get("ECOS_API_KEY", "")
    except Exception:
        fred_secret, ecos_secret = "", ""
    return Settings(
        fred_key=fred_key or os.getenv("FRED_API_KEY", "") or fred_secret,
        ecos_key=ecos_key or os.getenv("ECOS_API_KEY", "") or ecos_secret,
    )
