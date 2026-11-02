import os
import json
import time
import re
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf
import requests
from supabase import create_client


# ============================================================
# 1. CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

# NSE official holiday endpoint (cash/equities segment = CM).
NSE_HOME_URL = "https://www.nseindia.com/"
NSE_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master"
_NSE_HOLIDAY_CACHE = {}

GEMINI_MODEL = "gemini-3.5-flash-lite"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

DAILY_BUDGET = 10000.0
MAX_PER_STOCK = 2500.0
MAX_NEW_BUYS_PER_DAY = 2
STOP_LOSS_PERCENT = 0.05
MIN_BUY_SCORE = 70

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
NO_NEW_BUYS_AFTER = dt_time(15, 0)
SQUARE_OFF_AFTER = dt_time(15, 10)

MAX_DATA_AGE_MINUTES = 30
MAX_STOCKS = 20
MIN_CANDLES = 10

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

required = {
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SECRET_KEY": SUPABASE_KEY,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

missing = [key for key, value in required.items() if not value]

if missing:
    raise RuntimeError(
        "Missing required environment variables: "
        + ", ".join(missing)
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ============================================================
# 2. TIME AND TELEGRAM
# ============================================================

def now_ist():
    return datetime.now(IST)


def today_ist():
    return now_ist().date()


def fetch_nse_holidays(year):
    """
    Fetch official NSE cash-market (CM) trading holidays.
    NSE's web endpoint is not a documented public API and may change.
    Any fetch/parse failure raises an exception (fail closed).
    Results are cached for this worker process.
    """
    year = int(year)

    if year in _NSE_HOLIDAY_CACHE:
        return _NSE_HOLIDAY_CACHE[year]

    session = requests.Session()

    headers = {
