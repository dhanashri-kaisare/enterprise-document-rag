#!/bin/bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
sleep 3
streamlit run streamlit_app.py --server.port 7860 --server.address 0.0.0.0