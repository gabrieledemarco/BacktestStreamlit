import streamlit as st

# Creazione di un contenitore con un bordo
with st.container(border=1):  # Bordo rosso di 3px
    st.date_input("Scegli una data:", key="date", format="YYYY-MM-DD")

# Mostra la data selezionata
date = st.session_state.get("date", None)
if date:
    st.write(f"La data selezionata è: {date}")
