from modeller.duygusal_denge import DuygusalDengeModel
from modeller.dısadonukluluk import DisadonuklulukModel
from modeller.ozdenemetimsorumluluk import OzdenetimSorumlulukModel
from profilescrapper import TwitterUserScraper
import streamlit as st
import pandas as pd
import time
import plotly.express as px

st.title("Twitter Personality analyzer")
user_name = st.text_input("Enter a twitter username")
if user_name:
    if user_name.startswith('@'):
        user_name = user_name[1:]


    @st.cache(allow_output_mutation=True, suppress_st_warning=True)
    def Data_Loading():
        data = TwitterUserScraper(user_name).get_tweets_df()
        print("done")
        data["duygusaldenge"] = data[["text"]].apply(
            lambda x: DuygusalDengeModel.EndiseyeYatkinlikVeKendineGuven(x["text"]).predict(), axis=1)
        print("done1")
        data["duygsaldemge_label"] = data["duygusaldenge"].apply(
            lambda x: "kendinegüven" if x == 1 else "endişeye yatkınlık")
        print("done2")

        data["canlılık_dısadonukluk"] = data[["text"]].apply(
            lambda x: DisadonuklulukModel.CanlilikModel(x["text"]).predict(), axis=1)
        print("done3")
        data["canlılık"] = data["canlılık_dısadonukluk"].apply(lambda x: "canlı" if x == 1 else "None")
        print("done4")
        data["içedönüklülük_vs_girişkenlik_dışadönüklülük"] = data[["text"]].apply(
            lambda x: DisadonuklulukModel.IcedonuklukModelveGiriskenlikModel(x["text"]).predict(), axis=1)
        print("done5")
        data["içedönüklülük_vs_girişkenlik"] = data["içedönüklülük_vs_girişkenlik_dışadönüklülük"].apply(
            lambda x: "girişkenlik" if x == 1 else "içedönüklülük")
        print("done6")

        data["düzenlilik_ozdenetim"] = data[["text"]].apply(
            lambda x: OzdenetimSorumlulukModel.Duzenlilik(x["text"]).predict(), axis=1)
        print("done7")
        data["düzenlilik"] = data["düzenlilik_ozdenetim"].apply(lambda x: "düzenlilik" if x == 1 else "None")
        print("done8")
        data["sorumluluk_ozdenetim"] = data[["text"]].apply(
            lambda x: OzdenetimSorumlulukModel.Sorumluluk(x["text"]).predict(), axis=1)
        print("done9")
        data["sorumluluk"] = data["sorumluluk_ozdenetim"].apply(lambda x: "sorumluluk" if x == 1 else "None")
        print("done10")
        data["kurallarabağlılıkvsheyecanarama_özdenemetimsorumluluk"] = data[["text"]].apply(
            lambda x: OzdenetimSorumlulukModel.KurallarabaglilikVeHeyecanarama(x["text"]).predict(), axis=1)
        data["kurallarabağlılıkvsheyecanarama"] = data["kurallarabağlılıkvsheyecanarama_özdenemetimsorumluluk"].apply(
            lambda x: "kurallarabağlılık" if x == 1 else "heyecanarama")
        print("done")
        return data
    data_load_state = st.text('Loading data...')
    df = Data_Loading()
    time.sleep(1)
    if df["text"].count() == 0:
        data_load_state.error('User not found please enter a valid user name')
        st.stop()
    data_load_state.text("Magic is done!")

    if st.checkbox('Show raw data'):
        st.subheader('Raw data')
        st.dataframe(df.drop("quoted_tweet", axis=1))