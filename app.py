import math
import pickle
import numpy as np
import streamlit as st

pipe = pickle.load(open('pipe.pkl','rb'))
df =  pickle.load(open('df.pkl','rb'))

st.set_page_config("Laptope Price.com","images.jpg")
col1,col2 = st.columns(2)
with col1:
    col1.title("Laptop Predictor")
with col2:
    col2.image("images.jpg",width=100)

company = st.selectbox('Brand',df['Company'].unique())
type = st.selectbox('Type',df['TypeName'].unique())
ram = st.selectbox('RAM(in GB',[2,4,6,8,12,16,24,32,64])
weight = st.number_input('Weight of the laptope')
touchscreen = st.selectbox('Touchscreen',['No','Yes'])
ips = st.selectbox('IPS',['No','Yes'])
screen_size = st.number_input('Screen Size', min_value=1.0)

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2560x1600','2560x1440','2304x1440'])

cpu = st.selectbox('CPU',df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB',[128,256,512,1024,2048])
ssd = st.selectbox('SSD(in GB',[8,128,256,512,1024])
gpu = st.selectbox('GPU',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())
if st.button('Predict Price'):
    #query
    ppi = None
    if touchscreen == 'yes':
        touchscreen = 1
    else:
        touchscreen = 0
    if ips == 'yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res**2)+(Y_res**2))**0.5/screen_size

    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    query = query.reshape(1,12)
    st.write(f"<h3>The predicted Price  { str(int(np.exp(pipe.predict(query)[0])))}</h3>",unsafe_allow_html=True)


