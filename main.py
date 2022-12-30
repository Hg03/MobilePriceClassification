import streamlit as st
import pandas as pd
import pickle

df = pd.read_csv('train.csv')
columns = list(df.columns)
oneto8 = [1,2,3,4,5,6,7,8]
onetotwenty = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

st.set_page_config(layout="wide")
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.title('Mobile Price Classification')

st.sidebar.header('About Me')
st.sidebar.markdown('### Myself Harish Gehlot, and I am machine learning practitioner. If you like my effort, connect with me')
st.sidebar.markdown('[Github](https://github.com/Hg03)')
st.sidebar.markdown('[LinkedIn](https://www.linkedin.com/in/harish-gehlot-5338a021a/)')

with st.container():
    with st.form('my-form',clear_on_submit=False):
        battery_power = st.text_input('Battery Power (in mHz)',placeholder="842, 563,1021")
        blue = st.select_slider('Does it contain bluetooth or not ?',['No','Yes'])
        blue = 1 if blue == 'Yes' else 0
        clock_speed = st.text_input('Clock Speed (in milliseconds)',placeholder="2.2, 0.5 ...")
        dual_sim = st.select_slider('Is it Dual - Sim ?',['No','Yes'])
        dual_sim = 1 if dual_sim == 'Yes' else 0
        fc = st.text_input('Front Camera Megapixels',placeholder='13,1,2,58')
        four_g = st.select_slider('Is it 4g or not ?',['No','Yes'])
        four_g = 1 if four_g == 'Yes' else 0
        int_memory = st.text_input('Internal memory in Gigabytes',placeholder="64, 128")
        m_dep = st.text_input('Mobile depth in cm',placeholder="Measure it or check the box of phone")
        mobile_wt = st.text_input('Weight of Mobile Phone (in Kg)',placeholder="Measure it or check the box of phone")
        n_cores = st.number_input('Number of cores of processor',min_value=1,max_value=8,)
        pc = st.text_input('Primary Camera Megapixels',placeholder="13, 1, 2, 58")
        px_height = st.text_input('Pixel Resolution Height',placeholder="Measure it or check the box of phone")
        px_width = st.text_input('Pixel resolution width',placeholder="Measure it or check the box of phone")
        ram = st.text_input('Random Access Memory in Megabytes',placeholder="Check out settings in phone")
        sc_h = st.text_input('Screen height of mobile in cm',placeholder="Measure it or check the box of phone")
        sc_w = st.text_input('Screen width of mobile in cm',placeholder="Measure it or check the box of phone")
        talk_time = st.number_input('Longest battery charge will last',min_value=1,max_value=20)
        three_g = st.select_slider('Is 3g or not ?',['No','Yes'])
        three_g = 1 if three_g == 'Yes' else 0
        touch_screen = st.select_slider('Is touch screen or not',['No','Yes'])
        touch_screen = 1 if touch_screen == 'Yes' else 0
        wifi = st.select_slider('Has wifi or not ?',['No','Yes'])
        wifi = 1 if wifi == 'Yes' else 0

        input = [battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram, sc_h,sc_w,talk_time,three_g,touch_screen,wifi]

        submit = st.form_submit_button('Predict the range of price')

    if(submit):
        model = pickle.load(open('final_estimator.sav','rb'))
        ans = model.predict([input])[0]
        st.info(ans)
        




