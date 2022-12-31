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

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')



def mobilepriceclassification():
    st.title('Mobile Price Classification')
    df = pd.read_csv('train.csv')
    csv = convert_df(df)

    st.download_button(label="Download data as CSV", data=csv, file_name='train.csv', mime='text/csv',)

    with st.container():
        with st.form('my-form',clear_on_submit=True):
            battery_power = st.text_input('Battery Power (in mHz)',placeholder="842, 563,1021")
            blue = st.checkbox('Does it contain bluetooth or not ?')
            clock_speed = st.text_input('Clock Speed (in milliseconds)',placeholder="2.2, 0.5 ...")
            dual_sim = st.checkbox('Is it Dual - Sim ?')
            fc = st.text_input('Front Camera Megapixels',placeholder='13,1,2,58')
            four_g = st.checkbox('Is it 4g or not ?')
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
            three_g = st.checkbox('Is 3g or not ?')
            touch_screen = st.checkbox('Is touch screen or not')
            wifi = st.checkbox('Has wifi or not ?')

            input = [battery_power,blue,clock_speed,dual_sim,fc,four_g,int_memory,m_dep,mobile_wt,n_cores,pc,px_height,px_width,ram, sc_h,sc_w,talk_time,three_g,touch_screen,wifi]
            submit = st.form_submit_button('Predict the range of price')

        if(submit and '' not in input):
            model = pickle.load(open('final_estimator.sav','rb'))
            ans = model.predict([input])[0]
            st.info(f'{ans} is the price range of a mobile')
        else:
            st.error('Please fill all the fields') 


def fakenewsclassifier():
    st.title('Fake News Classifier')

def aboutme():
    st.title("Hii, What's Up")


page_names_to_funcs = {
    "About Me":aboutme,
    "Mobile Price Classification": mobilepriceclassification,
    "Fake News Classifier": fakenewsclassifier,
}
st.sidebar.title('Projects')
project = st.sidebar.selectbox('List',page_names_to_funcs.keys())
page_names_to_funcs[project]()

