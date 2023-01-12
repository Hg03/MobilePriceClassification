import streamlit as st
import spacy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer


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

def preprocess_text(feature):
#   Initializing spacy pipeline
    nlp = spacy.load('en_core_web_sm') 
    lemmatized = []
    final = []
    for i in range(len(feature)):
        lemmatized.append([token.lemma_ for token in nlp(feature.iloc[i].lower()) if token.is_stop == False and token.text.isalpha() == True])
        
    for i in range(len(lemmatized)):
        final.append(" ".join(lemmatized[i]))
        
    return final
preprocess = FunctionTransformer(preprocess_text)



def mobilepriceclassification():
    st.title('Mobile Price Classification')
    df = pd.read_csv('data/mobilepriceclassification/train.csv')
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

    model = pickle.load(open('fake_news_estimator.sav','rb'))
    st.title('Fake News Classifier')
    df1 = pd.read_csv('data/fakenewsclassification/FakeNewsNet.csv')
    csv1 = convert_df(df1)
    st.download_button(label="Download data as CSV", data=csv1, file_name='train.csv', mime='text/csv',)

    input_text = st.text_area('Type your rumoured news',placeholder='Enter some valid news')
    
    submit = st.button(label = 'Is it fake or not ??')
    if submit and input_text == '':
        st.info('Please fill the field with some text')
    elif submit and input_text != '':
        result = model.predict(pd.Series([input_text]))[0]
        if(result == 0):
            st.warning('According to the data and model trained, it is considered as fake news, please go through some articles to verify it.')
        else:    
            st.info('According to the data and model trained, it is considered as genuine news, please go through some articles to confirm it for safety purpose.')

def sentimentclassifier():
    model = pickle.load(open('sentimentestimator.sav','rb'))
    topics = {0:'world',1:'Sports',2:'Business',3:'Sci/Tech'}
    st.title('Sentiment Analysis & Prediction')
    df2 = pd.read_csv('data/sentimentclassification/train.csv')
    csv2 = convert_df(df2)
    st.download_button(label="Download data as CSV",data=csv2,file_name="train.csv",mime="text/csv")
    
    input_text = st.text_area('Type your sentiment',placeholder="Type some text related to above mentioned topics")
    submit = st.button(label="Identify the topic")
    if submit and input_text == '':
        st.info('Please fill the field with some text')
    elif submit and input_text != '':
        result = model.predict(pd.Series([input_text]))[0]
        st.warning(f'Your sentiment is related to the topic of {topics[result]}')
            
            
def aboutme():
    table = '''
                |Education|College/School|CGPA/% Secured|
                |---------|--------------|--------------|
                |B.Tech (Computer Science)|Shri Vaishnav Vidyapeeth Vishwavidyalaya|7.8 CGPA|
                |Higher Secondary Education (12th)|Angel Hearts Academy|88%|
                |Secondary Education (10th)|Angel Hearts Academy|81.8%|
            '''    
    st.title("Hii, What's Up 👋")
    st.markdown("## My name is Harish Gehlot ")
    st.markdown('🖊️ Data Science and Machine Learning Practitioner')
   
    with st.expander("🏫 Educational Background"):
        st.markdown(table)
    with st.expander("🔗 Links"):
        st.markdown(' ⇢ [Github](https://github.com/Hg03)')
        st.markdown(' ⇢ [LinkedIn](https://www.linkedin.com/in/harish-gehlot-5338a021a/)')
        st.markdown(' ⇢ [My Blog](https://hashcodenotes.hashnode.dev)')
        st.markdown(' ⇢ [Leetcode](https://leetcode.com/HarishG03/)')
    with st.expander("🧾 Certifications"):
        st.markdown(' ⇢ [Data Analysis with Python](https://www.freecodecamp.org/certification/penny03/data-analysis-with-python-v7)')
        st.markdown(' ⇢ [Machine Learning with Python](https://www.freecodecamp.org/certification/penny03/machine-learning-with-python-v7)')
        st.markdown(' [Deep Learning : Image Recognition](https://www.linkedin.com/learning/certificates/a596298c4f5f6fb2e2141126903ab39ae718eec68bc044c6af1277a196cb54ba)')
    with open("HarishGehlotUpdated.pdf", "rb") as file:
        btn=st.download_button(
        label="Look on my CV",
        data=file,
        file_name="resume.pdf",
        mime="application/octet-stream"
)

page_names_to_funcs = {
    "About Me":aboutme,
    "Mobile Price Classification": mobilepriceclassification,
    "Fake News Classifier": fakenewsclassifier,
    "Sentiment Analysis": sentimentclassifier
}
st.sidebar.title('Projects')
project = st.sidebar.selectbox('List',page_names_to_funcs.keys())
page_names_to_funcs[project]()

