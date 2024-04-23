import streamlit as st
import pandas as pd
import numpy as np
import mayplotlib.pyplot as plt

st.title('this is the app title')
st.header("this is the markdown")
st.subheader("the subheader")
st.caption("this is the caption")
st.code("x=2021")
st.latex(r'''a+ar^1+ar^2+ar^3''')

st.image("kk.png")
#st.audio("audio.mp3")
#st.video("video.mp4")

st.checkbox('yes')
st.button('Click')
st.radio('Pick your gender',['Male','Female'])
st.selectbox('Pick your gender',['Male','Female'])
st.multiselect('choose a planet',['Jupiter','Mars','neptune'])
st.select_slider('Pick a mark',['Bad','Good','Excellent'])
st.slider('Pick a number',0,50)


st.number_input('Pick a number',0,10)
st.text_input('Email address')
st.date_input('Travelling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')


st.sidebar.title("This is written inside the sidebar")
st.sidebar.button("Click", key="2")
st.sidebar.radio("Pick your gender",["Male","Female"], key=3)


container=st.container()
container.write("This is written inside the container")
st.write("This is written outside the container")


rand=np.random.noraml(1,2,size=20)
fig,ax=plt.subplots()
ax.hist(rand,bins=15)
st.pyplot(fig)


	
st.title('Uber pickups in NYC')
	
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
              'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
	
@st.cache
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data
	
data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache)")
	
if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)
	
st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)
	
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
	
st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
