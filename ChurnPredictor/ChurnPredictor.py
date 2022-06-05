import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image
import pickle, os
from helper import encoder, standardise

##---------------READING FILES-------------------------------
# path = '/home/emmanuel/Documents/FAKER/ChurnPredictor/staticfiles'
# path = 'https://github.com/E-Hammond/ChurnPredictor/blob/main/ChurnPredictor/staticfiles'
# img_path = '/home/emmanuel/Documents/FAKER/ChurnPredictor/images'
# model_path = '/home/emmanuel/Documents/FAKER/ChurnPredictor/model'
path = 'https://github.com/E-Hammond/ChurnPredictor/blob/main/ChurnPredictor/staticfiles'
# path2 = 'https://raw.githubusercontent.com/E-Hammond/ChurnPredictor/main/ChurnPredictor/staticfiles/sample.csv'


sample = pd.read_html(os.path.join(path,'sample.csv'))[0].drop(['Unnamed: 0'],axis=1)
sample_upload = pd.read_html(os.path.join(path,'sample_upload.csv'))
# file = os.path.join(model_path,'bagging_classifier.pkl')



###--------- SET THE TITLE OF THE APP----------------------------------------------
st.title('Churn Predictor App')

### Call the encoder constructor
ordencoder = OrdinalEncoder()

## ---------SETTING TITLE AND DESCRIPTION FOR PREDICTION BOARD----------------------
st.sidebar.title('Welcome to the Prediction Scoreboard')
st.sidebar.markdown('Select a set of indicators from the list below and click on the **Predict** button to make predictions.')


## Define container
c = st.sidebar.container()


##---------- CREATING VARIABLES FOR ALL INDICATORS------------------------------------
 
### Splittng into columns
ind_left,ind_mid,ind_right = st.sidebar.columns(3)

## 1. Age
age = ind_left.selectbox('Age',['18-29','30-49','50-64','65+'], help='Select the age group of customer')

## 2. Monthly charge
charge = ind_mid.text_input('Charge', help='Enter the charge paid by customer per month')

## 3. Status
status = ind_right.selectbox('Status',['Resolved','Not Resolved'], help='Choose whether complaint has been resolved or not')

## 4. Tenure
tenure = st.sidebar.slider('Tenure',1,36,help='Slide through the numbers to select a customer\'s tenure' )


## 5. Subcriber Complaints
complaint = st.sidebar.selectbox('Complaints',['No Coverage','Slow Internet','USSD Errors',
                        'Poor Customer Service','High Call Charges','Network Interruptions',
                        'Unsolicited Subscribe Messages'], help='Select from the dropdown the complaints made by the customer')

indicators = [complaint,status,age,charge,tenure]

def predictor(indicators):

    dataframe = pd.DataFrame(indicators).T
    columns=['complaint','status','age','monthly_charges','tenure']
    dataframe.columns = columns
    dataframe['monthly_charges'] = dataframe['monthly_charges'].apply(lambda x: float(x))
    dataframe = encoder(dataframe)
    dataframe = pd.concat([dataframe,sample]).reset_index().drop(['index'], axis=1)
    drop_index = range(1,len(dataframe))
    dataframe = standardise(dataframe).drop(drop_index)
    model_file = open(file,'rb')
    model = pickle.load(model_file)
    model_file.close()
    pred = model.predict(dataframe)
    confindence = np.round(model.predict_proba(dataframe)*100, 0)
    prediction = int(pred)
    if prediction == 0:
        return st.sidebar.write(f'The customer will **RETAIN** with **{confindence[0][0]}%** confidence.')
    elif prediction == 1:
        return st.sidebar.write(f'The customer will **CHURN** with **{confindence[0][1]}%** confidence.')

### PREDICT BUTTON
l,m,r = st.sidebar.columns(3)
if l.button('Predict'):
    try:
        predictor(indicators)
    except:
        st.sidebar.write('**The Charge field should be Numerical and not Empty!**')

## ACCURACY METRIC
r.metric('Accuracy','83%')




###------------------------ WRITING THE CONTENT OF THE BODY OF THE PREDICTOR-------------------------
# st.markdown('''
#            The churn predictor app is a tool used for making predictions based on a predefined set of indicators.
#            The impact of the indicators on customer churn rate is shown below :
#           ''')

###---------------------------- ADDING AN IMAGE TO THE PAGE--------------------------------------------

### FIGURE 1
# img = Image.open(os.path.join(img_path,'indicators.png'))
im_l,im_m,im_r = st.columns(3)
# im_l.image(img, caption='Figure 1: Indicator Importance', width=420)
#------------------------------------------------------------------------------------------------------
#st.write('***')
#st.markdown('''
#            The performance of the predictor is evaluated based on the its accuracy and misclassification rate using a confusion matrix (This is shown below).
#            ''')

### FIGURE 2
# img = Image.open(os.path.join(img_path,'confusion.png'))
im_l,im_m,im_r = st.columns(3)
# im_l.image(img, caption='Figure 2: Confusion Matrix', width=330)

#----------------------------------------------------------------------------------------------
# st.write('***')
## ADDING A FILE UPLOADER
st.subheader('Upload a CSV file and make predictions')
st.markdown('**Hint 1 :** Check the sample CSV file box below to view the format of the CSV file to be uploaded.')
st.markdown('**Hint 2 :** Click on the download button to download predictions.')


####----------------------- FILE UPLOADER CONTAINER--------------------------------------------
upload = st.file_uploader('')
box_l,box_m,box_r = st.columns(3)

### Encoding samples for multiple predictions
def encode_sample(dataframe):

    dataframe = dataframe.replace(['Yes','No','resolved','not resolved'],[1,0,1,0])
    dataframe['age'] = ordencoder.fit_transform(dataframe[['age']])
    dataframe['complaint'] = ordencoder.fit_transform(dataframe[['complaint']])
    
    return dataframe

###------------- A LOGIC TO RECEIVE AND PREDICT FROM A CSV FILE-------------------------------
def predictor_sample(upload):
    
    if upload:
        try:
            uploaded_file = pd.read_csv(upload)
            uploaded_file = encode_sample(uploaded_file)
            uploaded_file = standardise(uploaded_file)
            model_file = open(file,'rb')
            model = pickle.load(model_file)
            model_file.close()
            pred = model.predict(uploaded_file)
            pred = pd.DataFrame(pred,columns=['Predictions'])
            save_pred = pred.to_csv(os.path.join(path,'prediction.csv'), index=False)
        except:
            st.write('')

        return (pred,save_pred)
    
#### SAMPLE CHECKBOX
if box_l.checkbox('Sample CSV file'):
    st.write(sample_upload)

#### SHOW PREDICTION CHECKBOX
try:
    if box_m.checkbox('Show Predictions'):
        st.write(predictor_sample(upload)[0])
except:
    st.write('**Upload a correct file before checking this box!**')


##### DOWNLOAD BUTTON
# with open(os.path.join(path,'prediction.csv')) as f:
#     st.download_button('Download',f)



st.markdown('**Note :** **<<<<<** *You can make single predictions on the scoreboard on the sidebar.* **<<<<<**')
#----------------------------------------------------------------------------------
st.write('***')
