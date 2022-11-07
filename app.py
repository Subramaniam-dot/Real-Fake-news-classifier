import streamlit as st
import joblib,os

#import spacy



#vectorizer
news_vectorizer = open('news_vectorizer.sav','rb')
news_cv = joblib.load(news_vectorizer)

#Load our models
def load_prediction_models(model_file):
    loaded_models = joblib.load(open(os.path.join(model_file),'rb'))
    return loaded_models


def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val == value:
            return key

def main():
    """News Classifier App"""
    st.title("News Classifier App")
    st.subheader("Machine Learning App using Logistic Regression Model")

    activities = ["Prediction"]

    choice = st.sidebar.selectbox("Select",activities)

    if choice == 'Prediction':
        st.info('Prediction with ML')

        news_text = st.text_area("Enter Text","Type Here")
        all_ml_model = ['Logistic Regression']
        model_choice = st.selectbox('Choose model',all_ml_model)
        prediction_labels = {'Real':0 , 'Fake':1}
        if st.button('classify'):
            vect_text = news_cv.transform([news_text]).toarray()
            if model_choice == 'Logistic Regression':
                predictor = load_prediction_models('trained_model.sav')
                prediction = predictor.predict(vect_text)
                st.write(prediction)
                final_results = get_keys(prediction,prediction_labels)
                st.success(final_results)
           

if __name__ =='__main__':
 main()
