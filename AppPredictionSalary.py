# conda
# quit()
# select default shell --> Reload Window
# conda create -n enviro python=3.7
# conda info --envs
# activate enviro
# conda list

# cd C:\Users\HP\OneDrive\Documents\Python Anaconda\Streamlit_Salary_App
# streamlit run AppPredictionSalary.py
# conda env remove -n test
# ctr+shift+p
# conda list -e > requirements.txt

import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def main():

    st.title("IT Developer Salary Prediction")

    # Image
    from PIL import Image
    image = Image.open('./Salary.jpg')
    st.image(image, caption='Salary Prediction with AI', use_column_width=True)

    # Info
    st.write("""### Enter information to predict the salary:""")

    countries = ('Australia', 'Austria', 'Belgium', 'Brazil', 'Canada',
       'Czech Republic', 'Denmark', 'France', 'Germany', 'India', 'Iran',
       'Ireland', 'Israel', 'Italy', 'Mexico', 'Netherlands', 'Norway',
       'Pakistan', 'Poland', 'Portugal', 'Romania', 'Russian Federation',
       'South Africa', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
       'Ukraine', 'United Kingdom', 'United States')

    education = ('Bachelor’s degree', 'Master’s degree', 'Less than a Bachelors',
       'Post grad')

    default_country = countries.index('Czech Republic')
    country = st.selectbox("Country", countries, index=default_country)

    default_education = education.index('Master’s degree')
    education = st.selectbox("Education", education, index=default_education)

    experience = st.slider("Years of Experience", 0, 50, 2)

    pretty_result = {"Country": country, "Education": education, "Experience": experience}
    st.json(pretty_result)

    st.subheader("Run the model:")

    predict = st.button("Predict")
    if predict:
        X_sample = np.array([[country, education, experience]])
        X_sample[:,0] = le_country.transform(X_sample[:,0])
        X_sample[:,1] = le_education.transform(X_sample[:,1])
        X_sample = X_sample.astype(float)

        salary = model_loaded.predict(X_sample)
        st.subheader(f"The estimated salary: USD {salary[0]:.2f} per year.")
        st.subheader(f"The estimated salary: EUR {salary[0]*0.85:.2f} per year.")
        st.subheader(f"The estimated salary: CZK {salary[0]*21.5/12:.2f} per month.")

    st.subheader("Sources:")
    st.write("Source of the data for the model: https://insights.stackoverflow.com/survey")
    st.write("To see other author’s projects: https://jaroslavkotrba.com")

if __name__ == '__main__':
    main()
