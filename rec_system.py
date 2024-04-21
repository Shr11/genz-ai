
import pickle
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st






# Recommendation

# Load the scaler, label encoder, model, and class names
scaler = pickle.load(open("C:/Users/asus/Desktop/counsello/scaler.pkl", 'rb'))
model = pickle.load(open("C:/Users/asus/Desktop/counsello/model.pkl", 'rb'))


class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']

def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score,average_score):

    
    
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,total_score,average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs

    
    
def main():
    
    st.set_page_config(
    page_title="Multipage App",
    page_icon="👋",
)
     # giving a title
    st.title('Counsello')
    st.write("An AI career counselling assistant for students")
    
    
    # getting the input data from the user
    gender = st.selectbox('Gender', ['Male', 'Female'])
    part_time_job = st.selectbox('Do you have a part-time job?', ['Yes', 'No'])
    absence_days = st.number_input('Number of absence days', min_value=0, max_value=70, value=0)
    extracurricular_activities = st.selectbox('Do you participate in extracurricular activities?', ['Yes', 'No'])
    weekly_self_study_hours = st.selectbox('Weekly self-study hours', list(range(0, 43)))
    math_score = st.text_input('Math score')
    history_score = st.text_input('History score')
    physics_score = st.text_input('Physics score')
    chemistry_score = st.text_input('Chemistry score')
    biology_score = st.text_input('Biology score')
    english_score = st.text_input('English score')
    geography_score = st.text_input('Geography score')
    total_score = st.text_input('Total score ')
    average_score = st.text_input('Average score')
    
    

    submit = st.button("Generate")
    if submit:
        new_rec = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score,total_score,average_score)
        print(new_rec)
        for class_name, probability in new_rec:
            st.write('Your can be a ', class_name,'with a probability of ', probability * 100 , '%')
    
    
if __name__ == "__main__":
    main()