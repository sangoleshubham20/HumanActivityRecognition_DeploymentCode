import streamlit as st
from PIL import Image


def show_home_page():
    """
    This function displays the Home Page.
    """
    st.title("Human Activity Recognition using Machine Learning")
    st.text("")
    # col1, col2, col3 = st.columns([1, 3, 1])
    col1, col2, col3, col4, col5 = st.columns([1, 1, 4, 1, 1])
    with col3:
        # st.markdown("### HUMAN ACTIVITY RECOGNITION USING MACHINE LEARNING")
        st.text("")
        st.text("")
        st.markdown("![Alt Text](https://media.giphy.com/media/OM87ilEkmKz0zMMKP5/giphy.gif)")

    st.text("")
    st.text("")
    st.caption("### **OVERVIEW**")
    st.markdown("""**Human activity recognition, or HAR for short, is a broad field of study concerned with
                    identifying the specific movement or action of a person based on sensor data**. Movements are
                    often typical activities performed indoors such as walking, talking, standing and sitting.
                    They may also be more focused activities such as those types of activities performed in a
                    kitchen or on a factory floor. HAR has multifaceted applications due to its worldly usage of
                    acquisition devices such as smartphones, video cameras, and its ability to capture human
                    activity data. While electronic devices and their applications are steadily growing, the
                    advances in Machine Learning & Artificial intelligence (AI) have revolutionized the ability to
                    extract deep hidden information for accurate detection and its interpretation. This yields a
                    better understanding of rapidly growing acquisition devices, AI, and applications, the three
                    pillars of HAR under one roof.
                """)

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img = Image.open("HAR_2.png")
        st.image(img)

    st.write("""The existence of devices like sensors, video cameras, radio frequency identification (RFID), and
                Wi-Fi are not new, but the usage of these devices in HAR is in its infancy. The reason for HAR’s
                evolution is the fast growth of techniques such as Machine Learning & AI, which enables the use
                of these devices in various application domains. Therefore, we can say that there is a mutual
                relationship between the AI/ML models and HAR devices. Earlier these models were based on a
                single image or a small sequence of images, but the advancements in AI have provided more
                opportunities. In the last decade, researchers have developed various HAR models for different
                domains.
             """)
    st.markdown("Here are some of the applications of HAR :")
    st.markdown("""* **Crowd surveillance (cSurv)**: Crowd pattern monitoring and detecting panic situations in the
                   crowd.
                """)
    st.markdown("* **Health care monitoring (mHealthcare)**: Assistive care to ICU patients, Trauma resuscitation.")
    st.markdown("* **Smart home (sHome)**: Care to elderly or dementia patients and child activity monitoring.")
    st.markdown("* **Fall detection (fDetect)**: Detection of abnormality in action which results in a person's fall.")
    st.markdown("* **Exercise monitoring (eMonitor)**: Pose estimation while doing exercise.")
    st.markdown("* **Gait analysis (gAnalysis)**: Analyze gait patterns to monitor health problems.")
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **PURPOSE OF THIS PROJECT & IT'S RELEVANCE**")
    st.markdown("""“What type of HAR device is suitable for which application domain and what is the suitable AI
                    methodology” is the biggest question that pops into the mind when developing the HAR
                    framework. In the last 10 years, various HAR models with high performance came into the
                    picture. While working on this project, I was constantly worried about a thought that "how
                    well these models can be used in real-environment without integrating specialized hardware?"
                    For applications with huge data such as videos, we need GPUs for training the model. Python
                    offers libraries (such as Keras, TensorFlow) for implementing AI framework on a general-
                    purpose CPU processor. For working on GPUs, one needs to explore special libraries for
                    implementing AI models. Sometimes, it may result in specialized hardware integration need in
                    the target application which makes it expensive. Processing power and costs are interrelated
                    i.e., one needs to pay more for extra power. Therefore, I had primarily two objectives in my
                    mind. **Firstly, I wanted to design a robust and lightweight system which can run in real-
                    environment without the need for specialized hardware like graphics processing units (GPUs)
                    and extra memory.**
                """)
    st.markdown("""**Secondly, I wanted my system to be helpful in tasks such as forecasting future actions**.
                   For example, in a smart home environment if an elderly person is doing exercise and there are
                   chances of fall then it will be very helpful if there is a smart system that can identify
                   fall in advance and inform the person timely for necessary precaution(s).
                """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **BUSINESS PROBLEM**")
    st.markdown("""There are various HAR devices for capturing human activity signals.
                   Miniaturized mobile devices are handy to use and offer a set of physiological sensors that
                   can be used for capturing activity signals. The foremost goal of HAR is to predict the
                   movement or action of a person based on the action data collected from a data acquisition
                   device. These movements include activities like walking, exercising, jogging, etc. It is
                   challenging to predict movements, as it involves huge amounts of unlabelled sensor data,
                   and video data which suffer from conditions like lights, background noise, and scale
                   variation. **We need to build a machine learning system for the end users that is not
                   intimidating & is easy to use**. It will help the users in performing tasks and to improve
                   their lifestyle such as remote care to the elderly living alone and posture monitoring during
                   exercise. A user can easily install this application on his/her smartphone and run it in a
                   couple of seconds to look at the activities he/she has performed throughout the day.
                """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **BUSINESS CONSTRAINTS**")
    st.markdown("""* **Interpretability of the model** is not that consequential because a user doesn't need to know why
                   the model has given a particular class to a specific sensor data.
                """)
    st.markdown("""* There are **strict latency requirements**. We don't want our latency to be in several minutes. We
                   want our model to predict the classes in real time.
                """)
    st.markdown("* **Errors won't cause extreme fatalities** but they must be minimised.")
    st.markdown("""* The model should return a **probability of the data point belonging to a class** rather than
                   simply returning a class.
                """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **MAPPING THE PROBLEM TO A MACHINE LEARNING PROBLEM**")
    st.markdown("""* Given an accelerometer & gyroscope reading, **we need to classify it into any 1 of the 6 classes**.
                """)
    st.markdown("""* Since there are 6 classes[Walking, Walking_upstairs, Walking_downstairs, Sitting, Standing,
                   Laying], it is a **Multi-class classification problem**.
                """)
    st.markdown("""* **Performance metric(s) :** Accuracy, Confusion Matrix, Precision, Recall, F1-Score""")
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **INTERNAL STRUCTURE**")
    st.text("")
    st.text("")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img_internal_struc = Image.open("internal_structure.png")
        st.image(img_internal_struc)
    st.text("")
    st.text("")
    st.markdown("A typical Human Activity Recognition system consists of **4 stages** :")
    st.markdown("* **Stage 1** : Capturing the signal activity(Data acquisition)")
    st.markdown("* **Stage 2** : Data pre-processing")
    st.markdown("* **Stage 3** : Model training & Performance evaluation")
    st.markdown("* **Stage 4** : Building a user interface for the management of HAR(Front-end)")
    st.text("")
    st.markdown("""In **stage 1**, depending on the target application, a HAR device is selected. For example, in
                   surveillance application involving multiple people, the HAR device for data collection is
                   the camera. Similarly, for applications where a person's daily activity monitoring is
                   involved, the data acquisition source is preferably some sort of a sensor. In **stage 2**,
                   data pre-processing is performed, which involves tasks like removal of noise or other unwanted
                   signals from the data, regional and boundary segmentation of the data, etc. The segmented
                   data can now be used for model training. **Stage 3** involves the training of HAR model using
                   various Machine Learning techniques. After training an HAR model it is ready to be used for
                   an application. In **stage 4**, the model is applied to the real data for predictions using a
                   front-end application.
                """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **SENSORS**")
    st.write("""The sensors-based approaches can be broadly categorized into **Object sensors** and **Wearable
                sensors**.
             """)
    st.text("")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img_sensors = Image.open("sensors.png")
        st.image(img_sensors)
    st.markdown("""In wearable sensor-based approach, a body-worn sensor module is designed which includes
                   inertial sensors & environmental sensors units. Sometimes the wearable sensor devices can be
                   stressful for the user, therefore the solution is the use of smart-device sensors. In device
                   sensor approach data is captured using smartphone inertial sensors. The most commonly used
                   sensor for HAR is accelerometer and gyroscope. The sensor data may be remotely recorded, such
                   as video, radar, or other wireless methods. Alternately, data may be recorded directly on the
                   subject such as by carrying custom hardware or smart phones that have accelerometers and
                   gyroscopes. 
                """)
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        img_gyro_acc = Image.open("gyro_acc.jpg")
        st.image(img_gyro_acc)

    st.markdown("""**Accelerometer** : It is an electronic sensor that measures the acceleration forces acting
                   on an object, in order to determine the object’s position in space and monitor the object’s
                   movement.
                """)
    st.markdown("""**Gyroscope** : It is a device that can measure and maintain the orientation and angular
                   velocity of an object. These are more advanced than accelerometers. These can measure the
                   tilt and lateral orientation of the object whereas accelerometer can only measure the linear
                   motion.
                """)

    col4, col5, col6 = st.columns([1, 3, 1])
    with col5:
        img = Image.open(
            "Orientation-of-the-axes-relative-to-a-typical-smartphone-device-using-a-gyroscope-sensor.png")
        st.image(img)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption('### **CLASSIFYING ACTIVITIES OF DAILY LIVING WITH SMARTPHONE SENSORS**')
    st.text("")
    col7, col8, col9 = st.columns([1, 3, 1])
    with col8:
        img = Image.open("HAR 3.png")
        st.image(img)
    st.write("""The project uses smartphone sensors to accurately classify six different activities of daily
                living: walking, walking upstairs, walking downstairs, sitting, standing, and lying down. By
                utilizing machine learning algorithms, we can accurately predict which activity a person is
                performing based on sensor data captured by their smartphone.
             """)
    st.write("""The dataset was collected from 30 study participants within an age range of 19-48 years.
                Participants wore a smartphone on their waist which captured 3-axial linear acceleration and
                3-axial angular velocity at a constant rate of 50Hz. Each participant performed the six
                activities of daily living while being video-recorded to manually label the data.
             """)
    st.write("""Further, I've pre-processed the sensor signals by applying noise filters and sampled them in
                fixed-width sliding windows of 2.56 seconds with 50% overlap. From each window, I've extracted a
                vector of features by calculating variables from the time and frequency domain. In our dataset,
                each datapoint represents a window with different readings and each window has 128 readings.
             """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **MACHINE LEARNING TECHNIQUES**")
    st.markdown("**Linear Models:**")
    st.markdown("* Logistic Regression with GridSearch")
    st.markdown("* Linear SVC with GridSearch")
    st.markdown("* Kernel SVM with GridSearch")
    st.markdown("**Tree based model:**")
    st.markdown("* Decision Trees with GridSearch")
    st.markdown("**Ensemble learning:**")
    st.markdown("* Random Forest Classifier with GridSearch")
    st.markdown("* Gradient Boosted Decision Trees With GridSearch")
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **FRONT-END**")
    st.write("""The front-end is a web application that a user can run on his/her device to track his/her
                activities throughout the day. It performs the following three actions:
             """)
    st.markdown("""* **Generate Model Inputs:** It captures the user's activity data through accelerometer and
                   gyroscope sensors from his/her device.
                """)
    st.markdown("""* **Generate Predictions:** It unpacks the pickle model and makes predictions using the
                   features generated above.
                """)
    st.markdown("""* **Visualize Results:** It takes the predictions and shapes the data so that it can be shown
                   interactively in a web application. The web application is made with Streamlit.
                """)
    st.markdown("""<hr style="height:0.5px;border:none;color:#333;background-color:#333;" /> """,
                unsafe_allow_html=True)

    st.caption("### **DEPENDENCIES & MODULES**")
    st.write("You need the following dependencies and modules installed in your environment:")
    st.markdown("""**`altair`, `datetime`, `matplotlib`, `numpy`, `pandas`, `pickle`, `PIL`, `scikit-learn`, `scipy`,
                   `seaborn`, `streamlit`**""")
