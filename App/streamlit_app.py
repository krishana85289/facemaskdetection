import streamlit as st
import cv2
import numpy as np

st.title("OpenCV Camera Stream in Streamlit")

# Create a VideoCapture object to access the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Cannot access the camera. Please check your webcam.")

# Function to read frames from the webcam and display them
def main():
    frame_placeholder = st.empty()  # Placeholder for displaying frames
    stop_button_press = st.button("Stop")  # Button to stop the stream

    while cap.isOpened() and (stop_button_press is False):
        ret, frame = cap.read()

        if not ret:
            st.write("Error reading frame from the camera.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB")

if __name__ == "__main__":
    main()

# Release the webcam when done
cap.release()
