from dotenv import load_dotenv
from huggingface_hub import InferenceApi
load_dotenv()   # load all env. variables

import streamlit as st
import os
import base64
import google.generativeai as genai
from PIL import Image
import time

img=Image.open('e:\mutli model summarizer\Multimodal-Content-Generation-using-LLMs\Colorful Modern Infinity Technology Free Logo.png')

# set api key
genai.configure(api_key=os.getenv("AIzaSyAeTxSxkzOSBMM2CNALAan498Qothp8d7Q"))
# HUGGING_FACE_API_TOKEN=os.getenv("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")

# initialize our models
vis_model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
langugae_model = genai.GenerativeModel("gemini-pro")

# function to load gemini pro vision model and get responses
def get_gemini_response(input, image):
    # if there present any text input besides image, then generate both
    if (input != "") and (image != ""):
        response = vis_model.generate_content([input, image])
    elif (input != "") and (image == ""):
        response = langugae_model.generate_content(input)
    else:
        response = vis_model.generate_content(image)
    
    return response.text

def stream_data(prompt, image):

    sentences = get_gemini_response(prompt, image).split(". ")

    for sentence in sentences:
        for word in sentence.split():
            yield word + " "
            time.sleep(0.02)

# initialize our streamlit app
st.set_page_config(
    page_title="Multimodel Content Generation",
    page_icon=img,
    layout="wide"
)
# Adding logo for the App
file_ = open("Colorful Modern Infinity Technology Free Logo.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.sidebar.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="" style="height:300px; width:400px;">',
    unsafe_allow_html=True,
)
# give title
st.sidebar.title(":rainbow[MULTIMODEL CONTENT GENERATION]")
st.sidebar.write("Built by [Gowtham Mahadikar] 😀")
st.sidebar.divider()

# Multimodals Options
multimodal_options = st.sidebar.radio(
    "**Select What To Do...**",
    options=["Chat and Image Summarization", "Text 2 Image",],
    index=0,
    horizontal=False,
)
st.sidebar.divider()

# =======================================================================================

if multimodal_options == "Chat and Image Summarization":

    # New chat button, to get the fresh chat page
    if st.sidebar.button("Get **New Chat** Fresh Page"):
        st.session_state["messages"] = []  # Clear chat history
        st.experimental_rerun()  # Trigger page reload

    # create image upload option in sidebar
    with st.expander("**Wanna Upload an Image?**"):
        uploaded_file = st.file_uploader("Choose an image for **Image Summarizer** task...",
                                        type=["jpg", "jpeg", "png"])
        image=""
        if uploaded_file is not None:
            image=Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image.", use_column_width=True)


    # Create a container to hold the entire chat history
    chat_container = st.container()

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with chat_container:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # create input prompt (textbox)
    if prompt := st.chat_input("Type here..."):

        # display user message in chat message container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        # add user message to chat history
        st.session_state.messages.append({"role" : "user",
                                        "content" : prompt})
        

        # display assistant message in chat message container
        with chat_container:
            with st.chat_message("assistant"):

                should_format_as_code = any(keyword in prompt.lower() for keyword in ["code", "python", "java", "javascript", "c++", "c", "program", "react", "reactjs", "node", "nodejs", "html", "css", "javascript", "js"])

                if should_format_as_code:
                    st.code(get_gemini_response(prompt, image))
                else:
                    st.write_stream(stream_data(prompt, image))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": get_gemini_response(prompt, image)})

# =============================================================================
import requests
def generate_and_display_image(submitted: bool, prompt: str, width: int, height: int, num_images: int):
    """
    Generates an image using the specified prompt.
    """
    if HUGGINGFACE_API_TOKEN.startswith('hf_') and submitted and prompt:
        with st.status('Generating your image...', expanded=True) as status:
            try:
                # Create an instance of the InferenceApi with your Hugging Face model
                api = InferenceApi(repo_id="black-forest-labs/FLUX.1-dev", token=HUGGINGFACE_API_TOKEN)
                
                # Create a list to hold the generated images
                generated_images = []
                
                # Loop to generate multiple images if required
                for _ in range(num_images):
                    # Call the Hugging Face API with the prompt
                    output = api(inputs=prompt, params={"width": width, "height": height})
                    
                    # Check if the output is a PIL Image
                    if isinstance(output, Image.Image):
                        generated_images.append(output)
                    else:
                        st.error(f"Unexpected response format: {output}")
                        return

                # Display all generated images
                if generated_images:
                    st.toast('Your images have been generated!', icon='😍')
                    for img in generated_images:
                        st.image(img, caption="Generated Image ❄️", use_column_width=True)

            except Exception as e:
                st.error(f"Error generating image: {e}")
    
    elif not prompt:
        st.toast("Please input some prompt!", icon="⚠️")


# Function to refine output parameters
def refine_output():
    """
    Provides options for users to refine output parameters and returns them.
    """
    with st.expander("**Refine your output if you want...**"):
         width = st.number_input("Width of output image", value=1024)
         height = st.number_input("Height of output image", value=1024)
         num_images = st.slider("Number of images to output", value=1, min_value=1, max_value=4)
         negative_prompt = st.text_input("Enter your negative prompt (what to avoid in the image):", value="(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck")
    
    # Prompt input
    prompt = st.text_input("Enter your prompt for the image:", value="a tiny astronaut hatching from an egg on the moon")

    # Submit button to trigger image generation
    submitted = st.button("Generate")

    return submitted, prompt, width, height, num_images


# Prompt input and image generation logic
if multimodal_options == "Text 2 Image":

    HUGGINGFACE_API_TOKEN = st.sidebar.text_input(
        "Enter your HUGGING FACE API TOKEN",
        placeholder="Paste token here...",
        type="password"
    )
    os.environ["HUGGINGFACE_API_TOKEN"] = HUGGINGFACE_API_TOKEN

    if not HUGGINGFACE_API_TOKEN.startswith('hf_'):
        st.warning('Please enter your HUGGING FACE API KEY in **Sidebar**!', icon='⚠')

    submitted, prompt, width, height, num_images = refine_output()
    generate_and_display_image(submitted, prompt, width, height, num_images)
