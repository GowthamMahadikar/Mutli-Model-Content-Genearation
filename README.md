 Multimodal Content Generation have following capabilities:

## 1. A `Conversational chatbot` as same as `ChatGPT + Image Summarization` Capabilities through `GOOGLE GEMINI VISION PRO API`.
## 2. `Text to Image` (using Stability Ai (Stable Diffusion)) through `HUGGINGFACE API`.

## Setup steps:
1. Create virtual environment
    ```
    python -m venv <name of virtual environment>
    ```

2. Activate it
    ```
    source <name of virtual environment>/bin/activate
    ```

3. Now install required libraries from requirements.txt file using...
    ```
    pip install -r requirements.txt
    ```
4. Create .env file and add your API TOKEN
   ```
   GOOGLE_API_KEY="Enter Your GOOGLE API TOKEN"
   HUGGINGFACE_API_KEY=""
   ```
5. To run app
    ```
    streamlit run <name-of-app>.py
    ```

