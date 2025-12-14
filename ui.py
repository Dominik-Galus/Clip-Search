import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/search"

st.set_page_config(page_title="Clip Video Search", layout="wide")

st.title("Clip Search Engine")
st.markdown("Video Search Engine using CLIP + FAISS")

with st.sidebar:
    st.header("Settings")
    k_search = st.slider("Amount of results", min_value=1, max_value=10, value=4)

query = st.text_input("Give description of the video you are searching for:", placeholder="for example: playing tennis, cutting vegetables")

if st.button("Search") or query:
    if not query:
        st.warning("Write Query")
    else:
        with st.spinner(f"Searching video for: '{query}'"):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query, "k_search": k_search},
                    timeout=10
                )

                if response.status_code == 200:
                    data = response.json()
                    videos = data.get("videos", [])
                    distances = data.get("distances", [])

                    st.success(f"Found {len(videos)} results.")

                    cols = st.columns(2)

                    for idx, (video_path, score) in enumerate(zip(videos, distances, strict=False)):
                        col = cols[idx % 2]
                        video_path = video_path.replace(".avi", ".mp4")
                        with col:
                            st.subheader(f"Result #{idx + 1} (Score: {score:.3f})")
                            st.text(video_path)

                            st.video(video_path)

                else:
                    st.error(f"API Error: {response.status_code}")
                    st.json(response.json())

            except Exception as e:
                st.error(f"Couldn't connect with API \nError: {e}")
