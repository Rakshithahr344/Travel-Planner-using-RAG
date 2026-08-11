%%writefile app.py

import os
import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ AI Prompt-Based Travel Planner")
st.write(
    "Create a personalized travel plan using "
    "RAG, Gemini and LangChain."
)


# ==============================
# API KEY
# ==============================

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    type="password"
)

if not api_key:
    st.info("Enter your Gemini API key in the sidebar.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key


# ==============================
# TRAVEL KNOWLEDGE BASE
# ==============================

travel_data = [
    {
        "destination": "Goa",
        "information": """
        Goa is popular for beaches, nightlife, water sports,
        Portuguese architecture, seafood and relaxing vacations.
        Popular places include Baga Beach, Calangute, Anjuna,
        Panjim, Fort Aguada and Palolem.
        Budget travelers can use hostels and budget hotels.
        Local transport includes buses, taxis and rental scooters.
        """
    },

    {
        "destination": "Bangalore",
        "information": """
        Bangalore is known for technology, gardens, cafes and
        pleasant weather. Popular attractions include Cubbon Park,
        Lalbagh, Bangalore Palace, Vidhana Soudha and ISKCON Temple.
        Budget travelers can use metro, buses and affordable hotels.
        """
    },

    {
        "destination": "Mysore",
        "information": """
        Mysore is famous for Mysore Palace, Chamundi Hill,
        Brindavan Gardens and St Philomena's Church.
        It is suitable for cultural and historical trips.
        Budget hotels and local buses are available.
        """
    },

    {
        "destination": "Manali",
        "information": """
        Manali is a mountain destination known for snow,
        valleys and adventure activities.
        Popular places include Solang Valley, Rohtang Pass,
        Hidimba Temple and Mall Road.
        """
    },

    {
        "destination": "Kerala",
        "information": """
        Kerala is famous for backwaters, beaches, hill stations
        and cultural experiences.
        Popular places include Munnar, Alleppey, Kochi,
        Varkala and Thekkady.
        """
    }
]


documents = [
    Document(
        page_content=item["information"],
        metadata={"destination": item["destination"]}
    )
    for item in travel_data
]


# ==============================
# RAG
# ==============================

@st.cache_resource
def create_retriever():

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


retriever = create_retriever()


# ==============================
# GEMINI
# ==============================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0.4
)


# ==============================
# PROMPT TEMPLATE
# ==============================

prompt = ChatPromptTemplate.from_template("""
You are an expert travel planner.

Create a practical travel plan based on the user's requirements
and the retrieved travel information.

Destination:
{destination}

Budget:
₹{budget}

Number of Days:
{days}

Interests:
{interests}

Retrieved Travel Information:
{context}

Generate the answer using:

## ✈️ Trip Overview

## 📅 Day-by-Day Itinerary

## 🏨 Hotel Suggestions

## 🎒 Packing List

## 💰 Estimated Cost

## 💡 Travel Tips

Keep the plan within the user's budget as much as possible.
Mention that prices are approximate.
Do not claim exact hotel availability or prices.
""")


# ==============================
# USER INPUT
# ==============================

st.sidebar.header("Trip Details")

destination = st.sidebar.selectbox(
    "Destination",
    ["Goa", "Bangalore", "Mysore", "Manali", "Kerala"]
)

budget = st.sidebar.number_input(
    "Budget (₹)",
    min_value=1000,
    max_value=500000,
    value=20000,
    step=1000
)

days = st.sidebar.number_input(
    "Number of Days",
    min_value=1,
    max_value=30,
    value=3
)

interests = st.sidebar.multiselect(
    "Interests",
    [
        "Beaches",
        "Nature",
        "Food",
        "Adventure",
        "History",
        "Shopping",
        "Culture",
        "Sightseeing"
    ]
)


# ==============================
# GENERATE PLAN
# ==============================

if st.button("🚀 Generate Travel Plan"):

    if not interests:
        st.warning("Please select at least one interest.")

    else:

        with st.spinner("Creating your travel plan..."):

            query = f"""
            Destination: {destination}
            Budget: ₹{budget}
            Days: {days}
            Interests: {", ".join(interests)}
            """

            retrieved_docs = retriever.invoke(query)

            context = "\n\n".join(
                doc.page_content
                for doc in retrieved_docs
            )

            formatted_prompt = prompt.format(
                destination=destination,
                budget=budget,
                days=days,
                interests=", ".join(interests),
                context=context
            )

            response = llm.invoke(formatted_prompt)

            st.markdown(response.content)


st.divider()

st.caption(
    "AI Prompt-Based Travel Planner | RAG + Gemini + LangChain + Streamlit"
)S