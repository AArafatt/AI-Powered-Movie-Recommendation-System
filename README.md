# AI-Powered-Movie-Recommendation-System

A movie recommendation system built with FastAPI, OpenAI GPT-3.5 Turbo, OpenAI embeddings, and Streamlit to deliver personalized movie suggestions based on natural language queries.

Overview
This project leverages state-of-the-art AI models to provide tailored movie recommendations. Users can input their movie preferences or describe the type of movies they like, and the system uses GPT-3.5 Turbo along with embedding techniques to retrieve and rank movies that best match their interests. Additionally, a user-friendly Streamlit interface is provided for interactive exploration and visualization of recommendations.

Features
Personalized Recommendations: Uses GPT-3.5 Turbo to interpret user queries and generate customized recommendations.
Semantic Search: Utilizes OpenAI embeddings for effective similarity matching between user preferences and movie data.
RESTful API: Built with FastAPI for a scalable and efficient API service.
Interactive UI: Streamlit integration for a quick and easy-to-use graphical interface.
Easy Integration: Simple setup and deployment, making it easy to extend or integrate with other services.
Architecture
FastAPI: Serves as the backend framework that handles API requests and responses.
GPT-3.5 Turbo: Processes natural language queries to understand user intent.
OpenAI Embeddings: Converts movie descriptions and user queries into vector representations for similarity searches.
Streamlit: Provides an interactive user interface for a seamless user experience.
Getting Started
Prerequisites
Python 3.8+
pip

Installation
1.Clone the Repository:
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name

2.Create a Virtual Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3.Install Dependencies:
pip install -r requirements.txt

4.Configure Environment Variables:
OPENAI_API_KEY=your_openai_api_key_here

Running the Application

FastAPI Server
Start the FastAPI server using Uvicorn: uvicorn main:app --reload

Streamlit Interface
To launch the Streamlit interface for a graphical view of the recommendations:
streamlit run app.py


Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch with your feature or bug fix.
Commit your changes and push the branch.
Open a pull request detailing your changes.
For major changes, please open an issue first to discuss what you would like to change.
