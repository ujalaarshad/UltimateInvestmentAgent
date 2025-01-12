Ultimate Investment Agent

Overview

The Ultimate Investment Agent is a cutting-edge RAG (Retrieval-Augmented Generation) application designed to streamline investment-related queries. It integrates web scraping, document uploads, and a powerful knowledge base to provide an efficient and user-friendly experience. Built with Streamlit, the application features distinct functionalities for administrators and regular users, ensuring secure and effective knowledge management.

Key Features:

Web Scraping: Extract data from websites by providing a URL.

Document Uploads: Upload documents to enhance the knowledge base.

Knowledge Base Management (Admin):

Add, delete, and update documents.

Attach metadata to documents.

RAG Conversation Chain:

Retrieve and generate responses using ChromaDB.

Intuitive chat interface for regular users.

Authentication: Admin access is restricted and secured.

Streamlit Integration: Hosted on Streamlit Cloud, providing a seamless and interactive user experience.

Project Structure:

main.py: Entry point of the application.

chain.py: Contains the logic for:

Storing and fetching data from the vector database (ChromaDB).

Implementing the RAG conversation chain.

app.py: Manages the frontend interface and integrates the chain functionality.

requirements.txt: Lists dependencies required for the application.

How It Works

Normal User Access:

Users can chat with the agent to retrieve investment-related insights.

No direct access to manage the knowledge base.

Admin Access:

Authenticate to access admin features.

Manage the knowledge base by adding/deleting documents and metadata.

Data Handling:

Scraped or uploaded data is stored in ChromaDB.

Metadata enhances document indexing and retrieval accuracy.

Installation and Setup

Clone the Repository:

git clone <repository_url>
cd UltimateInvestmentAgent

Install Dependencies:

pip install -r requirements.txt

Run the Application:

streamlit run main.py

Deployment

The application is hosted on Streamlit Cloud. Access the live version via the hosted URL.

Security

Authentication ensures that only authorized administrators can manage the knowledge base.

Regular users have restricted access to maintain the integrity of the knowledge base.

Enhance your investment decisions with the Ultimate Investment Agent. Whether you're an administrator managing a knowledge base or a user seeking investment insights, our application delivers accuracy and efficiency.
