import os
import numpy as np
import yaml
from docx import Document


from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from ragatouille import RAGPretrainedModel

from data_ingester import ChatbotDataIngester
from data_query import ChatbotDataQuery
from getpass import getpass
from pinecone import Pinecone, ServerlessSpec

import torch
import torch.nn.functional as F
from transformers import AutoModel

from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from PRESET_QUERIES import Queries, Query_Doc_Map
from data_query import generate_openai_response

from dotenv import load_dotenv
load_dotenv()

class RAGChatbot:
    def __init__(self, pinecone_api_key=None, index_name="test-index", config_path="../config.yml"):
        """
        Initialize the RAGChatbot. Handles embeddings, vector store, data ingestion, and query.
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")# or getpass("Enter your Pinecone API key: ")
        self.index_name = index_name
        self.embeddings = self.initialize_embeddings()
        self.dimensions = len(self.embeddings.embed_query("Hello World!"))
        self.vector_store = self.initialize_vector_store()
        self.data_ingester = ChatbotDataIngester(vector_store=self.vector_store, embeddings=self.embeddings)
        self.data_query = ChatbotDataQuery(vector_store=self.vector_store)
        self.reranker = self.initialize_reranker()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        
    def load_config(self, config_path):
        """
        Load the configuration file (config.yml).
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def initialize_embeddings(self):
        """
        Initialize the embedding model based on the config file.
        """
        model_name = "BAAI/bge-large-en-v1.5"
        model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        hf = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
        return hf

    def initialize_reranker(self):
        """
        Initialize the reranker
        """
        return RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

    def initialize_vector_store(self):
        """
        Initialize Pinecone vector store.
        """
        pc = Pinecone(api_key=self.pinecone_api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

        if self.index_name not in existing_indexes:
            pc.create_index(
                name=self.index_name,
                dimension=self.dimensions,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            while not pc.describe_index(self.index_name).status["ready"]:
                import time
                time.sleep(1)

        return PineconeVectorStore(index=pc.Index(self.index_name), embedding=self.embeddings)

    def ingest_data(self, dir_path, empty=False):
        """
        Ingest data from a directory using the ChatbotDataIngester.
        """
        self.data_ingester.load_and_ingest(dir_path, empty_db=empty)

    def __route(self, query_text):
        query_text = query_text.lower()
        def cosine_similarity_calc(vec1, vec2):
            vec1 = np.array(vec1).reshape(1, -1)
            vec2 = np.array(vec2).reshape(1, -1)
            return cosine_similarity(vec1, vec2)[0][0]

        def get_embeddings(client, text):
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-large"
            )
            return response.data[0].embedding

        # Generate embeddings for the incoming query
        query_embedding = get_embeddings(self.client, query_text)
        
        best_match = None
        highest_similarity = 0

        for main_query, similar_queries in Queries.items():
            for query in similar_queries:
                query = query.lower()
                preset_embedding = get_embeddings(self.client, query)
                similarity_score = cosine_similarity_calc(query_embedding, preset_embedding)
                if similarity_score > highest_similarity:
                    highest_similarity = similarity_score
                    best_match = main_query

        if highest_similarity >= 0.5100:
            # print(f'Response from routing:query_text: {query_text} - best_match query: {best_match} - Doc: {Query_Doc_Map[best_match][0]}')
            response, file_path = self.__generate_response_from_file(query_text, Query_Doc_Map[best_match][0])
            return response, file_path
        else:
            return None, None

    def __generate_response_from_file(self, query_text, file_path):
        """
        Generate response from a file.
        """
        def read_docx(file_path):
            doc = Document(file_path)
            full_text = []
            for paragraph in doc.paragraphs:
                full_text.append(paragraph.text)
            return '\n'.join(full_text)

        file_content = read_docx(os.path.join('../../Data', file_path))

        system_prompt = '''
        You are an intelligent assistant designed to provide clear, accurate, and helpful responses. 
        Focus on understanding user intent, give concise answers, and offer step-by-step solutions when necessary.
        Be friendly, professional, and avoid unnecessary information.\n'''

        input_prompt = f'Query: {query_text}\nContext: {file_content}'

        response = generate_openai_response(input_prompt, system_prompt)
        return response.split('\n')[1], os.path.join('../../Data', file_path)

    def query_chatbot(self, query_text, k=1, rerank=False): #, fetch_k=2, lambda_mult=0.5
        """
        Query the chatbot using the provided query text and optional search parameters.
        """

        route_response, file_path = self.__route(query_text)
        if route_response == None:
            if rerank:
                response, context_docs = self.data_query.query(
                    query_text=query_text,
                    k=k,
                    reranker=self.reranker
                )
            else:
                response = self.data_query.query(
                    query_text=query_text,
                    k=k,
                )
            return response, context_docs
        else:
            return route_response, file_path