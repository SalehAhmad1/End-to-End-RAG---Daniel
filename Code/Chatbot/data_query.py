import getpass
import os
import numpy as np
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def generate_openai_response(input_prompt, system_prompt=None, temperature=0):
    if system_prompt is None:
        system_prompt = '''You are an assistant designed to provide answers when no (0) relevant documents are retrieved from the vector database. When this happens, you should follow these steps:
                        1) First, determine if you can answer the user's query using general knowledge or internal information. If so, generate a confident, helpful response in a straightforward narrative style. Do not use phrases such as 'According to me,' 'As of my knowledge,' 'I don’t know but,' or mention knowledge cutoffs or lack of information. Simply provide the answer as if you are certain of the facts.
                        2) If the question is domain-specific, too specific (e.g., about a particular person or object that could mislead), or outside your knowledge, do not attempt to answer. Politely respond with: 'I'm sorry, I currently do not have enough information to answer your question.
                        
                        When asked about you, answer from the file(s):
                        - Who-is-Wagner-Chatbot-Response.docx

                        When asked about Daniel, answer from the file(s):
                        - Ringel_Daniel_CV_V1.docx
                        '''
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"), temperature=temperature)
    return llm(system_prompt+input_prompt).content
    
class ChatbotDataQuery:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = '''You are Wagner, a highly intelligent and friendly AI assistant.
        You are developed to provide answers to given prompts about you, Daniel and Daniel's work. You will be given some context to generate the answer from to the user.
        Answer in a very affirmative, concise, and clear manner. Never refuse to answer, always provide relevant information.

        When asked about you, answer from the file(s):
        - Who-is-Wagner-Chatbot-Response.docx

        When asked about Daniel, answer from the file(s):
        - Ringel_Daniel_CV_V1.docx
        '''

        if vector_store is None:
            raise ValueError("Vector store cannot be None")
        else:
            self.vector_store = vector_store

    def initialize_reranker(self):
        """
        Initialize the custom reranker.
        """
        return CustomReranker()

    def __generate_response(self, query_text, k, reranker=None, reranker_docs=0):

        retrieved_results = self.vector_store.similarity_search_with_score(
            query_text, k=k
        )

        Highest_Similarity_Score = retrieved_results[0][1]
        print(f'The highest similarity score is {Highest_Similarity_Score}')

        if Highest_Similarity_Score > 0.51:
            context_docs = []
            for res, score in retrieved_results:
                context_docs.append(res)

            context_docs_texts = []
            for res, score in retrieved_results:
                context_docs_texts.append(res.page_content)

            if (reranker is not None) and (reranker_docs > 0):
                relevant_docs = reranker.rerank(query_text, context_docs_texts, k=reranker_docs)

                All_Scores = [doc['score'] for doc in relevant_docs]
                Min = min(All_Scores)
                Max = max(All_Scores)
                Normalized_Scores = [(doc['score'] - Min) / (Max - Min) for doc in relevant_docs]
                
                for idx,doc in enumerate(relevant_docs):
                    doc['score'] = Normalized_Scores[idx]

                final_reranked_docs = []
                for reranked_doc in relevant_docs:
                    if reranked_doc['score'] < 0.35:
                        continue
                    else:
                        idx_of_content_in_context_doc = reranked_doc['result_index']
                        meta_data = context_docs[idx_of_content_in_context_doc].metadata
                        final_reranked_docs.append(Document(page_content=reranked_doc['content'], metadata=meta_data))
                
                context_docs = final_reranked_docs

            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant that only answers questions about the context. "
                "You try your best to extract the relevant answers from the context. "
                "The context is:\n\n{context}\n\n"
                "Question: {question}\n"
                "Helpful Answer:"
            )

            chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt,
                document_variable_name="context",
            )

            context = '\n\n'.join([doc.page_content for doc in context_docs])
            query = [
                ("system", f"{self.system_prompt}"),
                ("human", f"context: {context}\nInput: {query_text}"),
            ]

            response = ''
            for chunk in self.llm.stream(query):
                response += chunk.content

            revised_context_docs = []
            for doc in context_docs:
                if doc.metadata['source'] not in revised_context_docs:
                    revised_context_docs.append(doc.metadata['source'])

            return response, revised_context_docs

        else:
            system_prompt = '''
            You are an intelligent assistant designed to provide clear, accurate, and helpful responses. 
            Focus on understanding user intent, give concise answers, and offer step-by-step solutions when necessary.
            Be friendly, professional, and avoid unnecessary information.\n'''

            input_prompt = f'Query: {query_text}'

            response = generate_openai_response(input_prompt, system_prompt)
            return response, 'GPT Response'

    def query(self, query_text, k=1, reranker=None):
        try:
            return self.__generate_response(query_text=query_text, k=k, reranker=reranker, reranker_docs=k//2)
        except Exception as e:
            print(f"Failed to retrieve documents: {str(e)}")
            return None