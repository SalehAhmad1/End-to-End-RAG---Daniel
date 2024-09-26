import getpass
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

def genetare_openai_response(input_prompt):
    print(f'In genetare_openai_response')
    system_prompt = '''You are an assistant designed to provide answers when no (0) relevant documents are retrieved from the vector database. When this happens, you should follow these steps:
                    1) First, determine if you can answer the user's query using general knowledge or internal information. If so, generate a confident, helpful response in a straightforward narrative style. Do not use phrases such as 'According to me,' 'As of my knowledge,' 'I don’t know but,' or mention knowledge cutoffs or lack of information. Simply provide the answer as if you are certain of the facts.
                    2) If the question is domain-specific, too specific (e.g., about a particular person or object that could mislead), or outside your knowledge, do not attempt to answer. Politely respond with: 'I'm sorry, I currently do not have enough information to answer your question.'''
    llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    return 'The number of retrieved documents from RAG pipeline was 0, so the answer is based on LLM\s internal knowledge.\n' + llm(system_prompt+input_prompt).content
    
class ChatbotDataQuery:
    def __init__(self, vector_store):
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = '''You are Wagner, a highly intelligent and friendly AI assistant. You are an assistant who helps answer queries about Daniel Ringel. WHen asked about you, simply asnwer about yourself and nothing else.
        For example:
        Input: Who are you?
        Answer: I am Wagner, a highly intelligent and friendly AI assistant. I am an assistant who helps answer queries.
        
        Input: What is your name?
        Answer: My name is Wagner.
        
        Input: How old are you?
        Answer: Sorry, I don't have an age as I am an AI assistant.
        
        Input: What is my name?
        Answer: My name is Wagner.'''

        if vector_store is None:
            raise ValueError("Vector store cannot be None")
        else:
            self.vector_store = vector_store

    def initialize_reranker(self):
        """
        Initialize the custom reranker.
        """
        return CustomReranker()

    def __generate_response(self, query_text, retriever, reranker=None, reranker_docs=0):
        context_docs = retriever.invoke(query_text)
        if len(context_docs) == 0:
            response = genetare_openai_response(input_prompt=query_text)
            return response

        context_docs_texts = [doc.page_content for doc in context_docs]

        if reranker is not None and reranker_docs > 0:
            # Use the custom reranker to rerank the context_docs
            relevant_docs = reranker.rerank(query_text, context_docs_texts, k=reranker_docs)

            All_Scores = [doc['score'] for doc in relevant_docs]
            Min = min(All_Scores)
            Max = max(All_Scores)
            Normalized_Scores = [(doc['score'] - Min) / (Max - Min) for doc in relevant_docs]
            
            for idx,doc in enumerate(relevant_docs):
                doc['score'] = Normalized_Scores[idx]

            final_reranked_docs = []
            for reranked_doc in relevant_docs:
                if reranked_doc['score'] < 0.50:
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

        # print(f'---\nThe Retrieved Documents are:')
        # for idx, doc in enumerate(context_docs):
        #     print(idx, '-', doc.metadata)
        # print('---\n\n')

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
        return {'response': response, 'context_docs': context_docs}
            # yield chunk.content
        # return context_docs

    def query(self, query_text, k=1, reranker=None):
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k},
            search_type="similarity",
        )
        try:
            return self.__generate_response(query_text=query_text, retriever=retriever, reranker=reranker, reranker_docs=k//2)
        except Exception as e:
            print(f"Failed to retrieve documents: {str(e)}")
            return None