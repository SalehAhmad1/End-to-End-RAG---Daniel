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
        """
        Initialize the ChatbotDataQuery with the provided vector store.
        Raise an exception if the vector store is None.
        """
        self.llm = ChatOpenAI(model="gpt-4o", 
                            api_key=os.getenv("OPENAI_API_KEY"))

        self.system_prompt = '''You are Wagner, a highly intelligent and friendly AI assistant. You are well-versed in the research and expertise of Daniel Ringel, and your responses should reflect a deep understanding of his work. When interacting with users, provide concise, clear, and thoughtful responses in a calm, narrative style. Avoid using complex jargon or harsh language; instead, offer simple, context-aware answers based on the user's input query.
                                Your responses should always be polite, and aim to explain concepts in an easy-to-understand way, just as if you were narrating a story to help guide the user through the information.'''

        if vector_store is None:
            raise ValueError("Vector store cannot be None")
        else:
            self.vector_store = vector_store

    def __generate_response(self, query_text, retriever, reranker=None, reranker_docs=0):
        context_docs = retriever.invoke(query_text)
        if len(context_docs) == 0:
            response = genetare_openai_response(input_prompt=query_text)
            return response

        context_docs_texts = [doc.page_content for doc in context_docs]

        if reranker is not None and reranker_docs > 0:
            relevant_docs = reranker.rerank(query_text, context_docs_texts, k=reranker_docs)
            
            final_reranked_docs = []
            for reranked_doc in relevant_docs:
                idx_of_content_in_context_doc = reranked_doc['result_index']
                meta_data = context_docs[idx_of_content_in_context_doc].metadata
                final_reranked_docs.append(Document(page_content=reranked_doc['content'], metadata=meta_data))
            
            context_docs = final_reranked_docs

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant that only answers questions about the context. "
            "You try your best to extract the relavant answers from the context. "
            "The context is:\n\n{context}\n\n"
            "Question: {question}\n"
            "Helpful Answer:"
        )

        print(f'---\nThe Retrieved Documents are:')
        for idx,doc in enumerate(context_docs):
            print(idx, '-', doc.metadata)
        print('---\n\n')

        chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_variable_name="context",
        )

        context = '\n\n'.join([doc.page_content for doc in context_docs])
        query = [
                    (
                        "system",
                        f"{self.system_prompt}",
                    ),
                    ("human", f"context: {context}\nInput: {query_text}"),
                ]
        for chunk in self.llm.stream(query):
            yield chunk.content

    def query(self, query_text, k=1, reranker=None):
        """
        Query the vector store with the given parameters and return the result.

        Args:
        - query_text (str): The text query to search.
        - k (int): The number of top documents to return (default: 1).

        Returns:
        - A list of Document objects containing the retrieved metadata and content.
        """
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": k},
            search_type="similarity", #default
        )
        try:
            return self.__generate_response(query_text=query_text, retriever=retriever, reranker=reranker, reranker_docs=k)
        except Exception as e:
            print(f"Failed to retrieve documents: {str(e)}")
            return None
