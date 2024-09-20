from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import getpass
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class ChatbotDataQuery:
    def __init__(self, vector_store):
        """
        Initialize the ChatbotDataQuery with the provided vector store.
        Raise an exception if the vector store is None.
        """
        self.llm = ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))

        if vector_store is None:
            raise ValueError("Vector store cannot be None")
        else:
            self.vector_store = vector_store

    def __generate_response(self, query_text, retriever):
        # context = "\n\n".join([doc.page_content for doc in retriever.invoke(query_text)])
        # if context in ["", None]:
        #     raise ValueError("No context found")

        context_docs = retriever.invoke(query_text)
        if len(context_docs) == 0:
            raise ValueError("No context found")

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant that only answers questions about the context. "
            "If you don't know the answer, just say you don't know. "
            "The context is:\n\n{context}\n\n"
            "Question: {question}\n"
            "Helpful answer in markdown."
        )

        print(f'The Retrieved Documents are:')
        for idx,doc in enumerate(context_docs):
            print(idx, '-', doc.metadata)

        chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=prompt,
            document_variable_name="context",
        )

        return chain.invoke({"question": query_text, "context": context_docs})

    def query(self, query_text, k=1):
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
        )
        try:
            return self.__generate_response(query_text=query_text, retriever=retriever)
        except Exception as e:
            print(f"Failed to retrieve documents: {str(e)}")
            return None
