import os
from uuid import uuid4
from langchain_core.documents import Document
from data_loader import ChatbotDataLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import SpacyTextSplitter


class ChatbotDataIngester:
    def __init__(self, vector_store, embeddings):
        """
        Initialize the ChatbotDataIngester with an external vector store and embeddings model.
        Raise an exception if either of them is None.
        """
        if vector_store in [None, '']:
            raise ValueError("Vector store cannot be None/empty")
        if embeddings in [None, '']:
            raise ValueError("Embeddings model cannot be None/empty")

        self.loader = ChatbotDataLoader()
        self.vector_store = vector_store
        self.embeddings = embeddings
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

    def embed_content(self, content):
        """
        Embed the text content using the provided embedding model.
        """
        return self.embeddings.embed_query(content)

    def load_and_ingest(self, dir_path, empty_db=False):
        """
        Load documents from the directory, generate embeddings, and ingest them into the vector store.
        
        :param dir_path: Directory path to load the documents from.
        :param empty_db: If True, the vector store will be emptied before adding new documents.
        """
        # Optionally clear the vector store
        if empty_db:
            self.clear_vector_store()

        # Load files from the directory
        file_contents = self.loader.load_directory(dir_path)

        # Create documents from the file contents
        documents = [
            Document(page_content=content, metadata={"source": file_path})
            for file_path, content in file_contents.items()
        ]

        print(f'{len(documents)} documents loaded from the database')

        split_docs = self.text_splitter.split_documents(documents)

        # Generate UUIDs for documents
        uuids = [str(uuid4()) for _ in range(len(split_docs))]

        print(f'{len(documents)} documents splitted into {len(split_docs)} chunks')

        # Ingest documents into the vector store
        self.ingest_to_vector_store(split_docs, uuids)

    def clear_vector_store(self):
        """
        Clear all documents in the vector store.
        """
        try:
            current_index = self.vector_store.get_pinecone_index('test')
            check = False
            for ids in current_index.list(namespace='default'):
                check = True
                break
            if not check:
                print("The vector store is already empty.")
                return
            else:
                self.vector_store.delete(delete_all=True)
                print("Cleared the vector store.")
        except Exception as e:
            print(f"Failed to clear the vector store: {str(e)}")

    def ingest_to_vector_store(self, documents, uuids):
        """
        Ingest the documents into the vector store.
        """
        try:
            self.vector_store.add_documents(documents, ids=uuids)
            print(f'Ingested {len(documents)} chunks to the vector store')
        except Exception as e:
            print(f'Failed to ingest documents: {str(e)}')