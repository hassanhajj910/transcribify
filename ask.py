from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms import HuggingFaceHub



from langchain.document_loaders import TextLoader, DirectoryLoader
loader = DirectoryLoader('./data/corpus', glob="**/*.txt")

