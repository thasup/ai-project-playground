from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

emb = embeddings.embed_query("hi there")

print(emb)

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,
    chunk_overlap=0
  )

loader = TextLoader("./documents/andreas_workshop.txt")
documents = loader.load_and_split(text_splitter=text_splitter)

db = Chroma.from_documents(
  documents,
  embedding=embeddings,
  persist_directory="emb",
  collection_name="andreas_workshop"
)

results = db.max_marginal_relevance_search_by_vector(
  "How can I categorize things based on priority?",
  k=4
)

for result in results:
    print("\n")
    print(result[1])
    print(result[0].page_content)