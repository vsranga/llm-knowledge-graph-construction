import os
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.graphs.graph_document import Node, Relationship
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Prompt user for PDF file path
FILE_PATH = input("Enter the path to the PDF file: ").strip()

# llm = ChatOpenAI(
#    base_url=os.getenv('HUGGINGFACE_ENDPOINT_URL'),
#    openai_api_key=os.getenv('HUGGINGFACE_API_TOKEN')
#)

llm = ChatOpenAI(
    model_name="tgi",
    openai_api_key=os.getenv('HUGGINGFACE_API_TOKEN'),
    openai_api_base=os.getenv('HUGGINGFACE_ENDPOINT_URL')+'/v1/'
)

hf_llm = HuggingFaceEndpoint(
   endpoint_url=os.getenv('HUGGINGFACE_ENDPOINT_URL'),
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_TOKEN'),
    task="text-classification"
)

embedding_provider = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

template = ChatPromptTemplate([
    ("system", "Extract entities and relationship in the text and return them in json format"),
])

doc_transformer = LLMGraphTransformer( 
    llm=llm          
    )
                                                        
# Check if file exists
if not os.path.exists(FILE_PATH):
    print(f"Error: File '{FILE_PATH}' not found.")
    sys.exit(1)

# Load and split the documents
loader = PyPDFLoader(FILE_PATH)

text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200,
)

docs = loader.load()
chunks = text_splitter.split_documents(docs)

for chunk in chunks:


    filename = os.path.basename(chunk.metadata["source"])
    chunk_id = f"{filename}.{chunk.metadata['page']}"
    print("Processing -", chunk_id)

    # Embed the chunk
    chunk_embedding = embedding_provider.embed_query(chunk.page_content)

    print("Chunk embedding: ",  chunk_embedding[:10] if chunk_embedding else "None"
          )

    # Add the Document and Chunk nodes to the graph
    properties = {
        "filename": filename,
        "chunk_id": chunk_id,
        "text": chunk.page_content,
        "embedding": chunk_embedding
    }
    
    graph.query("""
        MERGE (d:Document {id: $filename})
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text
        MERGE (d)<-[:PART_OF]-(c)
        WITH c
        CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, 
        properties
    )

    # Generate the entities and relationships from the chunk
    graph_docs = doc_transformer.convert_to_graph_documents([chunk])

    print("Graph documents generated: ", len(graph_docs))
    print(graph_docs[0].nodes if graph_docs else "No graph documents generated")
    # Map the entities in the graph documents to the chunk node
    for graph_doc in graph_docs:
        chunk_node = Node(
            id=chunk_id,
            type="Chunk"
        )

        print('Number of Nodes in this chunk: ', len(graph_doc.nodes))
        for node in graph_doc.nodes:

            print("Node: ", node.id, node.type, node.properties if node.properties else ""      )

            graph_doc.relationships.append(
                Relationship(
                    source=chunk_node,
                    target=node, 
                    type="HAS_ENTITY"
                    )
                )
            
        print('Number of relationships in this chunk: ', len(graph_doc.relationships))

        for relationship in graph_doc.relationships:
            print("Relationship: ", relationship.source.id, 
                  relationship.target.id, relationship.type)    
            
    # add the graph documents to the graph
    graph.add_graph_documents(graph_docs)
  

# Create the vector index
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {indexConfig: {
    `vector.dimensions`: 1024,
    `vector.similarity_function`: 'cosine'
    }};""")
