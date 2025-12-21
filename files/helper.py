from llama_index.core import SimpleDirectoryReader, SummaryIndex, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import QueryEngineTool
from typing import Tuple
from pathlib import Path


async def create_doc_tools(
    doc_name: str,
    document_fp: Path
) -> Tuple[QueryEngineTool, QueryEngineTool]:
    """
    Create vector and summary tools for a given document.
    
    Args:
        doc_name: Name of the document (typically the file stem)
        document_fp: Path to the document file
        
    Returns:
        Tuple of (vector_tool, summary_tool)
    """
    # Load the document
    documents = SimpleDirectoryReader(input_files=[str(document_fp)]).load_data()
    
    # Split into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create indexes
    summary_index = SummaryIndex(nodes=nodes)
    vector_index = VectorStoreIndex(nodes=nodes)
    
    # Create query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    
    vector_query_engine = vector_index.as_query_engine()
    
    # Create tools with document-specific descriptions
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name=f"{doc_name}_summary_tool",
        description=(
            f"Useful for summarization questions related to the {doc_name} paper."
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name=f"{doc_name}_vector_tool",
        description=(
            f"Useful for retrieving specific context from the {doc_name} paper."
        ),
    )
    
    return vector_tool, summary_tool


def get_doc_tools(
    document_fp: str,
    doc_name: str
) -> Tuple[QueryEngineTool, QueryEngineTool]:
    """
    Create vector and summary tools for a given document.
    
    Args:
        document_fp: Path to the document file (as string)
        doc_name: Name of the document (typically the file stem)
        
    Returns:
        Tuple of (vector_tool, summary_tool)
    """
    # Load the document
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()
    
    # Split into nodes
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Create indexes
    summary_index = SummaryIndex(nodes=nodes)
    vector_index = VectorStoreIndex(nodes=nodes)
    
    # Create query engines
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    
    vector_query_engine = vector_index.as_query_engine()
    
    # Create tools with document-specific descriptions
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        name=f"{doc_name}_summary_tool",
        description=(
            f"Useful for summarization questions related to the {doc_name} paper."
        ),
    )
    
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        name=f"{doc_name}_vector_tool",
        description=(
            f"Useful for retrieving specific context from the {doc_name} paper."
        ),
    )
    
    return vector_tool, summary_tool