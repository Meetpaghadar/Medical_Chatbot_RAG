from logger import logger

def query_chain(chain, user_input: str):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        
        # RetrievalQA input key can be version-dependent ("query" vs "question").
        # Try both to keep the API stable.
        try:
            result = chain.invoke({"query": user_input})
        except Exception:
            result = chain.invoke({"question": user_input})

        answer = result.get("result", "")
        source_documents = result.get("source_documents", [])
        sources = []
        for doc in source_documents:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source") or metadata.get("file_path") or "unknown"
            sources.append(source)

        response = {
            "response": answer,
            "sources": list(dict.fromkeys(sources))
        }
        
        logger.debug(f"Chain response: {response}")
        return response
        
    except Exception as e:
        logger.exception("Error on query chain")
        raise