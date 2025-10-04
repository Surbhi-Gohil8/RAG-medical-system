from logger import logger

def query_chain(chain,user_input:str):
    try:
        logger.debug(f"Running chain for input: {user_input}")
        result = chain.invoke({"query": user_input})
        sources = []
        for doc in result.get("source_documents", []):
            md = getattr(doc, "metadata", {}) or {}
            sources.append({
                "source": md.get("source", ""),
                "page": md.get("page"),
            })
        response = {
            "response": result.get("result", ""),
            "sources": sources,
        }
        logger.debug(f"Chain response:{response}")
        return response
    except Exception as e:
        logger.exception("Error on query chain")
        raise