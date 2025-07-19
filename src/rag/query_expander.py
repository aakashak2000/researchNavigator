class QueryExpander:
    def __init__(self, llm):
        self.llm = llm

    def get_llm_expressions(self, query):

        prompt = f"""
Given this research query, suggest 3-5 related technical terms that would appear in academic papers on this topic. Only return the terms, separated by spaces.

Query: {query}

Related Terms: """
        
        return self.llm.invoke(prompt)
        

    def expand_query(self, query):

        llm_additions = self.get_llm_expressions(query)
        
        return f"{query} {llm_additions}"