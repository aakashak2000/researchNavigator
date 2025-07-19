class QueryExpander:
    def __init__(self):
        pass

    def expand_query(self, query):
        return query + self.add_context(query)
    
    def add_context(self, query):
        
        expansion_dict = {
        "attention": "attention mechanism self-attention multi-head attention scaled dot-product attention",
        "transformer": "transformer architecture encoder decoder positional encoding",
        "neural network": "neural networks deep learning artificial neural networks",
        "optimization": "optimization gradient descent adam optimizer learning rate",
        "efficiency": "computational complexity inference speed memory usage FLOPs throughput",
        "performance": "accuracy metrics evaluation benchmarks results"
    }
        
        context = ""
        query = query.lower()
        for key_term, related_terms in expansion_dict.items():
            if key_term in query:
                context += " " +related_terms

        return context