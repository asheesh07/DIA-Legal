class DIAPipeline:
    def __init__(
            self,
            retriever,
            context_builder,
            llm_answerer,
            query_router
            ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.llm_answerer = llm_answerer
        self.query_router = query_router
        
    def run(
        self,
        query,
        case_id
        ):
        routing = self.query_router.route(query)
        mode = routing['mode']
        
        retrieved_items = self.retriever.retrieve(
            case_id,
            query,
            top_k = 9,

        )
        retrieval_confidence = self.retriever._estimate_confidence(retrieved_items)
        
        context_result = self.context_builder.build(
            query,
            retrieved_items,
            mode
            
        )
        progressive_context = self.llm_answerer.get_progressive_context()
        
        if progressive_context:
            context_result["context"] += "\n\n" + progressive_context
            
        result = self.llm_answerer.answer(
            context_result,
            retrieval_confidence,
            mode)
        
        return result
        
        