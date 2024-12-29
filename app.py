from flask import Flask, request, jsonify
from knowledge_agents.run import run_knowledge_agents
from knowledge_agents.model_ops import ModelOperation, ModelProvider
import logging

app = Flask(__name__)

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "message": "Service is running"
    })

@app.route('/process_query', methods=['POST'])
async def process_query():
    """Process a query using the knowledge agents pipeline."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing required parameter: query"
            }), 400

        # Extract parameters from request
        query = data['query']
        process_new = data.get('process_new', False)
        batch_size = data.get('batch_size', 100)
        max_workers = data.get('max_workers', None)
        
        # Get provider configuration if specified
        providers = {}
        if data.get('embedding_provider'):
            providers[ModelOperation.EMBEDDING] = ModelProvider(data['embedding_provider'])
        if data.get('chunk_provider'):
            providers[ModelOperation.CHUNK_GENERATION] = ModelProvider(data['chunk_provider'])
        if data.get('summary_provider'):
            providers[ModelOperation.SUMMARIZATION] = ModelProvider(data['summary_provider'])

        # Get the coroutine and await it
        coroutine = run_knowledge_agents(
            query=query,
            process_new=process_new,
            batch_size=batch_size,
            max_workers=max_workers,
            providers=providers if providers else None
        )
        chunks, response = await coroutine
        
        return jsonify({
            "success": True,
            "results": {
                "query": query,
                "chunks": chunks,
                "summary": response
            }
        })

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

@app.route('/batch_process', methods=['POST'])
async def batch_process():
    """Process multiple queries in batch using the knowledge agents pipeline."""
    try:
        data = request.get_json()
        if not data or 'queries' not in data:
            return jsonify({
                "error": "Missing required parameter: queries"
            }), 400

        # Extract parameters from request
        queries = data['queries']
        process_new = data.get('process_new', False)
        batch_size = data.get('batch_size', 100)
        max_workers = data.get('max_workers', None)
        
        # Get provider configuration if specified
        providers = {}
        if data.get('embedding_provider'):
            providers[ModelOperation.EMBEDDING] = ModelProvider(data['embedding_provider'])
        if data.get('chunk_provider'):
            providers[ModelOperation.CHUNK_GENERATION] = ModelProvider(data['chunk_provider'])
        if data.get('summary_provider'):
            providers[ModelOperation.SUMMARIZATION] = ModelProvider(data['summary_provider'])
        
        # Process each query and collect results
        results = []
        for query in queries:
            # Get the coroutine and await it
            coroutine = run_knowledge_agents(
                query=query,
                process_new=process_new,
                batch_size=batch_size,
                max_workers=max_workers,
                providers=providers if providers else None
            )
            chunks, response = await coroutine
            results.append({
                "query": query,
                "summary": response
            })
        
        return jsonify({
            "success": True,
            "results": results
        })

    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
