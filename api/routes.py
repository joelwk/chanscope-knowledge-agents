from flask import jsonify, request
from knowledge_agents.run import run_knowledge_agents
from knowledge_agents.model_ops import ModelOperation, ModelProvider
import logging
import traceback

# Setup logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def register_routes(app):
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "message": "Service is running",
            "environment": {
                "FLASK_ENV": app.config.get('ENV', 'development'),
                "DEBUG": app.debug
            }
        })

    @app.route('/process_query', methods=['POST'])
    async def process_query():
        """Process a query using the knowledge agents pipeline."""
        try:
            logger.info("Received process_query request")
            data = request.get_json()
            
            if not data:
                logger.error("No JSON data received")
                return jsonify({"error": "No JSON data received"}), 400
                
            if 'query' not in data:
                logger.error("Missing required parameter: query")
                return jsonify({"error": "Missing required parameter: query"}), 400

            # Get base configuration and update with request parameters
            config = app.config['KNOWLEDGE_CONFIG']
            if data.get('batch_size'):
                config.batch_size = int(data['batch_size'])
            if data.get('max_workers') is not None:
                config.max_workers = int(data['max_workers'])
            
            # Update providers if specified
            for op in [ModelOperation.EMBEDDING, ModelOperation.CHUNK_GENERATION, ModelOperation.SUMMARIZATION]:
                provider_key = f"{op.value}_provider"
                if provider_key in data:
                    config.providers[op] = ModelProvider(data[provider_key])

            # Process the query
            chunks, response = await run_knowledge_agents(
                query=data['query'],
                config=config,
                process_new=data.get('process_new', False)
            )
            
            logger.info("Successfully processed query")
            return jsonify({
                "success": True,
                "results": {
                    "query": data['query'],
                    "chunks": chunks,
                    "summary": response
                }
            })

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }), 500

    @app.route('/batch_process', methods=['POST'])
    async def batch_process():
        """Process multiple queries in batch using the knowledge agents pipeline."""
        try:
            logger.info("Received batch_process request")
            data = request.get_json()
            
            if not data:
                logger.error("No JSON data received")
                return jsonify({"error": "No JSON data received"}), 400
                
            if 'queries' not in data:
                logger.error("Missing required parameter: queries")
                return jsonify({"error": "Missing required parameter: queries"}), 400

            # Get base configuration and update with request parameters
            config = app.config['KNOWLEDGE_CONFIG']
            if data.get('batch_size'):
                config.batch_size = int(data['batch_size'])
            if data.get('max_workers') is not None:
                config.max_workers = int(data['max_workers'])
            
            # Update providers if specified
            for op in [ModelOperation.EMBEDDING, ModelOperation.CHUNK_GENERATION, ModelOperation.SUMMARIZATION]:
                provider_key = f"{op.value}_provider"
                if provider_key in data:
                    config.providers[op] = ModelProvider(data[provider_key])
            
            # Process each query and collect results
            results = []
            for query in data['queries']:
                chunks, response = await run_knowledge_agents(
                    query=query,
                    config=config,
                    process_new=data.get('process_new', False)
                )
                results.append({
                    "query": query,
                    "chunks": chunks,
                    "summary": response
                })
            
            logger.info("Successfully processed batch")
            return jsonify({
                "success": True,
                "results": results
            })

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }), 500 