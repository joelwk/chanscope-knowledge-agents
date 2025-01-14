from quart import jsonify, request
from knowledge_agents.run import run_knowledge_agents
from knowledge_agents.model_ops import ModelOperation, ModelProvider, KnowledgeAgent
from knowledge_agents import KnowledgeAgentConfig
from knowledge_agents.data_processing.cloud_handler import S3Handler
import aioboto3, openai, logging, traceback, time, os

from config.settings import Config
logger = logging.getLogger(__name__)

async def check_service_connection(service_fn):
    """Helper to check service connection with latency."""
    start_time = time.time()
    try:
        await service_fn()
        status = "connected"
    except Exception as e:
        logger.error(f"Service connection error: {str(e)}")
        status = "error"
    latency = round((time.time() - start_time) * 1000, 2)
    return {"status": status, "latency": latency}

async def check_openai() -> bool:
    """Check if OpenAI API is accessible."""
    try:
        client = KnowledgeAgent().models['openai']
        models = await client.models.list()
        return True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {str(e)}")
        return False

def register_routes(app):
    @app.route('/health', methods=['GET'])
    async def health_check():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "message": "Service is running",
            "environment": {
                "QUART_ENV": app.config.get('ENV', 'development'),
                "DEBUG": app.debug
            }
        })

    @app.route('/health_replit', methods=['GET'])
    async def health_check_replit():
        """Basic health check endpoint with Replit-specific info."""
        return jsonify({
            "status": "Replit healthy",
            "message": "Replit Service is running",
            "environment": {
                "QUART_ENV": os.getenv('QUART_ENV', 'development'),
                "DEBUG": app.debug,
                "REPL_SLUG": os.getenv('REPL_SLUG'),
                "REPL_OWNER": os.getenv('REPL_OWNER'),
                "REPL_ID": os.getenv('REPL_ID')
            },
            "service_url": f"https://{os.getenv('REPL_SLUG')}.{os.getenv('REPL_OWNER')}.repl.co"
        })

    @app.route('/health/connections', methods=['GET'])
    async def service_connections():
        """Check status of all service connections."""
        config = app.config['KNOWLEDGE_CONFIG']
        agent = KnowledgeAgent()
        agent.model_config = config  # Initialize agent with app config

        async def check_openai():
            client = agent._get_client(ModelProvider.OPENAI)
            models = await client.models.list()
            return models

        async def check_s3():
            s3_handler = S3Handler()
            session = aioboto3.Session()
            async with session.client('s3', 
                                    region_name=s3_handler.region_name,
                                    aws_access_key_id=s3_handler.aws_access_key_id,
                                    aws_secret_access_key=s3_handler.aws_secret_access_key) as s3_client:
                await s3_client.head_bucket(Bucket=s3_handler.bucket_name)

        services = {
            "s3": await check_service_connection(check_s3),
            "openai": await check_service_connection(check_openai),
        }

        return jsonify({"services": services})

    @app.route('/health/s3', methods=['GET'])
    async def s3_health():
        """Check S3 connection and bucket access."""
        start_time = time.time()

        try:
            # Initialize S3Handler
            s3_handler = S3Handler()
            session = aioboto3.Session()

            async with session.client('s3', 
                                    region_name=s3_handler.region_name,
                                    aws_access_key_id=s3_handler.aws_access_key_id,
                                    aws_secret_access_key=s3_handler.aws_secret_access_key) as s3_client:
                # Test bucket access
                try:
                    await s3_client.head_bucket(Bucket=s3_handler.bucket_name)
                    bucket_exists = True
                except s3_client.exceptions.ClientError:
                    bucket_exists = False

                # Get bucket details if it exists
                bucket_details = {}
                if bucket_exists:
                    try:
                        response = await s3_client.list_objects_v2(
                            Bucket=s3_handler.bucket_name,
                            Prefix=s3_handler.bucket_prefix,
                            MaxKeys=1
                        )
                        bucket_details = {
                            "prefix": s3_handler.bucket_prefix,
                            "region": s3_handler.region_name,
                            "has_contents": 'Contents' in response
                        }
                    except Exception as e:
                        logger.warning(f"Could not fetch bucket details: {str(e)}")

            return jsonify({
                "s3_status": "connected" if bucket_exists else "bucket_not_found",
                "bucket_access": bucket_exists,
                "bucket_name": s3_handler.bucket_name,
                "bucket_details": bucket_details,
                "aws_region": s3_handler.region_name,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })

        except Exception as e:
            logger.error(f"S3 health check error: {str(e)}")
            error_details = {
                "message": str(e),
                "type": e.__class__.__name__
            }
            if hasattr(e, 'response'):
                error_details["status_code"] = e.response.get('Error', {}).get('Code', 'Unknown')

            return jsonify({
                "s3_status": "error",
                "bucket_access": False,
                "error": error_details,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }), 500

    @app.route('/health/provider/<provider>', methods=['GET'])
    async def provider_health(provider):
        """Check specific provider health status."""
        start_time = time.time()

        try:
            # Validate provider
            try:
                provider_enum = ModelProvider.from_str(provider)
            except ValueError:
                return jsonify({
                    "error": f"Invalid provider: {provider}"
                }), 400

            # Initialize agent with app config
            config = app.config['KNOWLEDGE_CONFIG']
            agent = KnowledgeAgent()
            agent.model_config = config  # Initialize agent with app config

            # Get provider client and check connection
            client = agent._get_client(provider_enum)

            # Basic connection check based on provider
            if provider_enum == ModelProvider.OPENAI:
                models = await client.models.list()
            elif provider_enum == ModelProvider.GROK:
                # Add Grok-specific check if available
                pass
            elif provider_enum == ModelProvider.VENICE:
                # Add Venice-specific check if available
                pass

            return jsonify({
                "status": "connected",
                "provider": provider,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })
        except Exception as e:
            logger.error(f"Provider health check error: {str(e)}")
            return jsonify({
                "status": "error",
                "provider": provider,
                "error": str(e),
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }), 500

    @app.route('/process_query', methods=['POST'])
    async def process_query():
        """Process a query using the knowledge agents pipeline."""
        try:
            logger.info("Received process_query request")
            data = await request.get_json()

            if not data:
                logger.error("No JSON data received")
                return jsonify({"error": "No JSON data received"}), 400

            if 'query' not in data:
                logger.error("Missing required parameter: query")
                return jsonify({"error": "Missing required parameter: query"}), 400

            # Get base configuration and validate it exists
            if 'KNOWLEDGE_CONFIG' not in app.config:
                error_msg = "KNOWLEDGE_CONFIG not found in application configuration"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            knowledge_config = app.config['KNOWLEDGE_CONFIG']

            # Validate required configuration fields
            required_fields = ['PATHS', 'ROOT_PATH', 'PROVIDERS']
            missing_fields = [field for field in required_fields if field not in knowledge_config]
            if missing_fields:
                error_msg = f"Missing required configuration fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            paths = knowledge_config['PATHS']

            # Validate required paths
            required_paths = ['knowledge_base', 'all_data', 'stratified', 'temp']
            missing_paths = [path for path in required_paths if path not in paths]
            if missing_paths:
                error_msg = f"Missing required paths in configuration: {', '.join(missing_paths)}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            config = KnowledgeAgentConfig(
                root_path=knowledge_config['ROOT_PATH'],
                knowledge_base_path=paths['knowledge_base'],
                all_data_path=paths['all_data'],
                stratified_data_path=paths['stratified'],
                temp_path=paths['temp'],
                sample_size=int(data.get('sample_size', knowledge_config.get('DEFAULT_SAMPLE_SIZE', 2500))),
                batch_size=int(data.get('batch_size', knowledge_config.get('DEFAULT_BATCH_SIZE', 100))),
                max_workers=int(data.get('max_workers') or knowledge_config.get('DEFAULT_MAX_WORKERS') or 4),
                providers=knowledge_config['PROVIDERS'].copy()
            )

            # Update providers if specified
            for op in [ModelOperation.EMBEDDING, ModelOperation.CHUNK_GENERATION, ModelOperation.SUMMARIZATION]:
                provider_key = f"{op.value}_provider"
                if provider_key in data:
                    config.providers[op] = ModelProvider(data[provider_key])

            # Process the query
            chunks, response = await run_knowledge_agents(
                query=data['query'],
                config=config,
                force_refresh=data.get('force_refresh', False)
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
            data = await request.get_json()

            if not data:
                logger.error("No JSON data received")
                return jsonify({"error": "No JSON data received"}), 400

            if 'queries' not in data:
                logger.error("Missing required parameter: queries")
                return jsonify({"error": "Missing required parameter: queries"}), 400

            # Get base configuration and validate it exists
            if 'KNOWLEDGE_CONFIG' not in app.config:
                error_msg = "KNOWLEDGE_CONFIG not found in application configuration"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            knowledge_config = app.config['KNOWLEDGE_CONFIG']

            # Validate required configuration fields
            required_fields = ['PATHS', 'ROOT_PATH', 'PROVIDERS']
            missing_fields = [field for field in required_fields if field not in knowledge_config]
            if missing_fields:
                error_msg = f"Missing required configuration fields: {', '.join(missing_fields)}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            paths = knowledge_config['PATHS']

            # Validate required paths
            required_paths = ['knowledge_base', 'all_data', 'stratified', 'temp']
            missing_paths = [path for path in required_paths if path not in paths]
            if missing_paths:
                error_msg = f"Missing required paths in configuration: {', '.join(missing_paths)}"
                logger.error(error_msg)
                return jsonify({"error": error_msg}), 500

            config = KnowledgeAgentConfig(
                root_path=knowledge_config['ROOT_PATH'],
                knowledge_base_path=paths['knowledge_base'],
                all_data_path=paths['all_data'],
                stratified_data_path=paths['stratified'],
                temp_path=paths['temp'],
                sample_size=int(data.get('sample_size', knowledge_config.get('DEFAULT_SAMPLE_SIZE', 2500))),
                batch_size=int(data.get('batch_size', knowledge_config.get('DEFAULT_BATCH_SIZE', 100))),
                max_workers=int(data.get('max_workers') or knowledge_config.get('DEFAULT_MAX_WORKERS') or 4),
                providers=knowledge_config['PROVIDERS'].copy()
            )

            # Update providers if specified
            for op in [ModelOperation.EMBEDDING, ModelOperation.CHUNK_GENERATION, ModelOperation.SUMMARIZATION]:
                provider_key = f"{op.value}_provider"
                if provider_key in data:
                    config.providers[op] = ModelProvider(data[provider_key])

            # Process each query
            results = []
            for query in data['queries']:
                chunks, response = await run_knowledge_agents(
                    query=query,
                    config=config,
                    force_refresh=data.get('force_refresh', False)
                )
                results.append({
                    "query": query,
                    "chunks": chunks,
                    "summary": response
                })

            logger.info("Successfully processed batch queries")
            return jsonify({
                "success": True,
                "results": results
            })

        except Exception as e:
            logger.error(f"Error processing batch queries: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "error": "Internal server error",
                "message": str(e),
                "type": type(e).__name__
            }), 500