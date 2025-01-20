from quart import jsonify, request
from knowledge_agents.run import run_knowledge_agents
from knowledge_agents.model_ops import ModelOperation, ModelProvider, KnowledgeAgent
from knowledge_agents import KnowledgeAgentConfig
from knowledge_agents.data_processing.cloud_handler import S3Handler
import aioboto3, openai, logging, traceback, time, os
from typing import Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pytz

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
        logger.info("Initiating OpenAI health check")
        client = KnowledgeAgent().models['openai']
        logger.info("OpenAI client initialized, attempting to list models")
        models = await client.models.list()
        logger.info("Successfully listed OpenAI models")
        return True
    except Exception as e:
        logger.error(f"OpenAI health check failed: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def register_routes(app):
    @app.route('/', methods=['GET'])
    async def root():
        """Root endpoint providing API information."""
        return jsonify({
            "name": "Knowledge Agents API",
            "version": "1.0",
            "health_check": "/health",
            "documentation": {
                "process_query": "/process_query",
                "batch_process": "/batch_process",
                "process_recent_query": "/process_recent_query",
                "health_check": "/health",
                "health_replit": "/health_replit",
                "connections": "/health/connections",
                "s3_health": "/health/s3",
                "provider_health": "/health/provider/{provider}"
            }
        })
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

    @app.post("/process_query")
    async def process_query() -> Dict[str, Any]:
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

            # Extract batch sizes with appropriate defaults
            embedding_batch_size = int(data.get('embedding_batch_size', Config.EMBEDDING_BATCH_SIZE))
            chunk_batch_size = int(data.get('chunk_batch_size', Config.CHUNK_BATCH_SIZE))
            summary_batch_size = int(data.get('summary_batch_size', Config.SUMMARY_BATCH_SIZE))

            # Create configuration
            config = KnowledgeAgentConfig(
                root_path=Path(knowledge_config['ROOT_PATH']),
                knowledge_base_path=Path(paths['knowledge_base']),
                all_data_path=Path(paths['all_data']),
                stratified_data_path=Path(paths['stratified']),
                sample_size=int(data.get('sample_size', Config.SAMPLE_SIZE)),
                embedding_batch_size=embedding_batch_size,
                chunk_batch_size=chunk_batch_size,
                summary_batch_size=summary_batch_size,
                max_workers=int(data.get('max_workers', Config.MAX_WORKERS)),
                providers={
                    ModelOperation.EMBEDDING: ModelProvider(data.get('embedding_provider', 'openai')),
                    ModelOperation.CHUNK_GENERATION: ModelProvider(data.get('chunk_provider', 'openai')),
                    ModelOperation.SUMMARIZATION: ModelProvider(data.get('summary_provider', 'openai'))
                }
            )

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
                root_path=Path(knowledge_config['ROOT_PATH']),
                knowledge_base_path=Path(paths['knowledge_base']),
                all_data_path=Path(paths['all_data']),
                stratified_data_path=Path(paths['stratified']),
                sample_size=int(data.get('sample_size', Config.SAMPLE_SIZE)),
                embedding_batch_size=int(data.get('embedding_batch_size', Config.EMBEDDING_BATCH_SIZE)),
                chunk_batch_size=int(data.get('chunk_batch_size', Config.CHUNK_BATCH_SIZE)),
                summary_batch_size=int(data.get('summary_batch_size', Config.SUMMARY_BATCH_SIZE)),
                max_workers=int(data.get('max_workers', Config.MAX_WORKERS)),
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

    @app.route('/process_recent_query', methods=['GET'])
    async def process_recent_query():
        """Process a query with preconfigured settings for recent data.
        
        This endpoint automatically processes data from the last 3 hours with a sample size of 1000.
        Query parameters:
        - force_refresh (optional): Whether to force refresh the data. Defaults to True
        """
        try:
            logger.info("Received process_recent_query request")
            
            # Load stored queries
            stored_queries_path = Path(Config.PROJECT_ROOT) / 'config' / 'stored_queries.yaml'
            try:
                with open(stored_queries_path, 'r') as f:
                    stored_queries = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading stored queries: {str(e)}")
                return jsonify({
                    "error": "Failed to load stored queries",
                    "message": str(e)
                }), 500
            
            # Get the query template
            try:
                query = stored_queries['query']['example'][0]
            except KeyError as e:
                logger.error(f"Missing required key in stored queries: {str(e)}")
                return jsonify({
                    "error": "Invalid stored queries format",
                    "message": f"Missing key: {str(e)}"
                }), 500
            
            # Get query parameters - default to True for force_refresh
            force_refresh = request.args.get('force_refresh', 'true').lower() != 'false'
            logger.info(f"Force refresh enabled: {force_refresh}")
            
            # Calculate date range for last 3 hours using UTC
            end_time = datetime.now(pytz.UTC)
            start_time = end_time - timedelta(hours=3)
            
            # Set the filter date in environment for data operations with UTC timezone
            filter_date = start_time.strftime('%Y-%m-%d %H:%M:%S+00:00')
            os.environ['FILTER_DATE'] = filter_date
            logger.info(f"Set filter date to: {filter_date} (UTC)")
            
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

            # Create configuration with preconfigured settings
            config = KnowledgeAgentConfig(
                root_path=Path(knowledge_config['ROOT_PATH']),
                knowledge_base_path=Path(paths['knowledge_base']),
                all_data_path=Path(paths['all_data']),
                stratified_data_path=Path(paths['stratified']),
                sample_size=1000,  # Fixed sample size for recent data
                embedding_batch_size=Config.EMBEDDING_BATCH_SIZE,
                chunk_batch_size=Config.CHUNK_BATCH_SIZE,
                summary_batch_size=Config.SUMMARY_BATCH_SIZE,
                max_workers=Config.MAX_WORKERS,
                providers={
                    ModelOperation.EMBEDDING: ModelProvider.OPENAI,
                    ModelOperation.CHUNK_GENERATION: ModelProvider.OPENAI,
                    ModelOperation.SUMMARIZATION: ModelProvider.OPENAI
                }
            )

            # Process the query
            logger.info(f"Processing query with force_refresh={force_refresh}")
            chunks, response = await run_knowledge_agents(
                query=query,
                config=config,
                force_refresh=force_refresh
            )

            logger.info("Successfully processed recent query")
            return jsonify({
                "success": True,
                "results": {
                    "query": query,
                    "time_range": {
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat()
                    },
                    "chunks": chunks,
                    "summary": response
                }
            })

        except Exception as e:
            logger.error(f"Error processing recent query: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Determine if this is a known error type
            error_type = type(e).__name__
            if error_type in ['ValueError', 'KeyError', 'TypeError']:
                status_code = 400  # Bad request
            else:
                status_code = 500  # Internal server error
                
            return jsonify({
                "error": "Error processing query",
                "message": str(e),
                "type": error_type,
                "details": {
                    "traceback": traceback.format_exc()
                } if app.debug else {}
            }), status_code

    @app.errorhandler(Exception)
    async def handle_exception(e):
        """Global exception handler for all routes"""
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "type": str(type(e).__name__)
        }), 500