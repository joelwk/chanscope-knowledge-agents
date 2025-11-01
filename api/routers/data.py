"""Data management endpoints."""
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from knowledge_agents.data_ops import DataOperations
from api.errors import ConfigurationError
from api.models import StratificationResponse
from config.settings import Config
from config.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["data"])

# Dependency for DataOperations
async def get_data_ops() -> DataOperations:
    from knowledge_agents.data_ops import DataConfig, DataOperations
    data_config = DataConfig.from_config()
    return DataOperations(data_config)

@router.post("/data/stratify", response_model=StratificationResponse)
async def stratify_data_endpoint(
    data_ops: DataOperations = Depends(get_data_ops)
) -> StratificationResponse:
    """Create stratified sample from complete data."""
    try:
        paths = Config.get_paths()
        sample_settings = Config.get_sample_settings()
        column_settings = Config.get_column_settings()

        status = data_ops.verify_data_structure()
        if not status.get('complete_data', False):
            data_ops.update_complete_data()
            status = data_ops.verify_data_structure()
            if not status.get('complete_data', False):
                raise HTTPException(status_code=500, detail="Failed to create complete dataset.")

        complete_data_path = Path(paths['root_data_path']) / 'complete_data.csv'
        complete_data = pd.read_csv(complete_data_path)
        if complete_data.empty:
            raise HTTPException(status_code=400, detail="Complete dataset is empty.")

        from knowledge_agents.data_processing.sampler import Sampler
        sampler = Sampler(
            time_column=column_settings['time_column'],
            strata_column=column_settings['strata_column'],
            initial_sample_size=sample_settings['default_sample_size']
        )
        stratified_data = sampler.stratified_sample(complete_data)
        stratified_file = Path(paths['stratified']) / "stratified_sample.csv"
        stratified_file.parent.mkdir(parents=True, exist_ok=True)
        stratified_data.to_csv(stratified_file, index=False)
        return StratificationResponse(
            status="success",
            message='Stratification completed successfully',
            timestamp=datetime.now(),
            total_records=len(complete_data),
            stratified_records=len(stratified_data),
            sample_size=len(stratified_data),
            stratification_details={
                "stratified_rows": len(stratified_data),
                "stratified_file": str(stratified_file)
            }
        )
    except Exception as e:
        logger.error(f"Error during stratification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/data/prepare")
async def prepare_data_endpoint(request: Request):
    """Prepare data for inference."""
    try:
        from config.chanscope_config import ChanScopeConfig
        config = ChanScopeConfig.from_env()

        # Initialize storage with environment detection
        from config.storage import StorageFactory
        storage = StorageFactory.create(config)

        # TODO: Implement this
        return {"status": "not_implemented", "message": "Data preparation endpoint not yet implemented"}
    except Exception as e:
        from api.errors import APIError
        raise APIError(
            message=f"Error preparing data: {str(e)}",
            status_code=500,
            error_code="DATA_PREPARATION_ERROR",
            details={"error": str(e)}
        )

# Keep the old /stratify endpoint for backward compatibility
@router.post("/stratify", response_model=StratificationResponse)
async def stratify_data_legacy(
    data_ops: DataOperations = Depends(get_data_ops)
) -> StratificationResponse:
    """Create stratified sample (legacy endpoint for backward compatibility)."""
    return await stratify_data_endpoint(data_ops)

