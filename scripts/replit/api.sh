#!/bin/bash
poetry run python -m quart --app api.app run --host=0.0.0.0 --port=5000 