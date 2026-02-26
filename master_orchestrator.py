"""
Master Orchestrator - Core control system for autonomous trading network evolution
Coordinates all components, manages lifecycle, and ensures system stability
"""
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import firebase_admin
from firebase_admin import firestore, credentials
from dataclasses import dataclass, asdict
import json
import numpy as np
import pandas as pd
from enum import Enum
import traceback
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from architecture_generator import ArchitectureGenerator
from training_engine import TrainingEngine
from deployment_manager import DeploymentManager
from performance_monitor import PerformanceMonitor
from evolution_engine import EvolutionEngine
from data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('orchestrator.log')
    ]
)
logger = logging.getLogger(__name__)

class SystemState(Enum):
    """System operation states"""
    INITIALIZING = "initializing"
    GENERATING = "generating_architectures"
    TRAINING = "training_models"
    EVALUATING = "evaluating_performance"
    DEPLOYING = "deploying_models"
    EVOLVING = "evolving_population"
    ERROR = "error_state"
    MAINTENANCE = "maintenance_mode"

@dataclass
class NetworkGeneration:
    """Represents a generation of neural networks"""
    generation_id: str
    timestamp: datetime
    parent_generation: Optional[str]
    models_count: int
    best_performance: float
    metadata: Dict

class MasterOrchestrator:
    """Main control system for autonomous trading network evolution"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize orchestrator with Firebase and components"""
        logger.info("Initializing Master Orchestrator")
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize Firebase
        self._init_firebase()
        
        # Initialize components
        self.state = SystemState.INITIALIZING
        self.current_generation = None
        self.components = {}
        
        try:
            self._initialize_components()
            logger.info("Orchestrator initialization complete")
        except Exception as e:
            logger.error(f"Component initialization failed: {str(e)}")
            self.state = SystemState.ERROR
            raise
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration file with error handling"""
        default_config = {
            "firebase": {
                "credential_path": "firebase_credentials.json",
                "project_id": "trading-networks"
            },
            "training": {
                "generation_size": 10,
                "validation_split": 0.2,
                "epochs": 100
            },
            "evolution": {
                "selection_rate": 0.3,
                "mutation_rate": 0.1,
                "crossover_rate": 0.4
            },
            "monitoring": {
                "performance_threshold": 0.55,
                "stagnation_limit": 5
            }
        }
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key in default_config:
                        if key in user_config:
                            default_config[key].update(user_config[key])
            return default_config
        except Exception as e:
            logger.warning(f"Config load failed, using defaults: {str(e)}")
            return default_config
    
    def _init_firebase(self):
        """Initialize Firebase connection with proper error handling"""
        try:
            cred_path = self.config['firebase']['credential_path']
            
            if os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': self.config['firebase']['project_id']
                })
                logger.info("Firebase initialized successfully")
            else:
                logger.warning(f"Firebase credentials not found at {cred_path}")
                logger.info("Using mock Firestore for development")
                # In production, this should raise an error
                from unittest.mock import Mock
                self.db = Mock()
        except Exception as e:
            logger.error(f"Firebase initialization failed: {str(e)}")
            # Fallback to mock for development
            from unittest.mock import Mock
            self.db = Mock()
    
    def _initialize_components(self):
        """Initialize all system components with dependency injection"""
        logger.info("Initializing system components")
        
        # Create component instances
        self.data_pipeline = DataPipeline()
        self.architecture_generator = ArchitectureGenerator(self.config)
        self.training_engine = TrainingEngine(self.config)
        self.deployment_manager = DeploymentManager(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.evolution_engine = EvolutionEngine(self.config)