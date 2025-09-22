"""
Massive Data Training API for Mental Health Crisis Detection
Handles 1000GB+ datasets with streaming, batch processing, and cloud integration
"""

import asyncio
import aiofiles
import aiohttp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer, AutoModel, pipeline
import requests
import json
import os
import gzip
import pickle
from datetime import datetime
import logging
from typing import Iterator, List, Dict, Optional, Tuple
import sqlite3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass
import hashlib
import time
from pathlib import Path
import boto3
from google.cloud import storage as gcs
import azure.storage.blob as azblob
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('massive_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for massive data training"""
    batch_size: int = 1024
    learning_rate: float = 1e-4
    num_epochs: int = 3
    max_length: int = 512
    model_name: str = "bert-base-uncased"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = mp.cpu_count()
    cache_size: int = 10000  # Number of samples to cache
    checkpoint_interval: int = 10000  # Save every N batches
    data_streaming: bool = True
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 4

class MassiveDatasetStreamer(IterableDataset):
    """Streaming dataset for massive data files"""
    
    def __init__(self, data_sources: List[str], config: TrainingConfig):
        self.data_sources = data_sources
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.current_source = 0
        self.processed_count = 0
        
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from multiple sources"""
        for source in self.data_sources:
            logger.info(f"Processing data source: {source}")
            
            if source.startswith('http'):
                yield from self._stream_from_url(source)
            elif source.endswith('.gz'):
                yield from self._stream_from_gzip(source)
            elif source.endswith('.parquet'):
                yield from self._stream_from_parquet(source)
            elif source.endswith('.jsonl'):
                yield from self._stream_from_jsonl(source)
            else:
                yield from self._stream_from_csv(source)
    
    def _stream_from_url(self, url: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from HTTP/HTTPS URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            buffer = ""
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                buffer += chunk
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if line.strip():
                        try:
                            data = json.loads(line)
                            yield self._process_sample(data)
                            self.processed_count += 1
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Error streaming from URL {url}: {e}")
    
    def _stream_from_gzip(self, filepath: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from compressed files"""
        try:
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        yield self._process_sample(data)
                        self.processed_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error streaming from gzip {filepath}: {e}")
    
    def _stream_from_parquet(self, filepath: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from Parquet files"""
        try:
            # Read in chunks to handle large files
            for chunk in pd.read_parquet(filepath, chunksize=self.config.batch_size):
                for _, row in chunk.iterrows():
                    yield self._process_sample(row.to_dict())
                    self.processed_count += 1
        except Exception as e:
            logger.error(f"Error streaming from parquet {filepath}: {e}")
    
    def _stream_from_jsonl(self, filepath: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from JSONL files"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        yield self._process_sample(data)
                        self.processed_count += 1
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error streaming from JSONL {filepath}: {e}")
    
    def _stream_from_csv(self, filepath: str) -> Iterator[Dict[str, torch.Tensor]]:
        """Stream data from CSV files"""
        try:
            # Read in chunks to handle large files
            for chunk in pd.read_csv(filepath, chunksize=self.config.batch_size):
                for _, row in chunk.iterrows():
                    yield self._process_sample(row.to_dict())
                    self.processed_count += 1
        except Exception as e:
            logger.error(f"Error streaming from CSV {filepath}: {e}")
    
    def _process_sample(self, data: Dict) -> Dict[str, torch.Tensor]:
        """Process individual sample"""
        # Extract text and label
        text = data.get('text', data.get('content', ''))
        label = int(data.get('label', data.get('crisis', 0)))
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class AdvancedCrisisModel(nn.Module):
    """Advanced transformer model for crisis detection"""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # Binary classification
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class CloudDataManager:
    """Manage data from cloud storage providers"""
    
    def __init__(self):
        self.aws_client = None
        self.gcp_client = None
        self.azure_client = None
        self._init_cloud_clients()
    
    def _init_cloud_clients(self):
        """Initialize cloud storage clients"""
        try:
            # AWS S3
            if os.getenv('AWS_ACCESS_KEY_ID'):
                self.aws_client = boto3.client('s3')
                logger.info("AWS S3 client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize AWS client: {e}")
        
        try:
            # Google Cloud Storage
            if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                self.gcp_client = gcs.Client()
                logger.info("Google Cloud Storage client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize GCP client: {e}")
        
        try:
            # Azure Blob Storage
            if os.getenv('AZURE_STORAGE_CONNECTION_STRING'):
                self.azure_client = azblob.BlobServiceClient.from_connection_string(
                    os.getenv('AZURE_STORAGE_CONNECTION_STRING')
                )
                logger.info("Azure Blob Storage client initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Azure client: {e}")
    
    def download_from_s3(self, bucket: str, key: str, local_path: str):
        """Download file from AWS S3"""
        if self.aws_client:
            self.aws_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {key} from S3 bucket {bucket}")
    
    def download_from_gcs(self, bucket: str, blob_name: str, local_path: str):
        """Download file from Google Cloud Storage"""
        if self.gcp_client:
            bucket = self.gcp_client.bucket(bucket)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            logger.info(f"Downloaded {blob_name} from GCS bucket {bucket}")
    
    def download_from_azure(self, container: str, blob_name: str, local_path: str):
        """Download file from Azure Blob Storage"""
        if self.azure_client:
            blob_client = self.azure_client.get_blob_client(
                container=container, blob=blob_name
            )
            with open(local_path, 'wb') as f:
                data = blob_client.download_blob()
                f.write(data.readall())
            logger.info(f"Downloaded {blob_name} from Azure container {container}")

class MassiveTrainer:
    """Main trainer for massive datasets"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = AdvancedCrisisModel(config).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.cloud_manager = CloudDataManager()
        self.scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
        
        # Training metrics
        self.global_step = 0
        self.total_loss = 0.0
        self.best_accuracy = 0.0
        
        # Database for tracking
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for tracking"""
        self.db_path = "training_progress.db"
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                epoch INTEGER,
                batch INTEGER,
                loss REAL,
                accuracy REAL,
                samples_processed INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_path TEXT,
                source_type TEXT,
                size_gb REAL,
                samples_count INTEGER,
                processed BOOLEAN DEFAULT FALSE
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Training database initialized")
    
    def log_progress(self, epoch: int, batch: int, loss: float, accuracy: float, samples: int):
        """Log training progress to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO training_log 
            (timestamp, epoch, batch, loss, accuracy, samples_processed)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (datetime.now(), epoch, batch, loss, accuracy, samples))
        
        conn.commit()
        conn.close()
    
    def add_data_source(self, source_path: str, source_type: str, size_gb: float):
        """Add data source to tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO data_sources (source_path, source_type, size_gb)
            VALUES (?, ?, ?)
        """, (source_path, source_type, size_gb))
        
        conn.commit()
        conn.close()
        logger.info(f"Added data source: {source_path} ({size_gb} GB)")
    
    def train_on_massive_data(self, data_sources: List[str]):
        """Train on massive datasets with streaming"""
        logger.info(f"Starting massive data training with {len(data_sources)} sources")
        logger.info(f"Device: {self.device}, Mixed precision: {self.config.use_mixed_precision}")
        
        # Create streaming dataset
        dataset = MassiveDatasetStreamer(data_sources, self.config)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")
            
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, batch in enumerate(dataloader):
                try:
                    # Move batch to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass with mixed precision
                    if self.config.use_mixed_precision and self.scaler:
                        with torch.cuda.amp.autocast():
                            logits = self.model(input_ids, attention_mask)
                            loss = self.criterion(logits, labels)
                            loss = loss / self.config.gradient_accumulation_steps
                        
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                    else:
                        logits = self.model(input_ids, attention_mask)
                        loss = self.criterion(logits, labels)
                        loss = loss / self.config.gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                    self.global_step += 1
                    
                    # Calculate accuracy
                    with torch.no_grad():
                        predictions = torch.argmax(logits, dim=1)
                        accuracy = (predictions == labels).float().mean().item()
                    
                    # Log progress
                    if batch_idx % 100 == 0:
                        avg_loss = epoch_loss / batch_count
                        logger.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}")
                        self.log_progress(epoch, batch_idx, avg_loss, accuracy, dataset.processed_count)
                    
                    # Save checkpoint
                    if batch_idx % self.config.checkpoint_interval == 0:
                        self.save_checkpoint(epoch, batch_idx)
                    
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
            
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            logger.info(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch, batch_count)
        
        logger.info("Massive data training completed!")
        self.save_final_model()
    
    def save_checkpoint(self, epoch: int, batch: int):
        """Save training checkpoint"""
        checkpoint_path = f"checkpoint_epoch_{epoch}_batch_{batch}.pt"
        torch.save({
            'epoch': epoch,
            'batch': batch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_model(self):
        """Save final trained model"""
        final_path = f"massive_crisis_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'tokenizer': self.model.config.model_name
        }, final_path)
        logger.info(f"Final model saved: {final_path}")

class MassiveDataAPI:
    """API for massive data training"""
    
    def __init__(self):
        self.trainer = None
        self.config = TrainingConfig()
        self.training_active = False
    
    def start_training(self, data_sources: List[str], config: Optional[Dict] = None):
        """Start massive data training"""
        if config:
            # Update configuration
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        self.trainer = MassiveTrainer(self.config)
        self.training_active = True
        
        # Add data sources to tracking
        for source in data_sources:
            # Estimate size (simplified)
            if os.path.exists(source):
                size_gb = os.path.getsize(source) / (1024**3)
            else:
                size_gb = 0.0  # Unknown for URLs
            
            self.trainer.add_data_source(source, "auto", size_gb)
        
        # Start training in background
        import threading
        training_thread = threading.Thread(
            target=self.trainer.train_on_massive_data,
            args=(data_sources,)
        )
        training_thread.start()
        
        return {
            "status": "training_started",
            "data_sources": len(data_sources),
            "config": self.config.__dict__
        }
    
    def get_training_status(self):
        """Get current training status"""
        if not self.trainer:
            return {"status": "not_started"}
        
        # Query database for latest progress
        conn = sqlite3.connect(self.trainer.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM training_log 
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        
        latest = cursor.fetchone()
        conn.close()
        
        if latest:
            return {
                "status": "training" if self.training_active else "completed",
                "epoch": latest[2],
                "batch": latest[3],
                "loss": latest[4],
                "accuracy": latest[5],
                "samples_processed": latest[6],
                "last_update": latest[1]
            }
        
        return {"status": "starting"}
    
    def add_external_data_source(self, source_config: Dict):
        """Add external data source (API, cloud, etc.)"""
        source_type = source_config.get("type", "unknown")
        
        if source_type == "huggingface":
            return self._add_huggingface_dataset(source_config)
        elif source_type == "kaggle":
            return self._add_kaggle_dataset(source_config)
        elif source_type == "s3":
            return self._add_s3_dataset(source_config)
        elif source_type == "gcs":
            return self._add_gcs_dataset(source_config)
        elif source_type == "azure":
            return self._add_azure_dataset(source_config)
        else:
            return {"error": f"Unsupported source type: {source_type}"}
    
    def _add_huggingface_dataset(self, config: Dict):
        """Add Hugging Face dataset"""
        try:
            from datasets import load_dataset
            dataset_name = config.get("dataset_name")
            subset = config.get("subset", None)
            
            dataset = load_dataset(dataset_name, subset)
            
            # Convert to our format and save locally
            local_path = f"hf_{dataset_name.replace('/', '_')}.jsonl"
            
            with open(local_path, 'w') as f:
                for split in dataset:
                    for item in dataset[split]:
                        f.write(json.dumps(item) + '\n')
            
            return {"status": "added", "local_path": local_path}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _add_kaggle_dataset(self, config: Dict):
        """Add Kaggle dataset"""
        try:
            import kaggle
            
            dataset_name = config.get("dataset_name")
            kaggle.api.dataset_download_files(dataset_name, path="./", unzip=True)
            
            return {"status": "downloaded", "path": "./"}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _add_s3_dataset(self, config: Dict):
        """Add AWS S3 dataset"""
        try:
            bucket = config.get("bucket")
            key = config.get("key")
            local_path = config.get("local_path", f"s3_{key.split('/')[-1]}")
            
            if self.trainer:
                self.trainer.cloud_manager.download_from_s3(bucket, key, local_path)
            
            return {"status": "downloaded", "local_path": local_path}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _add_gcs_dataset(self, config: Dict):
        """Add Google Cloud Storage dataset"""
        try:
            bucket = config.get("bucket")
            blob_name = config.get("blob_name")
            local_path = config.get("local_path", f"gcs_{blob_name.split('/')[-1]}")
            
            if self.trainer:
                self.trainer.cloud_manager.download_from_gcs(bucket, blob_name, local_path)
            
            return {"status": "downloaded", "local_path": local_path}
        
        except Exception as e:
            return {"error": str(e)}
    
    def _add_azure_dataset(self, config: Dict):
        """Add Azure Blob Storage dataset"""
        try:
            container = config.get("container")
            blob_name = config.get("blob_name")
            local_path = config.get("local_path", f"azure_{blob_name.split('/')[-1]}")
            
            if self.trainer:
                self.trainer.cloud_manager.download_from_azure(container, blob_name, local_path)
            
            return {"status": "downloaded", "local_path": local_path}
        
        except Exception as e:
            return {"error": str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Initialize API
    api = MassiveDataAPI()
    
    # Example data sources (replace with your actual data)
    data_sources = [
        "https://example.com/crisis_data_1.jsonl",
        "https://example.com/crisis_data_2.jsonl.gz",
        "/path/to/local/crisis_data.parquet",
        "/path/to/local/crisis_data.csv"
    ]
    
    # Custom configuration for massive training
    training_config = {
        "batch_size": 2048,
        "learning_rate": 5e-5,
        "num_epochs": 5,
        "use_mixed_precision": True,
        "gradient_accumulation_steps": 8
    }
    
    print("🚀 Starting massive data training...")
    result = api.start_training(data_sources, training_config)
    print(f"Training started: {result}")
    
    # Monitor training progress
    import time
    while True:
        status = api.get_training_status()
        print(f"Training status: {status}")
        
        if status.get("status") in ["completed", "error"]:
            break
        
        time.sleep(30)  # Check every 30 seconds
    
    print("✅ Massive data training completed!")
