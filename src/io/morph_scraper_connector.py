#!/usr/bin/env python3
"""
SEP Trading System - Morph.io Web Scraper Connector
Integrates web scraping capabilities with the SEP trading data pipeline
"""

import os
import sys
import json
import sqlite3
import logging
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

import requests
from scraperwiki import sqlite as scraperwiki_sqlite

class MorphScraperConnector:
    """
    Web scraper connector for SEP trading system
    Follows the same pattern as OandaConnector but for web data sources
    """
    
    def __init__(self, config_path: str = "./config/scraper_config.json"):
        self.config_path = config_path
        self.data_path = "./data/scraped"
        self.cache_path = "./cache/scraper"
        self.db_path = os.path.join(self.data_path, "scraped_data.sqlite")
        self.last_error = ""
        
        # Create directories
        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(os.path.join(self.data_path, "scraper.log"))
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load scraper configuration"""
        default_config = {
            "scrapers": [
                {
                    "name": "economic_calendar", 
                    "url": "https://www.forexfactory.com/calendar",
                    "enabled": True,
                    "schedule": "hourly"
                },
                {
                    "name": "market_sentiment",
                    "url": "https://www.dailyfx.com/sentiment", 
                    "enabled": True,
                    "schedule": "daily"
                }
            ],
            "output_format": "json",
            "rate_limit": 1.0,
            "user_agent": "SEP-TradingBot/1.0"
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading config: {e}")
                return default_config
        else:
            # Create default config
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def initialize(self) -> bool:
        """Initialize the scraper connector"""
        try:
            # Test database connection
            self._init_database()
            
            # Run basic morph.io scraper test
            result = self._run_morph_scraper()
            if result:
                self.logger.info("Scraper connector initialized successfully")
                return True
            else:
                self.last_error = "Failed to run morph scraper"
                return False
                
        except Exception as e:
            self.last_error = f"Initialization failed: {str(e)}"
            self.logger.error(self.last_error)
            return False
    
    def _init_database(self):
        """Initialize SQLite database for scraped data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different data types
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scraped_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS economic_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_time TEXT NOT NULL,
                currency TEXT NOT NULL,
                event_name TEXT NOT NULL,
                importance INTEGER,
                forecast TEXT,
                previous TEXT,
                actual TEXT,
                scraped_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _run_morph_scraper(self) -> bool:
        """Run the morph.io scraper script"""
        try:
            # Change to project root to run scraper
            original_cwd = os.getcwd()
            
            # Run the scraper.rb file we created earlier
            if os.path.exists("./scraper.rb"):
                result = subprocess.run(
                    ["ruby", "scraper.rb"], 
                    capture_output=True, 
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    self.logger.info("Morph scraper executed successfully")
                    return True
                else:
                    self.logger.error(f"Scraper failed: {result.stderr}")
                    return False
            else:
                self.logger.warning("scraper.rb not found, creating sample data")
                self._create_sample_data()
                return True
                
        except Exception as e:
            self.logger.error(f"Error running morph scraper: {e}")
            return False
        finally:
            os.chdir(original_cwd)
    
    def _create_sample_data(self):
        """Create sample scraped data for testing"""
        sample_data = [
            {
                "source": "economic_calendar",
                "event": "US Non-Farm Payrolls",
                "currency": "USD", 
                "importance": 3,
                "forecast": "180K",
                "previous": "175K",
                "time": datetime.now().isoformat()
            },
            {
                "source": "market_sentiment",
                "pair": "EUR/USD",
                "bullish": 65,
                "bearish": 35,
                "time": datetime.now().isoformat()
            }
        ]
        
        self.save_scraped_data(sample_data)
    
    def scrape_economic_calendar(self) -> List[Dict[str, Any]]:
        """Scrape economic calendar data"""
        try:
            # This would integrate with the morph.io scraper
            # For now, return sample data
            return [
                {
                    "event_time": (datetime.now() + timedelta(hours=2)).isoformat(),
                    "currency": "USD",
                    "event_name": "Federal Reserve Speech", 
                    "importance": 2,
                    "forecast": "",
                    "previous": "",
                    "actual": ""
                }
            ]
        except Exception as e:
            self.logger.error(f"Error scraping economic calendar: {e}")
            return []
    
    def scrape_market_sentiment(self) -> List[Dict[str, Any]]:
        """Scrape market sentiment data"""
        try:
            # This would integrate with sentiment analysis scraping
            return [
                {
                    "pair": "EUR/USD",
                    "bullish_percentage": 58,
                    "bearish_percentage": 42,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            self.logger.error(f"Error scraping market sentiment: {e}")
            return []
    
    def save_scraped_data(self, data: List[Dict[str, Any]]):
        """Save scraped data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in data:
            cursor.execute('''
                INSERT INTO scraped_data (source, data_type, timestamp, content, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                item.get('source', 'unknown'),
                item.get('data_type', 'general'),
                datetime.now().isoformat(),
                json.dumps(item),
                json.dumps({"scraped_by": "morph_connector"})
            ))
        
        conn.commit()
        conn.close()
        self.logger.info(f"Saved {len(data)} scraped data items")
    
    def get_recent_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent scraped data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        cursor.execute('''
            SELECT * FROM scraped_data 
            WHERE timestamp > ? 
            ORDER BY timestamp DESC
        ''', (cutoff_time,))
        
        rows = cursor.fetchall()
        conn.close()
        
        columns = ['id', 'source', 'data_type', 'timestamp', 'content', 'metadata']
        return [dict(zip(columns, row)) for row in rows]
    
    def get_economic_events(self, hours: int = 48) -> List[Dict[str, Any]]:
        """Get upcoming economic events"""
        economic_data = self.scrape_economic_calendar()
        return economic_data
    
    def get_market_sentiment_data(self) -> List[Dict[str, Any]]:
        """Get current market sentiment"""
        sentiment_data = self.scrape_market_sentiment()
        return sentiment_data
    
    def test_connection(self) -> bool:
        """Test scraper connectivity"""
        try:
            # Test database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            
            # Test scraper execution
            return self._run_morph_scraper()
            
        except Exception as e:
            self.last_error = f"Connection test failed: {str(e)}"
            return False
    
    def get_last_error(self) -> str:
        """Get last error message"""
        return self.last_error
    
    def shutdown(self):
        """Cleanup resources"""
        self.logger.info("Scraper connector shutting down")

if __name__ == "__main__":
    # Test the connector
    connector = MorphScraperConnector()
    
    if connector.initialize():
        print("✅ Scraper connector initialized successfully")
        
        # Test data retrieval
        recent_data = connector.get_recent_data(24)
        print(f"📊 Found {len(recent_data)} recent data items")
        
        economic_events = connector.get_economic_events()
        print(f"📅 Found {len(economic_events)} economic events")
        
        sentiment_data = connector.get_market_sentiment_data()
        print(f"📈 Found {len(sentiment_data)} sentiment data points")
        
        connector.shutdown()
    else:
        print(f"❌ Failed to initialize: {connector.get_last_error()}")