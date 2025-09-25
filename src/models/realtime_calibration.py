"""
Real-time calibration system for CRQC predictions.

This module provides continuous monitoring and updating of quantum computing
progress through multiple data sources including academic papers, industry
announcements, and live metrics.
"""

import asyncio
import aiohttp
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import feedparser
import numpy as np
from collections import deque
import re

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logging.warning("requests not available - API features limited")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup not available - web scraping limited")

logger = logging.getLogger(__name__)


@dataclass
class DataPoint:
    """Represents a single calibration data point."""
    
    source: str
    timestamp: datetime
    data_type: str  # 'paper', 'announcement', 'metric', 'patent'
    title: str
    content: Dict[str, Any]
    relevance_score: float = 0.0
    impact_on_timeline: float = 0.0  # Years acceleration/delay
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'data_type': self.data_type,
            'title': self.title,
            'content': self.content,
            'relevance_score': self.relevance_score,
            'impact_on_timeline': self.impact_on_timeline,
            'confidence': self.confidence
        }


@dataclass 
class CalibrationUpdate:
    """Results of a calibration update."""
    
    timestamp: datetime
    data_points_processed: int
    parameters_updated: Dict[str, Tuple[float, float]]  # param: (old, new)
    timeline_adjustment: float  # Years
    confidence_change: float
    anomalies_detected: List[str]
    recommendations: List[str]


class RealTimeCalibrator:
    """
    Main calibration system that continuously monitors and updates CRQC predictions.
    """
    
    def __init__(self, 
                 update_interval_hours: float = 24.0,
                 cache_dir: Path = Path("data/cache/realtime")):
        """
        Initialize the real-time calibration system.
        
        Args:
            update_interval_hours: How often to check for updates
            cache_dir: Directory for caching data
        """
        self.update_interval = timedelta(hours=update_interval_hours)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Data sources
        self.sources = self._initialize_sources()
        
        # Data storage
        self.recent_data = deque(maxlen=1000)  # Keep last 1000 data points
        self.anomaly_buffer = deque(maxlen=100)  # Recent anomalies
        
        # Statistical tracking
        self.baseline_metrics = self._load_baseline_metrics()
        self.running_stats = {
            'papers_per_week': [],
            'qubit_growth_rate': [],
            'investment_flow': [],
            'patent_filings': []
        }
        
        # Last update tracking
        self.last_update = datetime.now() - self.update_interval
        self.update_history = []
    
    def _initialize_sources(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data source configurations."""
        
        return {
            # Academic sources
            "arxiv": {
                "url": "http://export.arxiv.org/api/query",
                "type": "api",
                "categories": ["quant-ph", "cs.CR"],
                "keywords": ["quantum computing", "quantum algorithm", "error correction", 
                           "logical qubit", "shor algorithm", "grover algorithm",
                           "quantum supremacy", "quantum advantage"],
                "update_frequency": "daily"
            },
            
            # Industry sources
            "ibm_quantum": {
                "url": "https://quantum-computing.ibm.com/api/",
                "type": "api",
                "metrics": ["qubit_count", "quantum_volume", "error_rate"],
                "update_frequency": "weekly"
            },
            
            "google_quantum": {
                "url": "https://quantumai.google/cirq/",  # Would need actual API
                "type": "scrape",
                "selectors": {
                    "announcements": ".announcement",
                    "metrics": ".performance-metric"
                },
                "update_frequency": "weekly"
            },
            
            # News and announcements
            "quantum_computing_report": {
                "url": "https://quantumcomputingreport.com/feed/",
                "type": "rss",
                "categories": ["news", "analysis"],
                "update_frequency": "daily"
            },
            
            # Patent databases
            "uspto_quantum": {
                "url": "https://developer.uspto.gov/ibd-api/v1/patent/",
                "type": "api",
                "query": "quantum computing algorithm",
                "fields": ["title", "abstract", "claims"],
                "update_frequency": "weekly"
            },
            
            # Financial data
            "quantum_investments": {
                "url": "https://pitchbook.com/api/",  # Would need subscription
                "type": "api",
                "sectors": ["quantum computing", "quantum software"],
                "update_frequency": "monthly"
            },
            
            # GitHub activity
            "github_quantum": {
                "url": "https://api.github.com/search/repositories",
                "type": "api",
                "query": "quantum algorithm simulator",
                "sort": "updated",
                "metrics": ["stars", "forks", "commits"],
                "update_frequency": "weekly"
            }
        }
    
    def _load_baseline_metrics(self) -> Dict[str, float]:
        """Load baseline metrics for anomaly detection."""
        
        baseline_file = self.cache_dir / "baseline_metrics.json"
        
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        
        # Default baseline (as of early 2025)
        return {
            "papers_per_week_mean": 15.0,
            "papers_per_week_std": 5.0,
            "qubit_growth_monthly": 1.05,  # 5% monthly growth
            "patent_filings_monthly": 25.0,
            "investment_quarterly_millions": 500.0,
            "error_rate_improvement_monthly": 0.95,  # 5% improvement
            "github_stars_growth_weekly": 1.02
        }
    
    async def fetch_arxiv_papers(self, session: aiohttp.ClientSession) -> List[DataPoint]:
        """Fetch recent quantum computing papers from arXiv."""
        
        data_points = []
        
        # Build query
        categories = self.sources["arxiv"]["categories"]
        keywords = self.sources["arxiv"]["keywords"]
        
        query_parts = []
        for cat in categories:
            query_parts.append(f"cat:{cat}")
        
        # Add keyword search
        keyword_query = " OR ".join([f'all:"{kw}"' for kw in keywords[:3]])  # Limit keywords
        
        query = f"({' OR '.join(query_parts)}) AND ({keyword_query})"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            async with session.get(self.sources["arxiv"]["url"], params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries:
                        # Extract relevant information
                        title = entry.title
                        abstract = entry.summary
                        published = datetime.strptime(entry.published, "%Y-%m-%dT%H:%M:%SZ")
                        
                        # Skip if too old (> 7 days)
                        if datetime.now() - published > timedelta(days=7):
                            continue
                        
                        # Calculate relevance
                        relevance = self._calculate_paper_relevance(title, abstract)
                        
                        if relevance > 0.3:  # Threshold for inclusion
                            # Extract key findings
                            findings = self._extract_key_findings(title, abstract)
                            
                            data_point = DataPoint(
                                source="arxiv",
                                timestamp=published,
                                data_type="paper",
                                title=title,
                                content={
                                    "abstract": abstract[:500],  # Truncate
                                    "authors": [a.name for a in entry.authors],
                                    "arxiv_id": entry.id.split('/')[-1],
                                    "findings": findings
                                },
                                relevance_score=relevance,
                                impact_on_timeline=findings.get("timeline_impact", 0),
                                confidence=0.7
                            )
                            
                            data_points.append(data_point)
                            
        except Exception as e:
            logger.error(f"Error fetching arXiv papers: {e}")
        
        return data_points
    
    def _calculate_paper_relevance(self, title: str, abstract: str) -> float:
        """Calculate relevance score for a paper."""
        
        relevance = 0.0
        text = (title + " " + abstract).lower()
        
        # High-impact keywords
        high_impact = ["breakthrough", "crqc", "cryptanalysis", "shor", "grover",
                      "logical qubit", "error correction", "quantum supremacy",
                      "factoring", "discrete logarithm", "rsa", "ed25519"]
        
        medium_impact = ["quantum algorithm", "quantum circuit", "fault tolerant",
                        "quantum volume", "quantum advantage", "nisq"]
        
        # Count keyword matches
        for keyword in high_impact:
            if keyword in text:
                relevance += 0.2
        
        for keyword in medium_impact:
            if keyword in text:
                relevance += 0.1
        
        # Check for specific numbers (qubit counts)
        qubit_matches = re.findall(r'(\d+)\s*(?:logical\s*)?qubit', text)
        if qubit_matches:
            max_qubits = max(int(q) for q in qubit_matches)
            if max_qubits > 1000:
                relevance += 0.3
            elif max_qubits > 100:
                relevance += 0.1
        
        return min(1.0, relevance)
    
    def _extract_key_findings(self, title: str, abstract: str) -> Dict[str, Any]:
        """Extract key findings from paper text."""
        
        findings = {
            "qubit_improvements": None,
            "algorithm_speedup": None,
            "error_reduction": None,
            "timeline_impact": 0.0
        }
        
        text = (title + " " + abstract).lower()
        
        # Look for qubit improvements
        qubit_improvement = re.search(r'reduc\w+.*?(\d+)%.*?qubit', text)
        if qubit_improvement:
            findings["qubit_improvements"] = float(qubit_improvement.group(1))
            findings["timeline_impact"] -= 0.1  # Accelerates timeline
        
        # Look for algorithm improvements
        if "improved" in text and "algorithm" in text:
            speedup = re.search(r'(\d+)x?\s*(?:faster|speedup|improvement)', text)
            if speedup:
                findings["algorithm_speedup"] = float(speedup.group(1))
                findings["timeline_impact"] -= 0.2
        
        # Look for error rate improvements
        error_improvement = re.search(r'error.*?reduc\w+.*?(\d+)', text)
        if error_improvement:
            findings["error_reduction"] = float(error_improvement.group(1))
            findings["timeline_impact"] -= 0.05
        
        return findings
    
    async def fetch_quantum_news(self, session: aiohttp.ClientSession) -> List[DataPoint]:
        """Fetch quantum computing news and announcements."""
        
        data_points = []
        
        # Quantum Computing Report RSS
        try:
            async with session.get(self.sources["quantum_computing_report"]["url"]) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    for entry in feed.entries[:20]:  # Last 20 items
                        published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
                        
                        # Skip old news
                        if datetime.now() - published > timedelta(days=14):
                            continue
                        
                        # Analyze news impact
                        impact = self._analyze_news_impact(entry.title, entry.summary)
                        
                        if impact["relevance"] > 0.2:
                            data_point = DataPoint(
                                source="quantum_computing_report",
                                timestamp=published,
                                data_type="announcement",
                                title=entry.title,
                                content={
                                    "summary": entry.summary[:300],
                                    "link": entry.link,
                                    "category": impact.get("category", "general"),
                                    "company": impact.get("company"),
                                    "metrics": impact.get("metrics", {})
                                },
                                relevance_score=impact["relevance"],
                                impact_on_timeline=impact.get("timeline_impact", 0),
                                confidence=0.6
                            )
                            
                            data_points.append(data_point)
                            
        except Exception as e:
            logger.error(f"Error fetching quantum news: {e}")
        
        return data_points
    
    def _analyze_news_impact(self, title: str, content: str) -> Dict[str, Any]:
        """Analyze the impact of a news item."""
        
        impact = {
            "relevance": 0.0,
            "timeline_impact": 0.0,
            "category": "general"
        }
        
        text = (title + " " + content).lower()
        
        # Company announcements
        companies = ["ibm", "google", "microsoft", "amazon", "intel", "ionq", 
                    "rigetti", "psiquantum", "atom computing", "quantinuum"]
        
        for company in companies:
            if company in text:
                impact["company"] = company
                impact["relevance"] += 0.2
                break
        
        # Check for specific metrics
        metrics = {}
        
        # Qubit announcements
        qubit_announce = re.search(r'(\d+)\s*(?:qubit|qubits)', text)
        if qubit_announce:
            qubit_count = int(qubit_announce.group(1))
            metrics["qubits"] = qubit_count
            if qubit_count > 1000:
                impact["relevance"] += 0.5
                impact["timeline_impact"] -= 0.3
            elif qubit_count > 500:
                impact["relevance"] += 0.3
                impact["timeline_impact"] -= 0.1
        
        # Funding announcements
        funding = re.search(r'\$(\d+(?:\.\d+)?)\s*(million|billion)', text)
        if funding:
            amount = float(funding.group(1))
            if funding.group(2) == "billion":
                amount *= 1000
            metrics["funding_millions"] = amount
            impact["relevance"] += 0.3
            if amount > 100:
                impact["timeline_impact"] -= 0.1
        
        # Breakthrough announcements
        if any(word in text for word in ["breakthrough", "milestone", "achieved", "first"]):
            impact["relevance"] += 0.3
            impact["category"] = "breakthrough"
            impact["timeline_impact"] -= 0.2
        
        impact["metrics"] = metrics
        return impact
    
    async def fetch_github_activity(self, session: aiohttp.ClientSession) -> List[DataPoint]:
        """Monitor GitHub quantum computing activity."""
        
        data_points = []
        
        if not REQUESTS_AVAILABLE:
            return data_points
        
        headers = {
            "Accept": "application/vnd.github.v3+json",
            # Add token if available: "Authorization": "token YOUR_GITHUB_TOKEN"
        }
        
        params = {
            "q": self.sources["github_quantum"]["query"],
            "sort": self.sources["github_quantum"]["sort"],
            "order": "desc",
            "per_page": 20
        }
        
        try:
            async with session.get(
                self.sources["github_quantum"]["url"], 
                headers=headers, 
                params=params
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for repo in data.get("items", []):
                        # Check if updated recently
                        updated = datetime.strptime(repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
                        if datetime.now() - updated > timedelta(days=30):
                            continue
                        
                        # Analyze repository
                        relevance = self._analyze_github_repo(repo)
                        
                        if relevance > 0.2:
                            data_point = DataPoint(
                                source="github",
                                timestamp=updated,
                                data_type="metric",
                                title=f"GitHub: {repo['full_name']}",
                                content={
                                    "stars": repo["stargazers_count"],
                                    "forks": repo["forks_count"],
                                    "description": repo.get("description", "")[:200],
                                    "language": repo.get("language"),
                                    "topics": repo.get("topics", [])
                                },
                                relevance_score=relevance,
                                impact_on_timeline=0.0,  # GitHub activity is indirect
                                confidence=0.4
                            )
                            
                            data_points.append(data_point)
                            
        except Exception as e:
            logger.error(f"Error fetching GitHub data: {e}")
        
        return data_points
    
    def _analyze_github_repo(self, repo: Dict[str, Any]) -> float:
        """Analyze relevance of a GitHub repository."""
        
        relevance = 0.0
        
        # High star count indicates importance
        stars = repo.get("stargazers_count", 0)
        if stars > 1000:
            relevance += 0.3
        elif stars > 100:
            relevance += 0.1
        
        # Check description and topics
        description = (repo.get("description", "") + " ".join(repo.get("topics", []))).lower()
        
        relevant_terms = ["quantum circuit", "quantum algorithm", "quantum simulator",
                         "shor", "grover", "qiskit", "cirq", "quantum compiler"]
        
        for term in relevant_terms:
            if term in description:
                relevance += 0.2
                break
        
        # Recent activity
        updated = datetime.strptime(repo["updated_at"], "%Y-%m-%dT%H:%M:%SZ")
        if datetime.now() - updated < timedelta(days=7):
            relevance += 0.1
        
        return min(1.0, relevance)
    
    async def collect_all_data(self) -> List[DataPoint]:
        """Collect data from all sources asynchronously."""
        
        all_data_points = []
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for all data sources
            tasks = [
                self.fetch_arxiv_papers(session),
                self.fetch_quantum_news(session),
                self.fetch_github_activity(session)
            ]
            
            # Run all tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Error in data collection: {result}")
                else:
                    all_data_points.extend(result)
        
        return all_data_points
    
    def detect_anomalies(self, data_points: List[DataPoint]) -> List[str]:
        """Detect anomalous patterns in collected data."""
        
        anomalies = []
        
        # Calculate current metrics
        current_metrics = self._calculate_current_metrics(data_points)
        
        # Paper velocity anomaly
        papers_this_week = current_metrics.get("papers_per_week", 0)
        expected = self.baseline_metrics["papers_per_week_mean"]
        std = self.baseline_metrics["papers_per_week_std"]
        
        if abs(papers_this_week - expected) > 2 * std:
            if papers_this_week > expected:
                anomalies.append(f"Unusual surge in quantum papers: {papers_this_week} vs {expected:.0f} expected")
            else:
                anomalies.append(f"Unusual drop in quantum papers: {papers_this_week} vs {expected:.0f} expected")
        
        # Major announcements
        major_announcements = [dp for dp in data_points 
                             if dp.data_type == "announcement" and dp.relevance_score > 0.7]
        
        if len(major_announcements) > 2:
            anomalies.append(f"Multiple major announcements detected ({len(major_announcements)})")
        
        # Qubit count jumps
        qubit_announcements = []
        for dp in data_points:
            if "qubits" in dp.content.get("metrics", {}):
                qubit_announcements.append(dp.content["metrics"]["qubits"])
        
        if qubit_announcements and max(qubit_announcements) > 2000:
            anomalies.append(f"Major qubit milestone announced: {max(qubit_announcements)} qubits")
        
        # Funding surge
        funding_total = sum(
            dp.content.get("metrics", {}).get("funding_millions", 0)
            for dp in data_points
            if dp.data_type == "announcement"
        )
        
        if funding_total > 1000:  # $1B in recent funding
            anomalies.append(f"Massive funding surge: ${funding_total:.0f}M in recent announcements")
        
        return anomalies
    
    def _calculate_current_metrics(self, data_points: List[DataPoint]) -> Dict[str, float]:
        """Calculate current metrics from data points."""
        
        metrics = {}
        
        # Papers per week
        week_ago = datetime.now() - timedelta(days=7)
        recent_papers = [dp for dp in data_points 
                        if dp.data_type == "paper" and dp.timestamp > week_ago]
        metrics["papers_per_week"] = len(recent_papers)
        
        # Average relevance
        if data_points:
            metrics["avg_relevance"] = np.mean([dp.relevance_score for dp in data_points])
        
        # Timeline impact
        timeline_impacts = [dp.impact_on_timeline for dp in data_points if dp.impact_on_timeline != 0]
        if timeline_impacts:
            metrics["total_timeline_impact"] = sum(timeline_impacts)
            metrics["avg_timeline_impact"] = np.mean(timeline_impacts)
        
        return metrics
    
    def update_parameters(self, data_points: List[DataPoint]) -> Dict[str, Tuple[float, float]]:
        """Update model parameters based on collected data."""
        
        updates = {}
        
        # Calculate aggregated impact
        total_timeline_impact = sum(dp.impact_on_timeline for dp in data_points)
        avg_confidence = np.mean([dp.confidence for dp in data_points]) if data_points else 0.5
        
        # Update CRQC timeline
        if abs(total_timeline_impact) > 0.1:
            old_median = 2033  # Current baseline
            new_median = old_median + total_timeline_impact
            updates["crqc_median_year"] = (old_median, new_median)
        
        # Update confidence based on data volume and quality
        if len(data_points) > 50 and avg_confidence > 0.6:
            updates["prediction_confidence"] = (0.85, 0.90)
        
        # Update growth rates based on announcements
        qubit_growth_observed = self._calculate_qubit_growth(data_points)
        if qubit_growth_observed:
            old_growth = 1.5
            new_growth = 0.7 * old_growth + 0.3 * qubit_growth_observed
            updates["qubit_growth_rate"] = (old_growth, new_growth)
        
        # Update breakthrough probability based on anomalies
        if self.anomaly_buffer:
            old_prob = 0.05
            new_prob = min(0.15, old_prob * (1 + len(self.anomaly_buffer) * 0.1))
            updates["breakthrough_probability"] = (old_prob, new_prob)
        
        return updates
    
    def _calculate_qubit_growth(self, data_points: List[DataPoint]) -> Optional[float]:
        """Calculate observed qubit growth rate."""
        
        qubit_data = []
        
        for dp in data_points:
            if "qubits" in dp.content.get("metrics", {}):
                qubit_data.append({
                    "date": dp.timestamp,
                    "qubits": dp.content["metrics"]["qubits"],
                    "company": dp.content.get("company", "unknown")
                })
        
        if len(qubit_data) < 2:
            return None
        
        # Sort by date
        qubit_data.sort(key=lambda x: x["date"])
        
        # Calculate growth rate (simplified - in practice would be more sophisticated)
        latest = qubit_data[-1]["qubits"]
        earliest = qubit_data[0]["qubits"]
        time_diff = (qubit_data[-1]["date"] - qubit_data[0]["date"]).days / 365.25
        
        if time_diff > 0 and earliest > 0:
            growth_rate = (latest / earliest) ** (1 / time_diff)
            return growth_rate
        
        return None
    
    async def run_calibration_cycle(self) -> CalibrationUpdate:
        """Run a complete calibration cycle."""
        
        logger.info("Starting real-time calibration cycle")
        start_time = datetime.now()
        
        # Collect data from all sources
        data_points = await self.collect_all_data()
        
        # Store recent data
        self.recent_data.extend(data_points)
        
        # Detect anomalies
        anomalies = self.detect_anomalies(data_points)
        self.anomaly_buffer.extend(anomalies)
        
        # Update parameters
        parameter_updates = self.update_parameters(data_points)
        
        # Calculate overall timeline adjustment
        timeline_adjustment = sum(dp.impact_on_timeline for dp in data_points)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            data_points, anomalies, parameter_updates
        )
        
        # Create update record
        update = CalibrationUpdate(
            timestamp=start_time,
            data_points_processed=len(data_points),
            parameters_updated=parameter_updates,
            timeline_adjustment=timeline_adjustment,
            confidence_change=parameter_updates.get("prediction_confidence", (0, 0))[1] - 
                            parameter_updates.get("prediction_confidence", (0, 0))[0],
            anomalies_detected=anomalies,
            recommendations=recommendations
        )
        
        # Store update history
        self.update_history.append(update)
        self.last_update = datetime.now()
        
        # Save state
        self._save_calibration_state()
        
        logger.info(f"Calibration complete. Processed {len(data_points)} data points.")
        logger.info(f"Timeline adjustment: {timeline_adjustment:+.2f} years")
        
        return update
    
    def _generate_recommendations(self, 
                                data_points: List[DataPoint],
                                anomalies: List[str],
                                updates: Dict[str, Tuple[float, float]]) -> List[str]:
        """Generate actionable recommendations based on calibration."""
        
        recommendations = []
        
        # Major timeline changes
        if "crqc_median_year" in updates:
            old, new = updates["crqc_median_year"]
            if new < old:
                recommendations.append(
                    f"URGENT: CRQC timeline accelerated to {new:.0f} (from {old:.0f}). "
                    "Consider accelerating migration plans."
                )
            else:
                recommendations.append(
                    f"Timeline extended to {new:.0f} (from {old:.0f}). "
                    "Additional time available for migration."
                )
        
        # Anomaly-based recommendations
        if any("surge" in a for a in anomalies):
            recommendations.append(
                "Increased quantum research activity detected. "
                "Monitor for breakthrough announcements."
            )
        
        if any("qubit milestone" in a for a in anomalies):
            recommendations.append(
                "Major qubit milestone reached. "
                "Re-evaluate threat assessment and migration urgency."
            )
        
        # High-impact papers
        high_impact_papers = [dp for dp in data_points 
                            if dp.data_type == "paper" and dp.relevance_score > 0.8]
        
        if high_impact_papers:
            recommendations.append(
                f"Review {len(high_impact_papers)} high-impact papers for "
                "potential algorithm improvements."
            )
        
        # Confidence changes
        if "prediction_confidence" in updates:
            old, new = updates["prediction_confidence"]
            if new > old:
                recommendations.append(
                    "Prediction confidence increased. Current models well-calibrated."
                )
            else:
                recommendations.append(
                    "Prediction confidence decreased. Consider expanding data sources."
                )
        
        return recommendations
    
    def _save_calibration_state(self):
        """Save current calibration state to disk."""
        
        state = {
            "last_update": self.last_update.isoformat(),
            "baseline_metrics": self.baseline_metrics,
            "recent_anomalies": list(self.anomaly_buffer),
            "update_count": len(self.update_history),
            "current_adjustments": {
                "timeline": sum(u.timeline_adjustment for u in self.update_history[-10:])
                if self.update_history else 0
            }
        }
        
        state_file = self.cache_dir / "calibration_state.json"
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_current_adjustments(self) -> Dict[str, Any]:
        """Get current calibration adjustments for models."""
        
        if not self.update_history:
            return {
                "timeline_adjustment": 0,
                "confidence_multiplier": 1.0,
                "growth_rate_adjustment": 1.0,
                "breakthrough_probability_multiplier": 1.0
            }
        
        # Aggregate recent updates (last 30 days)
        recent_updates = [u for u in self.update_history 
                         if datetime.now() - u.timestamp < timedelta(days=30)]
        
        if not recent_updates:
            recent_updates = self.update_history[-5:]  # Last 5 updates
        
        adjustments = {
            "timeline_adjustment": sum(u.timeline_adjustment for u in recent_updates),
            "confidence_multiplier": 1.0,
            "growth_rate_adjustment": 1.0,
            "breakthrough_probability_multiplier": 1.0
        }
        
        # Extract multipliers from parameter updates
        for update in recent_updates:
            if "prediction_confidence" in update.parameters_updated:
                old, new = update.parameters_updated["prediction_confidence"]
                adjustments["confidence_multiplier"] *= (new / old) if old > 0 else 1.0
            
            if "qubit_growth_rate" in update.parameters_updated:
                old, new = update.parameters_updated["qubit_growth_rate"]
                adjustments["growth_rate_adjustment"] *= (new / old) if old > 0 else 1.0
            
            if "breakthrough_probability" in update.parameters_updated:
                old, new = update.parameters_updated["breakthrough_probability"]
                adjustments["breakthrough_probability_multiplier"] *= (new / old) if old > 0 else 1.0
        
        return adjustments
    
    async def start_continuous_monitoring(self):
        """Start continuous monitoring in the background."""
        
        logger.info("Starting continuous real-time monitoring")
        
        while True:
            try:
                # Check if update is needed
                if datetime.now() - self.last_update > self.update_interval:
                    update = await self.run_calibration_cycle()
                    
                    # Log summary
                    logger.info(f"Calibration update: {len(update.anomalies_detected)} anomalies, "
                               f"{update.timeline_adjustment:+.2f} year adjustment")
                
                # Sleep for an hour before checking again
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(3600)  # Wait before retrying


def test_realtime_calibration():
    """Test the real-time calibration system."""
    
    print("=" * 50)
    print("REAL-TIME CALIBRATION SYSTEM TEST")
    print("=" * 50)
    
    # Initialize calibrator
    calibrator = RealTimeCalibrator(update_interval_hours=0.1)  # Quick test
    
    # Run one calibration cycle
    print("\nRunning calibration cycle...")
    
    # Use asyncio to run the async method
    import asyncio
    
    async def run_test():
        update = await calibrator.run_calibration_cycle()
        
        print(f"\n=== Calibration Results ===")
        print(f"Data points processed: {update.data_points_processed}")
        print(f"Timeline adjustment: {update.timeline_adjustment:+.2f} years")
        print(f"Confidence change: {update.confidence_change:+.1%}")
        
        if update.anomalies_detected:
            print(f"\n=== Anomalies Detected ===")
            for anomaly in update.anomalies_detected:
                print(f"- {anomaly}")
        
        if update.parameters_updated:
            print(f"\n=== Parameters Updated ===")
            for param, (old, new) in update.parameters_updated.items():
                print(f"{param}: {old:.3f} → {new:.3f}")
        
        if update.recommendations:
            print(f"\n=== Recommendations ===")
            for rec in update.recommendations:
                print(f"• {rec}")
        
        # Get current adjustments
        adjustments = calibrator.get_current_adjustments()
        print(f"\n=== Current Model Adjustments ===")
        print(f"Timeline: {adjustments['timeline_adjustment']:+.2f} years")
        print(f"Confidence: {adjustments['confidence_multiplier']:.2f}x")
        print(f"Growth rate: {adjustments['growth_rate_adjustment']:.2f}x")
        print(f"Breakthrough prob: {adjustments['breakthrough_probability_multiplier']:.2f}x")
    
    # Run the test
    asyncio.run(run_test())


if __name__ == "__main__":
    test_realtime_calibration()
