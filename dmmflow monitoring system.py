import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import time
import json
from datetime import datetime
import logging
from collections import deque
import threading
from queue import Queue
import pandas as pd

@dataclass
class PerformanceMetrics:
    """Container for performance-related metrics."""
    loss: float
    accuracy: float
    training_time: float
    memory_usage: float
    cpu_usage: float
    gradient_norm: float
    
@dataclass
class PrivacyMetrics:
    """Container for privacy-related metrics."""
    epsilon: float
    delta: float
    noise_scale: float
    gradient_clipping_norm: float
    
@dataclass
class CommunicationMetrics:
    """Container for communication-related metrics."""
    bandwidth_usage: float
    compression_ratio: float
    latency: float
    packet_loss: float
    
@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics."""
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: Optional[float]
    disk_io: float
    network_io: float

class MetricsBuffer:
    """Circular buffer for storing historical metrics."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.performance_buffer = deque(maxlen=max_size)
        self.privacy_buffer = deque(maxlen=max_size)
        self.communication_buffer = deque(maxlen=max_size)
        self.resource_buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        
    def add_metrics(self,
                   performance: PerformanceMetrics,
                   privacy: PrivacyMetrics,
                   communication: CommunicationMetrics,
                   resource: ResourceMetrics):
        """Add new metrics to the buffer."""
        with self.lock:
            timestamp = datetime.now()
            self.performance_buffer.append((timestamp, performance))
            self.privacy_buffer.append((timestamp, privacy))
            self.communication_buffer.append((timestamp, communication))
            self.resource_buffer.append((timestamp, resource))
            
    def get_recent_metrics(self, window_size: int) -> Dict:
        """Get the most recent metrics within the specified window."""
        with self.lock:
            return {
                'performance': list(self.performance_buffer)[-window_size:],
                'privacy': list(self.privacy_buffer)[-window_size:],
                'communication': list(self.communication_buffer)[-window_size:],
                'resource': list(self.resource_buffer)[-window_size:]
            }

class AlertSystem:
    """System for monitoring and generating alerts based on metric thresholds."""
    
    def __init__(self):
        self.alert_queue = Queue()
        self.alert_thresholds = {
            'loss_threshold': 0.5,
            'accuracy_threshold': 0.85,
            'privacy_budget_threshold': 1.0,
            'bandwidth_threshold': 100.0,  # MB/s
            'latency_threshold': 1000.0,   # ms
            'resource_utilization_threshold': 0.9
        }
        self.logger = logging.getLogger("DMMFLO_AlertSystem")
        
    def check_performance_alerts(self, metrics: PerformanceMetrics) -> List[str]:
        """Check for performance-related alerts."""
        alerts = []
        if metrics.loss > self.alert_thresholds['loss_threshold']:
            alerts.append(f"High loss value detected: {metrics.loss:.4f}")
        if metrics.accuracy < self.alert_thresholds['accuracy_threshold']:
            alerts.append(f"Low accuracy detected: {metrics.accuracy:.4f}")
        return alerts
    
    def check_privacy_alerts(self, metrics: PrivacyMetrics) -> List[str]:
        """Check for privacy-related alerts."""
        alerts = []
        if metrics.epsilon > self.alert_thresholds['privacy_budget_threshold']:
            alerts.append(f"Privacy budget exceeded: ε = {metrics.epsilon:.4f}")
        return alerts
    
    def check_communication_alerts(self, metrics: CommunicationMetrics) -> List[str]:
        """Check for communication-related alerts."""
        alerts = []
        if metrics.bandwidth_usage > self.alert_thresholds['bandwidth_threshold']:
            alerts.append(f"High bandwidth usage: {metrics.bandwidth_usage:.2f} MB/s")
        if metrics.latency > self.alert_thresholds['latency_threshold']:
            alerts.append(f"High latency detected: {metrics.latency:.2f} ms")
        return alerts
    
    def check_resource_alerts(self, metrics: ResourceMetrics) -> List[str]:
        """Check for resource-related alerts."""
        alerts = []
        if metrics.cpu_utilization > self.alert_thresholds['resource_utilization_threshold']:
            alerts.append(f"High CPU utilization: {metrics.cpu_utilization:.2%}")
        if metrics.memory_utilization > self.alert_thresholds['resource_utilization_threshold']:
            alerts.append(f"High memory utilization: {metrics.memory_utilization:.2%}")
        return alerts

class MetricsAnalyzer:
    """Analytics engine for processing and analyzing collected metrics."""
    
    def __init__(self, metrics_buffer: MetricsBuffer):
        self.metrics_buffer = metrics_buffer
        self.logger = logging.getLogger("DMMFLO_MetricsAnalyzer")
        
    def calculate_moving_averages(self, window_size: int = 10) -> Dict:
        """Calculate moving averages for key metrics."""
        recent_metrics = self.metrics_buffer.get_recent_metrics(window_size)
        
        performance_df = pd.DataFrame([
            {
                'loss': m.loss,
                'accuracy': m.accuracy,
                'training_time': m.training_time
            }
            for _, m in recent_metrics['performance']
        ])
        
        return {
            'loss_ma': performance_df['loss'].mean(),
            'accuracy_ma': performance_df['accuracy'].mean(),
            'training_time_ma': performance_df['training_time'].mean()
        }
    
    def analyze_convergence(self) -> Dict:
        """Analyze model convergence patterns."""
        recent_metrics = self.metrics_buffer.get_recent_metrics(100)
        loss_values = [m.loss for _, m in recent_metrics['performance']]
        
        return {
            'is_converging': self._check_convergence(loss_values),
            'convergence_rate': self._calculate_convergence_rate(loss_values),
            'stability_score': self._calculate_stability_score(loss_values)
        }
    
    def _check_convergence(self, loss_values: List[float]) -> bool:
        """Check if the model is converging."""
        if len(loss_values) < 10:
            return False
        
        recent_slope = np.polyfit(range(10), loss_values[-10:], 1)[0]
        return recent_slope < -0.001
    
    def _calculate_convergence_rate(self, loss_values: List[float]) -> float:
        """Calculate the rate of convergence."""
        if len(loss_values) < 2:
            return 0.0
        
        return (loss_values[0] - loss_values[-1]) / len(loss_values)
    
    def _calculate_stability_score(self, loss_values: List[float]) -> float:
        """Calculate training stability score."""
        if len(loss_values) < 10:
            return 0.0
        
        return 1.0 / (1.0 + np.std(loss_values[-10:]))

class DMMFLOMonitor:
    """Main monitoring system for the DMMFLO algorithm."""
    
    def __init__(self):
        self.metrics_buffer = MetricsBuffer()
        self.alert_system = AlertSystem()
        self.metrics_analyzer = MetricsAnalyzer(self.metrics_buffer)
        self.logger = logging.getLogger("DMMFLO_Monitor")
        
    def start_monitoring(self):
        """Start the monitoring system."""
        self.logger.info("Starting DMMFLO monitoring system...")
        self._start_metric_collection()
        self._start_alert_processing()
        
    def _start_metric_collection(self):
        """Start collecting metrics in a separate thread."""
        def collect_metrics():
            while True:
                try:
                    metrics = self._collect_current_metrics()
                    self.metrics_buffer.add_metrics(**metrics)
                    self._process_alerts(metrics)
                    time.sleep(1)  # Collect metrics every second
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {str(e)}")
                    
        threading.Thread(target=collect_metrics, daemon=True).start()
        
    def _start_alert_processing(self):
        """Start processing alerts in a separate thread."""
        def process_alerts():
            while True:
                try:
                    alert = self.alert_system.alert_queue.get()
                    self._handle_alert(alert)
                except Exception as e:
                    self.logger.error(f"Error processing alert: {str(e)}")
                    
        threading.Thread(target=process_alerts, daemon=True).start()
        
    def _collect_current_metrics(self) -> Dict:
        """Collect current system metrics."""
        # Implement metric collection logic here
        return {
            'performance': PerformanceMetrics(
                loss=0.0,
                accuracy=0.0,
                training_time=0.0,
                memory_usage=0.0,
                cpu_usage=0.0,
                gradient_norm=0.0
            ),
            'privacy': PrivacyMetrics(
                epsilon=0.0,
                delta=0.0,
                noise_scale=0.0,
                gradient_clipping_norm=0.0
            ),
            'communication': CommunicationMetrics(
                bandwidth_usage=0.0,
                compression_ratio=0.0,
                latency=0.0,
                packet_loss=0.0
            ),
            'resource': ResourceMetrics(
                cpu_utilization=0.0,
                memory_utilization=0.0,
                gpu_utilization=None,
                disk_io=0.0,
                network_io=0.0
            )
        }
        
    def _process_alerts(self, metrics: Dict):
        """Process metrics and generate alerts if necessary."""
        alerts = []
        alerts.extend(self.alert_system.check_performance_alerts(metrics['performance']))
        alerts.extend(self.alert_system.check_privacy_alerts(metrics['privacy']))
        alerts.extend(self.alert_system.check_communication_alerts(metrics['communication']))
        alerts.extend(self.alert_system.check_resource_alerts(metrics['resource']))
        
        for alert in alerts:
            self.alert_system.alert_queue.put(alert)
            
    def _handle_alert(self, alert: str):
        """Handle generated alerts."""
        self.logger.warning(f"Alert: {alert}")
        # Implement alert handling logic here (e.g., sending notifications)
        
    def get_system_status(self) -> Dict:
        """Get current system status and analytics."""
        moving_averages = self.metrics_analyzer.calculate_moving_averages()
        convergence_analysis = self.metrics_analyzer.analyze_convergence()
        
        return {
            'metrics': self.metrics_buffer.get_recent_metrics(10),
            'analysis': {
                'moving_averages': moving_averages,
                'convergence': convergence_analysis
            }
        }
        
    def export_metrics(self, format: str = 'json') -> str:
        """Export collected metrics in the specified format."""
        metrics = self.metrics_buffer.get_recent_metrics(1000)
        
        if format == 'json':
            return json.dumps({
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'analysis': self.get_system_status()['analysis']
            })
        else:
            raise ValueError(f"Unsupported export format: {format}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and start the monitoring system
    monitor = DMMFLOMonitor()
    monitor.start_monitoring()
    
    # Run for a while to collect metrics
    try:
        while True:
            status = monitor.get_system_status()
            print(f"System Status: {json.dumps(status, indent=2)}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("Monitoring stopped by user")