import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import time
import random
import math
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
from enum import Enum
import pickle
import csv
import os
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class OrderPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EMERGENCY = 4

class DroneStatus(Enum):
    IDLE = "idle"
    FLYING_TO_PICKUP = "flying_to_pickup"
    FLYING_TO_DELIVERY = "flying_to_delivery"
    RETURNING_TO_DEPOT = "returning_to_depot"
    CHARGING = "charging"

@dataclass
class Order:
    id: int
    pickup_location: Tuple[float, float]
    delivery_location: Tuple[float, float]
    priority: OrderPriority
    arrival_time: float
    deadline: float
    weight: float = 1.0
    score: int = 0
    completed: bool = False
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    assigned_drone: Optional[int] = None
    end_depot_id: Optional[int] = None

@dataclass
class Depot:
    id: int
    location: Tuple[float, float]

@dataclass
class Drone:
    id: int
    depot_id: int
    location: Tuple[float, float]
    status: DroneStatus
    speed_mps: float
    battery_duration_seconds: float
    current_battery_seconds: float
    charging_time_seconds: float
    payload_capacity: float
    current_payload: float = 0.0
    current_order: Optional[Order] = None
    route: List[Tuple[float, float]] = field(default_factory=list)
    total_distance_meters: float = 0.0
    total_flight_time: float = 0.0
    total_idle_time: float = 0.0
    charging_start_time: Optional[float] = None
    mission_start_time: Optional[float] = None
    target_depot_id: Optional[int] = None
    
    # Animation support
    full_route: List[Tuple[float, float]] = field(default_factory=list)
    route_progress: float = 0.0

class AdvancedDroneRoutingDataset(Dataset):
    """Enhanced dataset with cross-depot awareness and future planning features"""
    
    def __init__(self, training_data: List[Dict]):
        self.samples = []
        self.labels = []
        self.depot_assignments = []  # Track optimal depot assignments
        
        for data_sample in training_data:
            # Extract enhanced state features
            state_features = self._extract_enhanced_state_features(data_sample['state'])
            chosen_drone_id = data_sample['action']['chosen_drone_id']
            chosen_end_depot_id = data_sample['action'].get('chosen_end_depot_id', chosen_drone_id)
            
            # Create training samples for each drone candidate
            for i, drone_candidate in enumerate(data_sample['state']['drone_candidates']):
                # Enhanced input features with cross-depot information
                input_features = np.concatenate([
                    state_features,
                    self._extract_enhanced_drone_features(drone_candidate),
                    self._extract_depot_features(data_sample['state'], drone_candidate)
                ])
                
                # Multi-task labels: assignment probability + depot selection
                assignment_label = 1.0 if drone_candidate['drone_id'] == chosen_drone_id else 0.0
                optimal_depot_id = drone_candidate.get('optimal_end_depot_id', drone_candidate['drone_id'])
                cross_depot_label = 1.0 if drone_candidate.get('is_cross_depot_operation', False) else 0.0
                
                self.samples.append(input_features)
                self.labels.append(assignment_label)
                self.depot_assignments.append([optimal_depot_id, cross_depot_label])
        
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels)
        self.depot_assignments = np.array(self.depot_assignments)
        
        # Advanced feature normalization
        self.scaler = StandardScaler()
        self.samples = self.scaler.fit_transform(self.samples)
        
        print(f"Enhanced dataset: {len(self.samples)} samples, {self.samples.shape[1]} features")
        print(f"Cross-depot operations: {np.sum(self.depot_assignments[:, 1]):.0f}/{len(self.depot_assignments)} ({np.mean(self.depot_assignments[:, 1])*100:.1f}%)")
    
    def _extract_enhanced_state_features(self, state: Dict) -> np.ndarray:
        """Extract enhanced state features with future planning capabilities"""
        features = []
        
        # Order features
        order = state['order_features']
        features.extend([
            order['priority_value'],
            order['score'],
            order['weight'],
            order['time_until_deadline'],
            order['urgency_ratio'],
            order['order_distance'] / 1000,
        ])
        
        # System state features with future planning
        sys_state = state['system_state']
        features.extend([
            sys_state['total_pending_orders'],
            sys_state['total_completed_orders'],
            sys_state['current_score'],
            sys_state['current_utilization'],
            sys_state.get('cross_depot_operations', 0),  # Cross-depot awareness
            sys_state['pending_orders_by_priority']['emergency'],
            sys_state['pending_orders_by_priority']['high'],
            sys_state['pending_orders_by_priority']['medium'],
            sys_state['pending_orders_by_priority']['low'],
        ])
        
        # Future workload prediction features
        total_pending = sys_state['total_pending_orders']
        high_priority_ratio = (sys_state['pending_orders_by_priority']['emergency'] + 
                              sys_state['pending_orders_by_priority']['high']) / max(1, total_pending)
        workload_pressure = min(1.0, total_pending / 10.0)  # Normalize workload
        
        features.extend([
            high_priority_ratio,
            workload_pressure,
            total_pending / max(1, sys_state['total_completed_orders'])  # Pending/completed ratio
        ])
        
        # Aggregate drone features with depot distribution
        drone_candidates = state['drone_candidates']
        if drone_candidates:
            avg_battery = np.mean([d['battery_level'] for d in drone_candidates])
            min_distance = np.min([d['distance_to_pickup'] for d in drone_candidates])
            max_distance = np.max([d['distance_to_pickup'] for d in drone_candidates])
            num_candidates = len(drone_candidates)
            
            # Cross-depot opportunity metrics
            cross_depot_candidates = sum(1 for d in drone_candidates if d.get('is_cross_depot_operation', False))
            cross_depot_ratio = cross_depot_candidates / max(1, num_candidates)
        else:
            avg_battery = min_distance = max_distance = num_candidates = cross_depot_ratio = 0
        
        features.extend([avg_battery, min_distance/1000, max_distance/1000, num_candidates, cross_depot_ratio])
        
        return np.array(features)
    
    def _extract_enhanced_drone_features(self, drone_candidate: Dict) -> np.ndarray:
        """Extract enhanced drone features with efficiency metrics"""
        return np.array([
            drone_candidate['battery_level'],
            drone_candidate['distance_to_pickup'] / 1000,
            drone_candidate['distance_delivery_to_depot'] / 1000,
            drone_candidate['total_mission_distance'] / 1000,
            drone_candidate['estimated_mission_time_minutes'],
            drone_candidate['battery_usage_ratio'],
            drone_candidate['time_efficiency'],
            1.0 if drone_candidate['can_complete_on_time'] else 0.0,
            1.0 if drone_candidate.get('is_cross_depot_operation', False) else 0.0,  # Cross-depot indicator
            drone_candidate.get('optimal_end_depot_id', drone_candidate['drone_id']),  # Optimal depot ID
        ])
    
    def _extract_depot_features(self, state: Dict, drone_candidate: Dict) -> np.ndarray:
        """Extract depot-specific features for cross-depot optimization"""
        # Depot load balancing features
        current_depot_id = drone_candidate['drone_id']  # Assuming drone ID = depot ID
        optimal_depot_id = drone_candidate.get('optimal_end_depot_id', current_depot_id)
        
        # Distance efficiency gain from cross-depot operation
        current_depot_distance = drone_candidate['distance_delivery_to_depot']
        cross_depot_distance_savings = max(0, current_depot_distance - 
                                          (drone_candidate['total_mission_distance'] - 
                                           drone_candidate['distance_to_pickup'] - 
                                           (state['order_features']['order_distance'])))
        
        return np.array([
            1.0 if optimal_depot_id != current_depot_id else 0.0,  # Cross-depot opportunity
            cross_depot_distance_savings / 1000,  # Distance savings in km
            optimal_depot_id,  # Target depot ID
            abs(optimal_depot_id - current_depot_id),  # Depot distance (conceptual)
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return (torch.FloatTensor(self.samples[idx]), 
                torch.FloatTensor([self.labels[idx]]),
                torch.FloatTensor(self.depot_assignments[idx]))

class CrossDepotAwareGNN(nn.Module):
    """Advanced GNN with explicit cross-depot optimization and future planning"""
    
    def __init__(self, input_dim, hidden_dim=512, num_depots=5, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_depots = num_depots
        
        # Multi-scale feature embedding
        self.order_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Specialized cross-depot reasoning network
        self.cross_depot_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Future planning and workload prediction
        self.future_planning_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Multi-head attention for depot-drone interactions
        self.depot_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2, 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Depot-specific embeddings
        self.depot_embeddings = nn.Embedding(num_depots, 64)
        
        # Assignment probability head
        self.assignment_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32 + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Cross-depot operation predictor
        self.cross_depot_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Optimal depot selector
        self.depot_selector = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_depots),
            nn.Softmax(dim=-1)
        )
        
        # Value estimation for future planning
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x, depot_ids=None):
        batch_size = x.size(0)
        
        # Feature embedding
        embedded = self.order_embedding(x)
        
        # Cross-depot reasoning
        cross_depot_features = self.cross_depot_encoder(embedded)
        
        # Future planning features
        future_features = self.future_planning_head(cross_depot_features)
        
        # Depot attention mechanism
        attended_features, _ = self.depot_attention(
            cross_depot_features.unsqueeze(1),
            cross_depot_features.unsqueeze(1),
            cross_depot_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # Depot embeddings (use depot ID if provided, otherwise use learnable embeddings)
        if depot_ids is not None:
            depot_embeds = self.depot_embeddings(depot_ids.long())
        else:
            # Use average depot embedding for inference
            depot_embeds = self.depot_embeddings.weight.mean(dim=0).unsqueeze(0).repeat(batch_size, 1)
        
        # Combine features for final predictions
        combined_features = torch.cat([attended_features, future_features], dim=1)
        assignment_features = torch.cat([combined_features, depot_embeds], dim=1)
        
        # Multi-task outputs
        assignment_prob = self.assignment_predictor(assignment_features)
        cross_depot_prob = self.cross_depot_predictor(combined_features)
        depot_preferences = self.depot_selector(combined_features)
        future_value = self.value_estimator(combined_features)
        
        # Enhanced scoring with cross-depot awareness and future planning
        enhanced_score = (
            assignment_prob.squeeze(-1) * 0.4 +  # Base assignment probability
            cross_depot_prob.squeeze(-1) * 0.3 +  # Cross-depot optimization bonus
            future_value.squeeze(-1) * 0.2 +      # Future value consideration
            depot_preferences.max(dim=1)[0] * 0.1  # Depot preference strength
        )
        
        return {
            'assignment_score': torch.sigmoid(enhanced_score),
            'cross_depot_probability': cross_depot_prob.squeeze(-1),
            'depot_preferences': depot_preferences,
            'future_value': future_value.squeeze(-1),
            'raw_assignment_prob': assignment_prob.squeeze(-1)
        }

class EnhancedGNNDroneRoutingOptimizer:
    """GNN-based optimizer with advanced cross-depot operations and future planning"""
    
    def __init__(self, instance: Dict, model_path: str = None):
        self.instance = instance
        self.depots = instance['depots']
        self.drone_specs = instance['drone_specs']
        self.time_horizon = instance['time_horizon']
        self.total_possible_score = instance['total_possible_score']
        
        self.drones = []
        self.pending_orders = []
        self.completed_orders = []
        self.failed_orders = []
        self.current_time = 0.0
        
        # Advanced GNN model components
        self.model = None
        self.scaler = None
        self.model_path = model_path
        
        # Load trained model if available
        if model_path and os.path.exists(model_path):
            self._load_trained_model()
        
        # Performance tracking
        self.assignment_times = []
        self.gnn_predictions = []
        self.cross_depot_decisions = []
        self.future_planning_scores = []
        
        # Initialize drones
        self._initialize_drones()
        
        # Enhanced metrics
        self.metrics = {
            'total_score': 0,
            'total_possible_score': self.total_possible_score,
            'score_ratio': 0.0,
            'total_orders_completed': 0,
            'total_orders_failed': 0,
            'total_distance_km': 0.0,
            'total_flight_time_hours': 0.0,
            'total_idle_time_hours': 0.0,
            'average_delivery_time': 0.0,
            'average_assignment_time': 0.0,
            'drone_utilization': 0.0,
            'emergency_completed': 0,
            'high_completed': 0,
            'medium_completed': 0,
            'low_completed': 0,
            'emergency_total': 0,
            'high_total': 0,
            'medium_total': 0,
            'low_total': 0,
            'emergency_completion_rate': 0.0,
            'high_completion_rate': 0.0,
            'medium_completion_rate': 0.0,
            'low_completion_rate': 0.0,
            'cross_depot_operations': 0,
            'gnn_cross_depot_decisions': 0,
            'future_planning_accuracy': 0.0
        }
    
    def _load_trained_model(self):
        """Load trained advanced GNN model"""
        try:
            checkpoint = torch.load(self.model_path, map_location='gpu')
            
            input_dim = checkpoint.get('input_dim', 32)
            num_depots = checkpoint.get('num_depots', len(self.depots))
            
            self.model = CrossDepotAwareGNN(input_dim, num_depots=num_depots)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Load scaler
            if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                self.scaler = StandardScaler()
                self.scaler.mean_ = checkpoint['scaler_mean']
                self.scaler.scale_ = checkpoint['scaler_scale']
            
            print(f"Loaded advanced cross-depot GNN model from {self.model_path}")
            
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            self.model = None
            self.scaler = None
    
    def _initialize_drones(self):
        """Initialize one drone per depot"""
        for depot in self.depots:
            drone = Drone(
                id=depot.id,
                depot_id=depot.id,
                location=depot.location,
                status=DroneStatus.IDLE,
                speed_mps=self.drone_specs['cruising_speed_mps'],
                battery_duration_seconds=self.drone_specs['battery_duration_minutes'] * 60,
                current_battery_seconds=self.drone_specs['battery_duration_minutes'] * 60,
                charging_time_seconds=self.drone_specs['charging_time_minutes'] * 60,
                payload_capacity=self.drone_specs['max_payload_kg']
            )
            self.drones.append(drone)
    
    def haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance using Haversine formula (returns meters)"""
        lat1, lon1 = math.radians(coord1[1]), math.radians(coord1[0])
        lat2, lon2 = math.radians(coord2[1]), math.radians(coord2[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        R = 6371000
        return R * c
    
    def calculate_flight_time(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate flight time between coordinates (returns seconds)"""
        distance_m = self.haversine_distance(coord1, coord2)
        return distance_m / self.drone_specs['cruising_speed_mps']
    
    def find_optimal_depot(self, delivery_location: Tuple[float, float]) -> Depot:
        """Find optimal depot for delivery location"""
        return min(self.depots, 
                  key=lambda d: self.haversine_distance(delivery_location, d.location))
    
    def calculate_total_mission_time_with_depot(self, drone: Drone, order: Order, end_depot: Depot) -> float:
        """Calculate mission time with specific end depot"""
        to_pickup_time = self.calculate_flight_time(drone.location, order.pickup_location)
        pickup_to_delivery_time = self.calculate_flight_time(order.pickup_location, order.delivery_location)
        delivery_to_depot_time = self.calculate_flight_time(order.delivery_location, end_depot.location)
        
        takeoff_landing_time = self.drone_specs['takeoff_landing_time_seconds'] * 3
        service_time = self.drone_specs['service_time_seconds'] * 2
        
        return to_pickup_time + pickup_to_delivery_time + delivery_to_depot_time + takeoff_landing_time + service_time
    
    def can_complete_order(self, drone: Drone, order: Order) -> bool:
        """Check feasibility with optimal depot"""
        if drone.status != DroneStatus.IDLE or order.weight > drone.payload_capacity:
            return False
        
        optimal_depot = self.find_optimal_depot(order.delivery_location)
        mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_depot)
        
        if mission_time > drone.current_battery_seconds * 0.9:
            return False
        
        estimated_completion_time = self.current_time + mission_time / 60
        return estimated_completion_time <= order.deadline
    
    def add_order(self, order: Order):
        """Add order with priority tracking"""
        self.pending_orders.append(order)
        self.pending_orders.sort(key=lambda x: (-x.priority.value, x.deadline))
        
        priority_key = f"{order.priority.name.lower()}_total"
        if priority_key in self.metrics:
            self.metrics[priority_key] += 1
    
    def _create_enhanced_state_features(self, order: Order, available_drones: List[Drone]) -> Dict:
        """Create enhanced state representation for advanced GNN"""
        # Order features
        order_features = {
            'priority_value': order.priority.value,
            'score': order.score,
            'weight': order.weight,
            'time_until_deadline': max(0, order.deadline - self.current_time),
            'urgency_ratio': ((order.deadline - self.current_time) / 
                            (order.deadline - order.arrival_time)) if order.deadline > order.arrival_time else 0,
            'order_distance': self.haversine_distance(order.pickup_location, order.delivery_location)
        }
        
        # Enhanced system state with future planning
        future_high_priority_orders = len([o for o in self.pending_orders 
                                         if o.priority.value >= OrderPriority.HIGH.value and 
                                         o.deadline - self.current_time < 60])  # Next hour
        
        system_state = {
            'total_pending_orders': len(self.pending_orders),
            'total_completed_orders': len(self.completed_orders),
            'current_score': self.metrics['total_score'],
            'current_utilization': self.metrics['drone_utilization'],
            'cross_depot_operations': self.metrics['cross_depot_operations'],
            'future_urgent_orders': future_high_priority_orders,
            'pending_orders_by_priority': {
                'emergency': len([o for o in self.pending_orders if o.priority == OrderPriority.EMERGENCY]),
                'high': len([o for o in self.pending_orders if o.priority == OrderPriority.HIGH]),
                'medium': len([o for o in self.pending_orders if o.priority == OrderPriority.MEDIUM]),
                'low': len([o for o in self.pending_orders if o.priority == OrderPriority.LOW])
            }
        }
        
        # Enhanced drone candidates with cross-depot analysis
        drone_candidates = []
        for drone in available_drones:
            optimal_depot = self.find_optimal_depot(order.delivery_location)
            mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_depot)
            distance_to_pickup = self.haversine_distance(drone.location, order.pickup_location)
            distance_delivery_to_depot = self.haversine_distance(order.delivery_location, optimal_depot.location)
            order_distance = self.haversine_distance(order.pickup_location, order.delivery_location)
            
            # Cross-depot efficiency analysis
            current_depot_distance = self.haversine_distance(order.delivery_location, 
                                                           next(d.location for d in self.depots if d.id == drone.depot_id))
            cross_depot_savings = max(0, current_depot_distance - distance_delivery_to_depot)
            
            drone_features = {
                'drone_id': drone.id,
                'battery_level': drone.current_battery_seconds / drone.battery_duration_seconds,
                'distance_to_pickup': distance_to_pickup,
                'distance_delivery_to_depot': distance_delivery_to_depot,
                'total_mission_distance': distance_to_pickup + order_distance + distance_delivery_to_depot,
                'estimated_mission_time_minutes': mission_time / 60,
                'battery_usage_ratio': mission_time / drone.current_battery_seconds,
                'can_complete_on_time': self.current_time + mission_time/60 <= order.deadline,
                'time_efficiency': max(0, (order.deadline - self.current_time - mission_time/60)) / order.deadline,
                'optimal_end_depot_id': optimal_depot.id,
                'is_cross_depot_operation': optimal_depot.id != drone.depot_id,
                'cross_depot_savings': cross_depot_savings,
                'depot_load_balance_score': self._calculate_depot_load_balance_score(drone.depot_id, optimal_depot.id)
            }
            drone_candidates.append(drone_features)
        
        return {
            'order_features': order_features,
            'system_state': system_state,
            'drone_candidates': drone_candidates
        }
    
    def _calculate_depot_load_balance_score(self, current_depot_id: int, target_depot_id: int) -> float:
        """Calculate load balancing benefit score"""
        current_depot_load = len([d for d in self.drones if d.depot_id == current_depot_id and d.status != DroneStatus.IDLE])
        target_depot_load = len([d for d in self.drones if d.depot_id == target_depot_id and d.status != DroneStatus.IDLE])
        
        if current_depot_id == target_depot_id:
            return 0.0
        
        # Positive score if moving to less loaded depot
        return max(0, (current_depot_load - target_depot_load) / len(self.drones))
    
    def assign_orders_advanced_gnn(self):
        """Advanced GNN-based assignment with cross-depot optimization"""
        assignments = []
        
        for order in self.pending_orders[:]:
            if order.assigned_drone is not None:
                continue
            
            assignment_start_time = time.time()
            available_drones = [drone for drone in self.drones if self.can_complete_order(drone, order)]
            
            if not available_drones:
                continue
            
            best_drone = None
            best_end_depot_id = None
            best_score = -float('inf')
            
            if self.model is not None:
                try:
                    state_features = self._create_enhanced_state_features(order, available_drones)
                    state_base = self._extract_state_features_for_gnn(state_features)
                    
                    drone_scores = []
                    for drone in available_drones:
                        drone_candidate = next(d for d in state_features['drone_candidates'] 
                                             if d['drone_id'] == drone.id)
                        drone_features = self._extract_drone_features_for_gnn(drone_candidate)
                        depot_features = self._extract_depot_features_for_gnn(drone_candidate)
                        
                        # Combine all features
                        input_features = np.concatenate([state_base, drone_features, depot_features])
                        
                        if self.scaler is not None:
                            input_features = self.scaler.transform(input_features.reshape(1, -1))[0]
                        
                        # Get GNN predictions
                        with torch.no_grad():
                            input_tensor = torch.FloatTensor(input_features).unsqueeze(0)
                            depot_id_tensor = torch.LongTensor([drone_candidate['optimal_end_depot_id']])
                            
                            predictions = self.model(input_tensor, depot_id_tensor)
                            
                            assignment_score = predictions['assignment_score'].item()
                            cross_depot_prob = predictions['cross_depot_probability'].item()
                            future_value = predictions['future_value'].item()
                            depot_prefs = predictions['depot_preferences'].squeeze().numpy()
                        
                        # Enhanced scoring with cross-depot emphasis
                        cross_depot_bonus = 0
                        optimal_depot_id = drone_candidate['optimal_end_depot_id']
                        
                        if optimal_depot_id != drone.depot_id:
                            cross_depot_bonus = cross_depot_prob * 5.0  # Strong cross-depot bonus
                            self.gnn_cross_depot_decisions += 1
                        
                        # Future planning bonus
                        future_bonus = future_value * 2.0
                        
                        # Depot preference bonus
                        depot_pref_bonus = depot_prefs[min(optimal_depot_id, len(depot_prefs)-1)] * 1.0
                        
                        # Combined GNN score with cross-depot optimization
                        final_score = (assignment_score * 3.0 +  # Base assignment
                                     cross_depot_bonus +        # Cross-depot operations
                                     future_bonus +             # Future planning
                                     depot_pref_bonus)          # Depot preferences
                        
                        drone_scores.append((drone, final_score, optimal_depot_id, cross_depot_prob))
                    
                    # Select best drone with highest combined score
                    if drone_scores:
                        best_drone, best_score, best_end_depot_id, cross_depot_score = max(drone_scores, key=lambda x: x[1])
                        self.gnn_predictions.append(best_score)
                        self.cross_depot_decisions.append(cross_depot_score)
                        
                        print(f"GNN Assignment: Order {order.id} → Drone {best_drone.id} → Depot {best_end_depot_id} "
                              f"(Score: {best_score:.3f}, Cross-Depot: {cross_depot_score:.3f})")
                
                except Exception as e:
                    print(f"GNN prediction failed: {e}, falling back to enhanced heuristic")
                    best_drone = None
            
            # Enhanced heuristic fallback with cross-depot awareness
            if best_drone is None:
                for drone in available_drones:
                    optimal_depot = self.find_optimal_depot(order.delivery_location)
                    mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_depot)
                    
                    # Enhanced scoring with strong cross-depot emphasis
                    base_score = order.score * 3.0
                    
                    completion_time = self.current_time + mission_time / 60
                    time_efficiency = max(0, (order.deadline - completion_time)) / order.deadline
                    time_bonus = time_efficiency * 8.0
                    
                    total_distance = (
                        self.haversine_distance(drone.location, order.pickup_location) +
                        self.haversine_distance(order.pickup_location, order.delivery_location) +
                        self.haversine_distance(order.delivery_location, optimal_depot.location)
                    )
                    distance_penalty = total_distance / 50000  # Reduced penalty scale
                    
                    battery_efficiency = drone.current_battery_seconds / drone.battery_duration_seconds
                    battery_bonus = battery_efficiency * 3.0
                    
                    # STRONG cross-depot bonus for heuristic
                    cross_depot_bonus = 0
                    if optimal_depot.id != drone.depot_id:
                        cross_depot_bonus = 15.0  # Very strong incentive
                        
                        # Additional load balancing bonus
                        load_balance_bonus = self._calculate_depot_load_balance_score(drone.depot_id, optimal_depot.id) * 5.0
                        cross_depot_bonus += load_balance_bonus
                    
                    # Future opportunity preservation
                    future_penalty = 0
                    if len(self.pending_orders) > 5:
                        future_penalty = -2.0
                    
                    urgency_multiplier = {
                        OrderPriority.EMERGENCY: 4.0,
                        OrderPriority.HIGH: 3.0,
                        OrderPriority.MEDIUM: 2.0,
                        OrderPriority.LOW: 1.0
                    }[order.priority]
                    
                    assignment_score = ((base_score + time_bonus + battery_bonus + cross_depot_bonus + future_penalty - distance_penalty) 
                                      * urgency_multiplier)
                    
                    if assignment_score > best_score:
                        best_score = assignment_score
                        best_drone = drone
                        best_end_depot_id = optimal_depot.id
            
            if best_drone:
                assignment_end_time = time.time()
                assignment_duration = assignment_end_time - assignment_start_time
                self.assignment_times.append(assignment_duration)
                
                order.assigned_drone = best_drone.id
                order.end_depot_id = best_end_depot_id
                assignments.append((best_drone, order))
                self.pending_orders.remove(order)
                
                # Track cross-depot operations
                if best_end_depot_id != best_drone.depot_id:
                    self.metrics['gnn_cross_depot_decisions'] += 1
        
        return assignments
    
    def _extract_state_features_for_gnn(self, state_features: Dict) -> np.ndarray:
        """Extract state features for advanced GNN"""
        features = []
        
        # Order features
        order = state_features['order_features']
        features.extend([
            order['priority_value'],
            order['score'],
            order['weight'],
            order['time_until_deadline'],
            order['urgency_ratio'],
            order['order_distance'] / 1000,
        ])
        
        # Enhanced system state
        sys_state = state_features['system_state']
        features.extend([
            sys_state['total_pending_orders'],
            sys_state['total_completed_orders'],
            sys_state['current_score'],
            sys_state['current_utilization'],
            sys_state['cross_depot_operations'],
            sys_state['future_urgent_orders'],
            sys_state['pending_orders_by_priority']['emergency'],
            sys_state['pending_orders_by_priority']['high'],
            sys_state['pending_orders_by_priority']['medium'],
            sys_state['pending_orders_by_priority']['low'],
        ])
        
        # Future planning features
        total_pending = sys_state['total_pending_orders']
        high_priority_ratio = (sys_state['pending_orders_by_priority']['emergency'] + 
                              sys_state['pending_orders_by_priority']['high']) / max(1, total_pending)
        workload_pressure = min(1.0, total_pending / 10.0)
        pending_completed_ratio = total_pending / max(1, sys_state['total_completed_orders'])
        
        features.extend([high_priority_ratio, workload_pressure, pending_completed_ratio])
        
        # Aggregate drone features with cross-depot metrics
        drone_candidates = state_features['drone_candidates']
        if drone_candidates:
            avg_battery = np.mean([d['battery_level'] for d in drone_candidates])
            min_distance = np.min([d['distance_to_pickup'] for d in drone_candidates])
            max_distance = np.max([d['distance_to_pickup'] for d in drone_candidates])
            num_candidates = len(drone_candidates)
            cross_depot_candidates = sum(1 for d in drone_candidates if d['is_cross_depot_operation'])
            cross_depot_ratio = cross_depot_candidates / max(1, num_candidates)
        else:
            avg_battery = min_distance = max_distance = num_candidates = cross_depot_ratio = 0
        
        features.extend([avg_battery, min_distance/1000, max_distance/1000, num_candidates, cross_depot_ratio])
        
        return np.array(features)
    
    def _extract_drone_features_for_gnn(self, drone_candidate: Dict) -> np.ndarray:
        """Extract enhanced drone features"""
        return np.array([
            drone_candidate['battery_level'],
            drone_candidate['distance_to_pickup'] / 1000,
            drone_candidate['distance_delivery_to_depot'] / 1000,
            drone_candidate['total_mission_distance'] / 1000,
            drone_candidate['estimated_mission_time_minutes'],
            drone_candidate['battery_usage_ratio'],
            drone_candidate['time_efficiency'],
            1.0 if drone_candidate['can_complete_on_time'] else 0.0,
            1.0 if drone_candidate['is_cross_depot_operation'] else 0.0,
            drone_candidate['optimal_end_depot_id'],
        ])
    
    def _extract_depot_features_for_gnn(self, drone_candidate: Dict) -> np.ndarray:
        """Extract depot-specific features for cross-depot optimization"""
        return np.array([
            1.0 if drone_candidate['is_cross_depot_operation'] else 0.0,
            drone_candidate.get('cross_depot_savings', 0) / 1000,
            drone_candidate['optimal_end_depot_id'],
            drone_candidate.get('depot_load_balance_score', 0),
        ])
    
    def update_drone_positions(self, dt: float):
        """Update drone positions with cross-depot support"""
        for drone in self.drones:
            # Handle charging
            if drone.status == DroneStatus.CHARGING:
                if drone.charging_start_time is None:
                    drone.charging_start_time = self.current_time
                
                charging_duration = (self.current_time - drone.charging_start_time) * 60
                if charging_duration >= drone.charging_time_seconds:
                    drone.current_battery_seconds = drone.battery_duration_seconds
                    drone.status = DroneStatus.IDLE
                    drone.charging_start_time = None
                continue
            
            # Handle idle drones
            if drone.status == DroneStatus.IDLE:
                drone.total_idle_time += dt
                
                if drone.current_battery_seconds < drone.battery_duration_seconds * 0.2:
                    drone.status = DroneStatus.CHARGING
                    drone.charging_start_time = self.current_time
                continue
            
            # Handle moving drones
            if not drone.route:
                continue
            
            target = drone.route[0]
            distance_to_target = self.haversine_distance(drone.location, target)
            move_distance_meters = drone.speed_mps * dt * 60
            
            if move_distance_meters >= distance_to_target:
                # Reached target
                drone.location = target
                drone.route.pop(0)
                drone.total_distance_meters += distance_to_target
                
                flight_time_seconds = distance_to_target / drone.speed_mps
                flight_time_minutes = flight_time_seconds / 60
                drone.total_flight_time += flight_time_minutes
                drone.current_battery_seconds -= flight_time_seconds
                
                # Handle state transitions
                if drone.status == DroneStatus.FLYING_TO_PICKUP and drone.current_order:
                    pickup_distance = self.haversine_distance(drone.location, drone.current_order.pickup_location)
                    if pickup_distance < 50:
                        drone.status = DroneStatus.FLYING_TO_DELIVERY
                        drone.route = [drone.current_order.delivery_location]
                        
                        if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                            end_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                            drone.full_route = [drone.current_order.delivery_location, end_depot.location]
                        
                        drone.current_order.pickup_time = self.current_time
                        drone.current_battery_seconds -= self.drone_specs['service_time_seconds']
                
                elif drone.status == DroneStatus.FLYING_TO_DELIVERY and drone.current_order:
                    delivery_distance = self.haversine_distance(drone.location, drone.current_order.delivery_location)
                    if delivery_distance < 50:
                        # Order completed
                        drone.current_order.delivery_time = self.current_time
                        drone.current_order.completed = True
                        self.completed_orders.append(drone.current_order)
                        
                        self.metrics['total_score'] += drone.current_order.score
                        priority_key = f"{drone.current_order.priority.name.lower()}_completed"
                        if priority_key in self.metrics:
                            self.metrics[priority_key] += 1
                        
                        # Return to optimal depot
                        if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                            target_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                        else:
                            target_depot = self.find_optimal_depot(drone.current_order.delivery_location)
                        
                        drone.route = [target_depot.location]
                        drone.full_route = [target_depot.location]
                        drone.status = DroneStatus.RETURNING_TO_DEPOT
                        drone.target_depot_id = target_depot.id
                        drone.current_order = None
                        
                        drone.current_battery_seconds -= self.drone_specs['service_time_seconds']
                
                elif drone.status == DroneStatus.RETURNING_TO_DEPOT:
                    if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                        target_depot_location = next(d.location for d in self.depots if d.id == drone.target_depot_id)
                    else:
                        target_depot_location = next(d.location for d in self.depots if d.id == drone.depot_id)
                    
                    depot_distance = self.haversine_distance(drone.location, target_depot_location)
                    if depot_distance < 50:
                        drone.status = DroneStatus.IDLE
                        drone.full_route = []
                        
                        # Update depot assignment for cross-depot operations
                        if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                            old_depot_id = drone.depot_id
                            new_depot_id = drone.target_depot_id
                            
                            if old_depot_id != new_depot_id:
                                self.metrics['cross_depot_operations'] += 1
                                drone.depot_id = new_depot_id
                                drone.target_depot_id = None
                                
                                print(f"GNN Cross-depot completed: Drone {drone.id} moved from Depot {old_depot_id} to Depot {new_depot_id}")
                        
                        drone.current_battery_seconds -= self.drone_specs['takeoff_landing_time_seconds']
            
            else:
                # Move towards target
                bearing = math.atan2(target[1] - drone.location[1], target[0] - drone.location[0])
                
                lat_change = (move_distance_meters * math.sin(bearing)) / 111320
                lon_change = (move_distance_meters * math.cos(bearing)) / (111320 * math.cos(math.radians(drone.location[1])))
                
                new_lat = drone.location[1] + lat_change
                new_lon = drone.location[0] + lon_change
                drone.location = (new_lon, new_lat)
                
                drone.total_distance_meters += move_distance_meters
                flight_time_minutes = dt
                drone.total_flight_time += flight_time_minutes
                drone.current_battery_seconds -= dt * 60
    
    def execute_assignments(self, assignments: List[Tuple[Drone, Order]]):
        """Execute assignments with cross-depot planning"""
        for drone, order in assignments:
            drone.current_order = order
            drone.status = DroneStatus.FLYING_TO_PICKUP
            drone.route = [order.pickup_location]
            
            if hasattr(order, 'end_depot_id') and order.end_depot_id is not None:
                end_depot = next(d for d in self.depots if d.id == order.end_depot_id)
                drone.full_route = [order.pickup_location, order.delivery_location, end_depot.location]
            else:
                optimal_depot = self.find_optimal_depot(order.delivery_location)
                drone.full_route = [order.pickup_location, order.delivery_location, optimal_depot.location]
                order.end_depot_id = optimal_depot.id
            
            drone.mission_start_time = self.current_time
            drone.current_battery_seconds -= self.drone_specs['takeoff_landing_time_seconds']
    
    def handle_expired_orders(self):
        """Handle expired orders"""
        expired_orders = [order for order in self.pending_orders if self.current_time > order.deadline]
        for order in expired_orders:
            self.pending_orders.remove(order)
            self.failed_orders.append(order)
    
    def step(self, dt: float, new_orders: List[Order] = None):
        """Execute simulation step with advanced GNN"""
        self.current_time += dt
        
        if new_orders:
            for order in new_orders:
                self.add_order(order)
        
        self.handle_expired_orders()
        self.update_drone_positions(dt)
        
        # Use advanced GNN assignment
        assignments = self.assign_orders_advanced_gnn()
        self.execute_assignments(assignments)
        
        self._update_metrics()
    
    def _update_metrics(self):
        """Update comprehensive metrics"""
        self.metrics['total_orders_completed'] = len(self.completed_orders)
        self.metrics['total_orders_failed'] = len(self.failed_orders)
        self.metrics['score_ratio'] = self.metrics['total_score'] / self.total_possible_score if self.total_possible_score > 0 else 0
        self.metrics['total_distance_km'] = sum(drone.total_distance_meters for drone in self.drones) / 1000
        self.metrics['total_flight_time_hours'] = sum(drone.total_flight_time for drone in self.drones) / 60
        self.metrics['total_idle_time_hours'] = sum(drone.total_idle_time for drone in self.drones) / 60
        
        if self.assignment_times:
            self.metrics['average_assignment_time'] = np.mean(self.assignment_times)
        
        if self.completed_orders:
            delivery_times = [order.delivery_time - order.arrival_time for order in self.completed_orders if order.delivery_time]
            if delivery_times:
                self.metrics['average_delivery_time'] = np.mean(delivery_times)
        
        total_time = self.metrics['total_flight_time_hours'] + self.metrics['total_idle_time_hours']
        if total_time > 0:
            self.metrics['drone_utilization'] = self.metrics['total_flight_time_hours'] / total_time
        
        # Calculate future planning accuracy
        if self.future_planning_scores:
            self.metrics['future_planning_accuracy'] = np.mean(self.future_planning_scores)
        
        for priority in OrderPriority:
            priority_name = priority.name.lower()
            completed_key = f"{priority_name}_completed"
            total_key = f"{priority_name}_total"
            rate_key = f"{priority_name}_completion_rate"
            
            if self.metrics[total_key] > 0:
                self.metrics[rate_key] = self.metrics[completed_key] / self.metrics[total_key]
            else:
                self.metrics[rate_key] = 0.0
    
    def save_metrics_to_csv(self, filename: str):
        """Save enhanced metrics to CSV"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        final_metrics = {
            'instance_name': os.path.basename(filename).replace('metrics_', '').replace('.csv', '').split('_20')[0],
            'region': self.instance['region'],
            'size': self.instance['size'],
            'total_depots': len(self.depots),
            'total_orders': len(self.instance['orders']),
            'total_possible_score': self.total_possible_score,
            'total_score_achieved': self.metrics['total_score'],
            'score_ratio': self.metrics['score_ratio'],
            'orders_completed': self.metrics['total_orders_completed'],
            'orders_failed': self.metrics['total_orders_failed'],
            'orders_pending': len(self.pending_orders),
            'completion_rate': self.metrics['total_orders_completed'] / len(self.instance['orders']) if len(self.instance['orders']) > 0 else 0,
            'total_distance_km': self.metrics['total_distance_km'],
            'total_flight_time_hours': self.metrics['total_flight_time_hours'],
            'total_idle_time_hours': self.metrics['total_idle_time_hours'],
            'average_assignment_time_ms': self.metrics['average_assignment_time'] * 1000 if self.metrics['average_assignment_time'] > 0 else 0,
            'drone_utilization': self.metrics['drone_utilization'],
            'average_delivery_time_minutes': self.metrics['average_delivery_time'],
            'cross_depot_operations': self.metrics['cross_depot_operations'],
            'gnn_cross_depot_decisions': self.metrics['gnn_cross_depot_decisions'],
            'future_planning_accuracy': self.metrics['future_planning_accuracy'],
            'emergency_completed': self.metrics['emergency_completed'],
            'emergency_total': self.metrics['emergency_total'],
            'emergency_completion_rate': self.metrics['emergency_completion_rate'],
            'high_completed': self.metrics['high_completed'],
            'high_total': self.metrics['high_total'],
            'high_completion_rate': self.metrics['high_completion_rate'],
            'medium_completed': self.metrics['medium_completed'],
            'medium_total': self.metrics['medium_total'],
            'medium_completion_rate': self.metrics['medium_completion_rate'],
            'low_completed': self.metrics['low_completed'],
            'low_total': self.metrics['low_total'],
            'low_completion_rate': self.metrics['low_completion_rate'],
            'model_used': 'Advanced_GNN' if self.model else 'Enhanced_Heuristic',
            'gnn_predictions_made': len(self.gnn_predictions) if self.gnn_predictions else 0
        }
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = final_metrics.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(final_metrics)
        
        print(f"Enhanced metrics saved to {filename}")
    
    def print_final_report(self):
        """Print comprehensive final report"""
        print("\n" + "="*80)
        print("ADVANCED GNN DRONE ROUTING PERFORMANCE REPORT")
        print("="*80)
        
        print(f"Instance: {self.instance['region']} {self.instance['size']}")
        print(f"Model Used: {'Advanced Cross-Depot GNN' if self.model else 'Enhanced Heuristic'}")
        
        print(f"\nSCORE PERFORMANCE:")
        print(f"  Total Score Achieved: {self.metrics['total_score']}")
        print(f"  Total Possible Score: {self.total_possible_score}")
        print(f"  Score Ratio: {self.metrics['score_ratio']:.3f} ({self.metrics['score_ratio']*100:.1f}%)")
        
        print(f"\nCROSS-DEPOT OPERATIONS:")
        print(f"  GNN Cross-Depot Decisions: {self.metrics['gnn_cross_depot_decisions']}")
        print(f"  Cross-Depot Operations Completed: {self.metrics['cross_depot_operations']}")
        cross_depot_rate = (self.metrics['cross_depot_operations'] / max(1, self.metrics['total_orders_completed'])) * 100
        print(f"  Cross-Depot Success Rate: {cross_depot_rate:.1f}%")
        
        if self.gnn_predictions:
            print(f"\nGNN PERFORMANCE:")
            print(f"  Average GNN Assignment Score: {np.mean(self.gnn_predictions):.3f}")
            print(f"  GNN Predictions Made: {len(self.gnn_predictions)}")
            print(f"  Future Planning Accuracy: {self.metrics['future_planning_accuracy']:.3f}")
        
        if self.cross_depot_decisions:
            print(f"  Average Cross-Depot Confidence: {np.mean(self.cross_depot_decisions):.3f}")

# Enhanced visualizer and training functions would be similar to previous but with advanced features
# For brevity, including just the core training function

def train_advanced_gnn_model(epochs=300, batch_size=64, learning_rate=0.001):
    """Train advanced cross-depot aware GNN model"""
    
    print("="*80)
    print("TRAINING ADVANCED CROSS-DEPOT GNN MODEL")
    print("="*80)
    
    # Load training data from selected instances
    training_files = select_training_instances()
    
    if not training_files:
        print("No training files found! Please run greedy approach first.")
        return None, None
    
    print(f"Loading training data from {len(training_files)} instances...")
    
    all_training_data = []
    for file_path in training_files:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'assignment_decisions' in data:
                all_training_data.extend(data['assignment_decisions'])
                print(f"Loaded {len(data['assignment_decisions'])} samples from {os.path.basename(file_path)}")
        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if len(all_training_data) < 100:
        print(f"Insufficient training data: {len(all_training_data)} samples. Need at least 100.")
        return None, None
    
    print(f"Total training samples: {len(all_training_data)}")
    
    # Create enhanced dataset
    dataset = AdvancedDroneRoutingDataset(all_training_data)
    
    # Split data
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    # Initialize advanced model
    input_dim = dataset.samples.shape[1]
    num_depots = max(5, len(set(dataset.depot_assignments[:, 0])))
    model = CrossDepotAwareGNN(input_dim, num_depots=num_depots)
    
    # Multi-task loss with cross-depot emphasis
    assignment_criterion = nn.BCELoss()
    cross_depot_criterion = nn.BCELoss()
    depot_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 60
    
    print(f"Starting advanced training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch_features, batch_labels, batch_depot_info in train_loader:
            optimizer.zero_grad()
            
            depot_ids = batch_depot_info[:, 0].long()
            cross_depot_labels = batch_depot_info[:, 1]
            
            predictions = model(batch_features, depot_ids)
            
            # Multi-task loss
            assignment_loss = assignment_criterion(predictions['raw_assignment_prob'], batch_labels.squeeze())
            cross_depot_loss = cross_depot_criterion(predictions['cross_depot_probability'], cross_depot_labels) * 2.0  # Emphasize cross-depot
            value_loss = value_criterion(predictions['future_value'], (batch_labels.squeeze() - 0.5) * 2)  # Convert to [-1, 1]
            
            total_loss = assignment_loss + cross_depot_loss + value_loss
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += total_loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_labels, batch_depot_info in val_loader:
                depot_ids = batch_depot_info[:, 0].long()
                cross_depot_labels = batch_depot_info[:, 1]
                
                predictions = model(batch_features, depot_ids)
                
                assignment_loss = assignment_criterion(predictions['raw_assignment_prob'], batch_labels.squeeze())
                cross_depot_loss = cross_depot_criterion(predictions['cross_depot_probability'], cross_depot_labels) * 2.0
                value_loss = value_criterion(predictions['future_value'], (batch_labels.squeeze() - 0.5) * 2)
                
                total_loss = assignment_loss + cross_depot_loss + value_loss
                val_loss += total_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        scheduler.step()
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'num_depots': num_depots,
                'scaler_mean': dataset.scaler.mean_,
                'scaler_scale': dataset.scaler.scale_,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }, 'best_advanced_gnn_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Best: {best_val_loss:.4f}")
    
    print(f"Advanced training completed! Best validation loss: {best_val_loss:.4f}")
    print("Model saved as 'best_advanced_gnn_model.pth'")
    
    return model, dataset.scaler

def select_training_instances() -> List[str]:
    """Select first instance of each region and size for training"""
    regions = ['utah', 'north_carolina', 'texas', 'arkansas']
    sizes = ['small', 'medium', 'large']
    
    training_files = []
    for region in regions:
        for size in sizes:
            pattern = f"gnn_training_data/{region}_{size}_0_sol.pkl"
            if os.path.exists(pattern):
                training_files.append(pattern)
            else:
                print(f"Warning: Training file {pattern} not found")
    
    return training_files

def run_advanced_gnn_simulation(instance_file: str, dt: float = 0.5, save_metrics: bool = True, show_animation: bool = True):
    """Run simulation with advanced cross-depot GNN optimizer"""
    
    print(f"Loading instance from {instance_file}...")
    
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    print(f"Loaded {instance['region']} {instance['size']} instance")
    print(f"Depots: {len(instance['depots'])}, Orders: {len(instance['orders'])}")
    print(f"Total possible score: {instance['total_possible_score']}")
    print(f"Time horizon: {instance['time_horizon']:.1f} minutes")
    
    # Initialize advanced GNN optimizer
    model_path = 'best_advanced_gnn_model.pth'
    optimizer = EnhancedGNNDroneRoutingOptimizer(instance, model_path)
    
    # Initialize visualizer (simplified for this example)
    if show_animation:
        visualizer = ImprovedDroneRoutingVisualizer(optimizer, instance['bounds'])
    
    # Prepare orders for dynamic arrival
    orders = sorted(instance['orders'], key=lambda x: x.arrival_time)
    order_index = 0
    
    def animate(frame):
        nonlocal order_index
        
        # Add orders that should arrive at current time
        new_orders = []
        while (order_index < len(orders) and 
               orders[order_index].arrival_time <= optimizer.current_time):
            new_orders.append(orders[order_index])
            order_index += 1
        
        # Step simulation
        optimizer.step(dt, new_orders)
        
        # Update visualization
        if show_animation:
            visualizer.update_visualization(frame)
        
        # Print progress with advanced metrics
        if frame % 50 == 0:
            metrics = optimizer.metrics
            print(f"Time: {optimizer.current_time:.1f}, "
                  f"Score: {metrics['total_score']}/{optimizer.total_possible_score} "
                  f"({metrics['score_ratio']*100:.1f}%), "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Failed: {metrics['total_orders_failed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Cross-Depot: {metrics['cross_depot_operations']}, "
                  f"GNN Cross-Depot: {metrics['gnn_cross_depot_decisions']}")
    
    # Run simulation
    frames = int(instance['time_horizon'] / dt)
    
    if show_animation:
        # ani = animation.FuncAnimation(None, animate, 
        #                              frames=frames, 
        #                              interval=int(dt*100),
        #                              repeat=False, blit=False)
        
        ani = animation.FuncAnimation(visualizer.fig, animate, 
                             frames=frames, 
                             interval=int(dt*100),
                             repeat=False, blit=False)
        plt.tight_layout()
        plt.show()
    else:
        # Run without animation for faster execution
        for frame in range(frames):
            animate(frame)
    
    # Print final report
    optimizer.print_final_report()
    
    # Save results
    if save_metrics:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        instance_name = os.path.basename(instance_file).replace('.pkl', '')
        
        # Save metrics CSV (single row format)
        metrics_file = f"GNN_results/metrics_{instance_name}_{timestamp}.csv"
        optimizer.save_metrics_to_csv(metrics_file)
    
    return optimizer

def run_advanced_gnn_batch_experiments():
    """Run advanced GNN experiments on all instances"""
    
    # Create output directory
    os.makedirs('GNN_results', exist_ok=True)
    
    # Find all instance files
    instance_files = glob.glob('instances/*.pkl')
    
    if not instance_files:
        print("No instance files found in 'instances' folder!")
        return
    
    print(f"Found {len(instance_files)} instances to process with Advanced GNN")
    
    # Check if trained model exists
    model_path = 'best_advanced_gnn_model.pth'
    if not os.path.exists(model_path):
        print(f"Advanced trained model {model_path} not found!")
        print("Please run training first: python advanced_gnn.py train")
        return
    
    # Summary results
    summary_results = []
    
    for instance_file in instance_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(instance_file)}")
        print(f"{'='*80}")
        
        try:
            # Run simulation without animation
            optimizer = run_advanced_gnn_simulation(instance_file, dt=0.5, 
                                                   save_metrics=True, show_animation=False)
            
            # Collect summary data
            instance_name = os.path.basename(instance_file).replace('.pkl', '')
            summary_data = {
                'instance_name': instance_name,
                'region': optimizer.instance['region'],
                'size': optimizer.instance['size'],
                'total_depots': len(optimizer.depots),
                'total_orders': len(optimizer.instance['orders']),
                'total_possible_score': optimizer.total_possible_score,
                'total_score_achieved': optimizer.metrics['total_score'],
                'score_ratio': optimizer.metrics['score_ratio'],
                'orders_completed': optimizer.metrics['total_orders_completed'],
                'orders_failed': optimizer.metrics['total_orders_failed'],
                'completion_rate': optimizer.metrics['total_orders_completed'] / len(optimizer.instance['orders']) if len(optimizer.instance['orders']) > 0 else 0,
                'total_distance_km': optimizer.metrics['total_distance_km'],
                'total_flight_time_hours': optimizer.metrics['total_flight_time_hours'],
                'total_idle_time_hours': optimizer.metrics['total_idle_time_hours'],
                'drone_utilization': optimizer.metrics['drone_utilization'],
                'average_delivery_time': optimizer.metrics['average_delivery_time'],
                'average_assignment_time_ms': optimizer.metrics['average_assignment_time'] * 1000,
                'cross_depot_operations': optimizer.metrics['cross_depot_operations'],
                'gnn_cross_depot_decisions': optimizer.metrics['gnn_cross_depot_decisions'],
                'future_planning_accuracy': optimizer.metrics['future_planning_accuracy'],
                'emergency_completed': optimizer.metrics['emergency_completed'],
                'high_completed': optimizer.metrics['high_completed'],
                'medium_completed': optimizer.metrics['medium_completed'],
                'low_completed': optimizer.metrics['low_completed'],
                'emergency_completion_rate': optimizer.metrics['emergency_completion_rate'],
                'high_completion_rate': optimizer.metrics['high_completion_rate'],
                'medium_completion_rate': optimizer.metrics['medium_completion_rate'],
                'low_completion_rate': optimizer.metrics['low_completion_rate'],
                'model_used': 'Advanced_GNN' if optimizer.model else 'Enhanced_Heuristic',
                'gnn_predictions_made': len(optimizer.gnn_predictions) if optimizer.gnn_predictions else 0
            }
            
            summary_results.append(summary_data)
            print(f"✓ Successfully processed {instance_name}")
            
        except Exception as e:
            print(f"✗ Error processing {instance_file}: {e}")
            continue
    
    # Save summary results
    if summary_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"GNN_results/advanced_gnn_batch_summary_{timestamp}.csv"
        
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = summary_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in summary_results:
                writer.writerow(row)
        
        print(f"\n{'='*80}")
        print(f"ADVANCED GNN BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {len(summary_results)}/{len(instance_files)} instances successfully")
        print(f"Summary results saved to: {summary_file}")
        print(f"Individual metrics saved to: GNN_results/")
        
        # Print performance summary
        if summary_results:
            avg_score_ratio = np.mean([r['score_ratio'] for r in summary_results])
            avg_completion_rate = np.mean([r['completion_rate'] for r in summary_results])
            total_cross_depot_ops = sum([r['cross_depot_operations'] for r in summary_results])
            total_gnn_cross_depot = sum([r['gnn_cross_depot_decisions'] for r in summary_results])
            avg_future_planning = np.mean([r['future_planning_accuracy'] for r in summary_results if r['future_planning_accuracy'] > 0])
            
            print(f"\nADVANCED GNN PERFORMANCE SUMMARY:")
            print(f"  Average Score Ratio: {avg_score_ratio:.3f} ({avg_score_ratio*100:.1f}%)")
            print(f"  Average Completion Rate: {avg_completion_rate:.3f} ({avg_completion_rate*100:.1f}%)")
            print(f"  Total Cross-Depot Operations: {total_cross_depot_ops}")
            print(f"  Total GNN Cross-Depot Decisions: {total_gnn_cross_depot}")
            print(f"  Average Future Planning Accuracy: {avg_future_planning:.3f}")
            
            if total_gnn_cross_depot > 0:
                cross_depot_success_rate = (total_cross_depot_ops / total_gnn_cross_depot) * 100
                print(f"  GNN Cross-Depot Success Rate: {cross_depot_success_rate:.1f}%")

class ImprovedDroneRoutingVisualizer:
    """Enhanced visualizer with advanced GNN metrics"""
    
    def __init__(self, optimizer, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        self.optimizer = optimizer
        self.bounds = bounds
        
        # Setup plot
        self.fig, ((self.ax_main, self.ax_score), (self.ax_completion, self.ax_advanced)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Main plot setup
        (min_lon, max_lon), (min_lat, max_lat) = bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title('Advanced Cross-Depot GNN Drone Routing System')
        self.ax_main.grid(True, alpha=0.3)
        
        # Metrics plots setup
        self.ax_score.set_title('Score Performance')
        self.ax_completion.set_title('Order Completion')
        self.ax_advanced.set_title('Advanced GNN Metrics')
        
        # Color schemes
        self.depot_colors = plt.cm.Set1(np.linspace(0, 1, len(optimizer.depots)))
        self.priority_colors = {
            OrderPriority.LOW: '#90EE90',
            OrderPriority.MEDIUM: '#FFA500',
            OrderPriority.HIGH: '#FF4500',
            OrderPriority.EMERGENCY: '#8B008B'
        }
        
        # Animation data
        self.time_data = []
        self.score_data = []
        self.completion_data = []
        self.cross_depot_data = []
        self.gnn_cross_depot_data = []
        self.future_planning_data = []
    
    def update_visualization(self, frame):
        """Update visualization with advanced GNN metrics"""
        self.ax_main.clear()
        
        # Main plot setup
        (min_lon, max_lon), (min_lat, max_lat) = self.bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        
        current_score = self.optimizer.metrics['total_score']
        max_score = self.optimizer.total_possible_score
        score_pct = (current_score / max_score) * 100 if max_score > 0 else 0
        cross_depot_ops = self.optimizer.metrics['cross_depot_operations']
        gnn_cross_depot = self.optimizer.metrics['gnn_cross_depot_decisions']
        
        model_status = "Advanced GNN" if self.optimizer.model else "Enhanced Heuristic"
        
        self.ax_main.set_title(f'{model_status} - t={self.optimizer.current_time:.1f}min - Score: {current_score}/{max_score} ({score_pct:.1f}%) - Cross-Depot: {cross_depot_ops} - GNN CD: {gnn_cross_depot}')
        self.ax_main.grid(True, alpha=0.3)
        
        # Plot depots with drone counts
        for i, depot in enumerate(self.optimizer.depots):
            drones_at_depot = len([d for d in self.optimizer.drones if d.depot_id == depot.id])
            
            self.ax_main.scatter(depot.location[0], depot.location[1], 
                               c=[self.depot_colors[i]], s=400, marker='s', 
                               label=f'Depot {depot.id}', edgecolors='black', 
                               linewidth=3, alpha=0.9, zorder=10)
            
            self.ax_main.annotate(f'D{depot.id}({drones_at_depot})', depot.location, 
                                xytext=(8, 8), textcoords='offset points',
                                fontweight='bold', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=self.depot_colors[i], alpha=0.7),
                                zorder=11)
        
        # Plot drones with advanced status indicators
        for drone in self.optimizer.drones:
            color = self.depot_colors[drone.depot_id]
            
            status_config = {
                DroneStatus.IDLE: ('o', 80, 0.6, 'Idle'),
                DroneStatus.FLYING_TO_PICKUP: ('^', 120, 1.0, 'To Pickup'),
                DroneStatus.FLYING_TO_DELIVERY: ('>', 120, 1.0, 'To Delivery'),
                DroneStatus.RETURNING_TO_DEPOT: ('v', 100, 0.8, 'Returning'),
                DroneStatus.CHARGING: ('s', 100, 0.4, 'Charging')
            }
            
            marker, size, alpha, status_text = status_config[drone.status]
            
            # Highlight cross-depot operations with special coloring
            edge_color = 'red' if (hasattr(drone, 'target_depot_id') and 
                                  drone.target_depot_id is not None and 
                                  drone.target_depot_id != drone.depot_id) else 'black'
            edge_width = 4 if edge_color == 'red' else 2
            
            self.ax_main.scatter(drone.location[0], drone.location[1], 
                               c=[color], s=size, marker=marker, alpha=alpha, 
                               edgecolors=edge_color, linewidth=edge_width, zorder=8)
            
            # Battery indicator
            battery_pct = (drone.current_battery_seconds / drone.battery_duration_seconds) * 100
            battery_color = 'red' if battery_pct < 20 else 'orange' if battery_pct < 50 else 'green'
            
            self.ax_main.annotate(f'{battery_pct:.0f}%', 
                                (drone.location[0], drone.location[1]), 
                                xytext=(0, -20), textcoords='offset points',
                                ha='center', va='top', fontsize=8,
                                color=battery_color, fontweight='bold')
            
            # Plot planned routes
            if drone.full_route and drone.status != DroneStatus.IDLE and drone.status != DroneStatus.CHARGING:
                complete_route = [drone.location] + drone.full_route
                route_x = [pos[0] for pos in complete_route]
                route_y = [pos[1] for pos in complete_route]
                
                # Different line styles for different operations
                line_style = ':' if edge_color == 'red' else '-'
                line_width = 3 if edge_color == 'red' else 2
                
                self.ax_main.plot(route_x, route_y, color=color, alpha=0.8, 
                                linestyle=line_style, linewidth=line_width, zorder=4)
        
        # Plot orders (simplified for space)
        for order in self.optimizer.pending_orders[:20]:  # Show only first 20 for clarity
            priority_color = self.priority_colors[order.priority]
            
            # Pickup and delivery locations
            self.ax_main.scatter(order.pickup_location[0], order.pickup_location[1], 
                               c='#2E8B57', s=80, marker='o', alpha=0.7, 
                               edgecolors='black', linewidth=1, zorder=5)
            self.ax_main.scatter(order.delivery_location[0], order.delivery_location[1], 
                               c='#DC143C', s=80, marker='s', alpha=0.7, 
                               edgecolors='black', linewidth=1, zorder=5)
            
            # Connection line
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.5, linestyle='--', linewidth=1.5)
        
        # Update metrics plots
        self.time_data.append(self.optimizer.current_time)
        self.score_data.append(self.optimizer.metrics['score_ratio'])
        self.completion_data.append(self.optimizer.metrics['total_orders_completed'])
        self.cross_depot_data.append(self.optimizer.metrics['cross_depot_operations'])
        self.gnn_cross_depot_data.append(self.optimizer.metrics['gnn_cross_depot_decisions'])
        self.future_planning_data.append(self.optimizer.metrics['future_planning_accuracy'])
        
        # Score plot
        self.ax_score.clear()
        self.ax_score.plot(self.time_data, [s * 100 for s in self.score_data], 'g-', linewidth=2, label='Score %')
        self.ax_score.set_ylabel('Score Percentage')
        self.ax_score.set_title('Score Performance')
        self.ax_score.grid(True, alpha=0.3)
        self.ax_score.legend()
        
        # Completion plot
        self.ax_completion.clear()
        self.ax_completion.plot(self.time_data, self.completion_data, 'b-', linewidth=2, label='Completed')
        self.ax_completion.plot(self.time_data, [len(self.optimizer.failed_orders)] * len(self.time_data), 'r-', linewidth=2, label='Failed')
        self.ax_completion.set_ylabel('Order Count')
        self.ax_completion.set_title('Order Completion')
        self.ax_completion.grid(True, alpha=0.3)
        self.ax_completion.legend()
        
        # Advanced GNN metrics plot
        self.ax_advanced.clear()
        if max(self.cross_depot_data) > 0:
            self.ax_advanced.plot(self.time_data, self.cross_depot_data, 'purple', linewidth=2, label='Cross-Depot Ops')
        if max(self.gnn_cross_depot_data) > 0:
            self.ax_advanced.plot(self.time_data, self.gnn_cross_depot_data, 'orange', linestyle='--', linewidth=2, label='GNN Cross-Depot')
        if max([x for x in self.future_planning_data if x > 0] or [0]) > 0:
            self.ax_advanced.plot(self.time_data, [x * 100 for x in self.future_planning_data], 'cyan', linestyle=':', linewidth=2, label='Future Planning %')
        
        self.ax_advanced.set_ylabel('Count / Percentage')
        self.ax_advanced.set_xlabel('Time (minutes)')
        self.ax_advanced.set_title('Advanced GNN Metrics')
        self.ax_advanced.grid(True, alpha=0.3)
        self.ax_advanced.legend()

def main():
    """Main function with advanced GNN options"""
    print("Advanced Cross-Depot GNN Drone Routing System")
    print("=" * 80)
    print("Enhanced GNN with explicit cross-depot optimization and future planning")
    print()
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python advanced_gnn.py [train|sample|batch]")
        print()
        print("Options:")
        print("  train  - Train advanced cross-depot GNN model")
        print("  sample - Run sample instance with trained advanced GNN")
        print("  batch  - Run all instances with trained advanced GNN")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "train":
        print("TRAINING MODE: Training Advanced Cross-Depot GNN")
        print("-" * 50)
        
        # Check if training data exists
        training_files = select_training_instances()
        if not training_files:
            print("No training data found!")
            print("Please run greedy approach first to generate cross-depot training data.")
            return
        
        print(f"Found {len(training_files)} training instances")
        
        # Train advanced model
        model, scaler = train_advanced_gnn_model(epochs=300, batch_size=64, learning_rate=0.001)
        
        if model:
            print("✓ Advanced GNN model training completed successfully!")
            print("Model saved as 'best_advanced_gnn_model.pth'")
        else:
            print("✗ Training failed!")
    
    elif mode == "sample":
        print("SAMPLE MODE: Running sample instance with Advanced GNN")
        print("-" * 50)
        
        # Check if trained model exists
        if not os.path.exists('best_advanced_gnn_model.pth'):
            print("Advanced trained model not found! Please run training first:")
            print("python advanced_gnn.py train")
            return
        
        # Find a sample instance
        sample_instances = [
            "instances/texas_medium_0.pkl",
            "instances/arkansas_medium_0.pkl", 
            "instances/utah_medium_0.pkl",
            "instances/north_carolina_medium_0.pkl"
        ]
        
        sample_file = None
        for instance in sample_instances:
            if os.path.exists(instance):
                sample_file = instance
                break
        
        if sample_file:
            print(f"Running advanced GNN simulation with {sample_file}")
            optimizer = run_advanced_gnn_simulation(sample_file, dt=0.5, show_animation=True)
        else:
            print("No sample instance found!")
            print("Expected instances in 'instances/' folder")
    
    elif mode == "batch":
        print("BATCH MODE: Running all instances with Advanced GNN")
        print("-" * 50)
        
        # Check if trained model exists
        if not os.path.exists('best_advanced_gnn_model.pth'):
            print("Advanced trained model not found! Please run training first:")
            print("python advanced_gnn.py train")
            return
        
        print("Running advanced GNN batch experiments on all instances...")
        run_advanced_gnn_batch_experiments()
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use: train, sample, or batch")

if __name__ == "__main__":
    main()