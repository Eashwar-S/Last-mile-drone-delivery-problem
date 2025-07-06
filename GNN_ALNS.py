import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import time
import random
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor
import heapq
from enum import Enum
import json
from datetime import datetime, timedelta
import pickle  # ADDED: For loading training data
from sklearn.preprocessing import StandardScaler  # ADDED: For feature normalization
from torch.utils.data import Dataset, DataLoader  # ADDED: For training data management

# Import the exact same data structures from greedy approach
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
    completed: bool = False
    pickup_time: Optional[float] = None
    delivery_time: Optional[float] = None
    assigned_drone: Optional[int] = None
    end_depot_id: Optional[int] = None

@dataclass
class Depot:
    id: int
    location: Tuple[float, float]
    capacity: int
    charging_stations: int
    drones: List[int] = field(default_factory=list)

@dataclass
class Drone:
    id: int
    depot_id: int
    location: Tuple[float, float]
    status: DroneStatus
    speed: float
    battery_capacity: float
    current_battery: float
    payload_capacity: float
    current_payload: float = 0.0
    current_order: Optional[Order] = None
    route: List[Tuple[float, float]] = field(default_factory=list)
    total_distance: float = 0.0
    target_depot_id: Optional[int] = None

# Import the exact same data generator
class RealWorldDataGenerator:
    """Generate realistic datasets based on actual geographic regions"""
    
    REGIONS = {
        'arkansas': {
            'bounds': ((-94.6, -89.6), (33.0, 36.5)),
            'cities': [(-92.3, 34.7), (-94.1, 36.1), (-91.2, 35.2)],
        },
        'north_carolina': {
            'bounds': ((-84.3, -75.4), (33.8, 36.6)),
            'cities': [(-80.8, 35.2), (-78.6, 35.8), (-82.6, 35.6)],
        },
        'utah': {
            'bounds': ((-114.1, -109.0), (37.0, 42.0)),
            'cities': [(-111.9, 40.8), (-111.7, 39.3), (-111.5, 37.1)],
        },
        'texas': {
            'bounds': ((-106.6, -93.5), (25.8, 36.5)),
            'cities': [(-97.7, 30.3), (-95.4, 29.8), (-96.8, 32.8)],
        }
    }
    
    @staticmethod
    def generate_instance(region: str, size: str, time_horizon: float = 480.0) -> Dict:
        """Generate problem instance based on region and size"""
        if region not in RealWorldDataGenerator.REGIONS:
            raise ValueError(f"Region {region} not supported")
        
        region_data = RealWorldDataGenerator.REGIONS[region]
        (min_lon, max_lon), (min_lat, max_lat) = region_data['bounds']
        
        size_params = {
            'small': {'depots': 2, 'drones_per_depot': 1, 'orders': 150, 'area_scale': 0.3},
            'medium': {'depots': 3, 'drones_per_depot': 1, 'orders': 350, 'area_scale': 0.6},
            'large': {'depots': 5, 'drones_per_depot': 1, 'orders': 600, 'area_scale': 1.0}
        }
        
        params = size_params[size]
        
        # Generate depots with same logic as greedy approach
        depots = []
        city_clusters = []
        
        if len(region_data['cities']) >= params['depots']:
            for i in range(params['depots']):
                base_lon, base_lat = region_data['cities'][i]
                city_clusters.append((base_lon, base_lat))
        else:
            import math
            region_width = max_lon - min_lon
            region_height = max_lat - min_lat
            grid_size = math.ceil(math.sqrt(params['depots']))
            
            for i in range(params['depots']):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                lon_offset = (grid_x + random.uniform(-0.8, 0.8)) * region_width / grid_size
                lat_offset = (grid_y + random.uniform(-0.8, 0.8)) * region_height / grid_size
                
                depot_lon = min_lon + lon_offset
                depot_lat = min_lat + lat_offset
                
                depot_lon = max(min_lon, min(max_lon, depot_lon))
                depot_lat = max(min_lat, min(max_lat, depot_lat))
                
                city_clusters.append((depot_lon, depot_lat))
        
        for i, (base_lon, base_lat) in enumerate(city_clusters):
            depot_lon = base_lon + random.uniform(-0.3, 0.3)
            depot_lat = base_lat + random.uniform(-0.3, 0.3)
            
            depots.append(Depot(
                id=i,
                location=(depot_lon, depot_lat),
                capacity=params['drones_per_depot'],
                charging_stations=params['drones_per_depot']
            ))
        
        # Generate orders with PROPERLY DISTRIBUTED arrival times
        orders = []
        for i in range(params['orders']):
            pickup_lon = random.uniform(min_lon, max_lon)
            pickup_lat = random.uniform(min_lat, max_lat)
            
            delivery_lon = random.uniform(min_lon, max_lon)
            delivery_lat = random.uniform(min_lat, max_lat)
            
            priority_weights = [0.2, 0.6, 0.15, 0.05]
            priority = random.choices(list(OrderPriority), weights=priority_weights)[0]
            
            # FIXED: Distribute arrival times more evenly across the time horizon
            # Instead of exponential distribution, use uniform distribution for better spread
            arrival_time = random.uniform(0, time_horizon * 0.8)  # Orders arrive in first 80% of time
            
            deadline_multipliers = {
                OrderPriority.EMERGENCY: 0.5,
                OrderPriority.HIGH: 1.0,
                OrderPriority.MEDIUM: 2.0,
                OrderPriority.LOW: 4.0
            }
            
            # More reasonable deadline calculation
            base_service_time = random.uniform(30, 120)  # Base time to complete order
            deadline = arrival_time + base_service_time * deadline_multipliers[priority]
            
            orders.append(Order(
                id=i,
                pickup_location=(pickup_lon, pickup_lat),
                delivery_location=(delivery_lon, delivery_lat),
                priority=priority,
                arrival_time=arrival_time,
                deadline=min(deadline, time_horizon),
                weight=random.uniform(0.5, 5.0)
            ))
        
        # Sort orders by arrival time for proper dynamic release
        orders.sort(key=lambda x: x.arrival_time)
        
        print(f"Generated {len(orders)} orders with arrival times from {orders[0].arrival_time:.1f} to {orders[-1].arrival_time:.1f}")
        
        return {
            'region': region,
            'size': size,
            'depots': depots,
            'orders': orders,
            'bounds': region_data['bounds'],
            'time_horizon': time_horizon
        }

# ADDED: Dataset class for training data management
class DroneRoutingDataset(Dataset):
    """Dataset class for training the GNN model"""
    
    def __init__(self, training_data: List[Dict], feature_scaler=None):
        self.training_data = training_data
        self.feature_scaler = feature_scaler
        
        # Extract features and labels
        self.features = []
        self.labels = []
        
        for sample in training_data:
            state_features = self._extract_state_features(sample['state'])
            action_features = self._extract_action_features(sample['action'])
            
            # Combine state and action features
            combined_features = np.concatenate([state_features, action_features])
            self.features.append(combined_features)
            self.labels.append(sample['reward'])
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Normalize features if scaler provided
        if self.feature_scaler is not None:
            self.features = self.feature_scaler.fit_transform(self.features)
    
    def _extract_state_features(self, state: Dict) -> np.ndarray:
        """Extract numerical features from state representation"""
        features = []
        
        # Order features
        order = state['order']
        features.extend([
            order['pickup_location'][0], order['pickup_location'][1],
            order['delivery_location'][0], order['delivery_location'][1],
            order['priority'], order['urgency'], order['weight']
        ])
        
        # Aggregate drone features
        available_drones = [d for d in state['drones'] if d['available']]
        if available_drones:
            avg_battery = np.mean([d['battery_level'] for d in available_drones])
            min_distance = np.min([d['distance_to_pickup'] for d in available_drones])
            num_available = len(available_drones)
        else:
            avg_battery, min_distance, num_available = 0, float('inf'), 0
        
        features.extend([avg_battery, min_distance, num_available])
        
        # Aggregate depot features
        min_depot_distance = np.min([d['distance_to_delivery'] for d in state['depots']])
        total_depot_capacity = sum(d['capacity'] for d in state['depots'])
        
        features.extend([min_depot_distance, total_depot_capacity])
        
        # System state
        features.append(state['current_time'])
        return np.array(features)
    
    def _extract_action_features(self, action: Dict) -> np.ndarray:
        """Extract numerical features from action representation"""
        return np.array([action['chosen_drone_id'], action['chosen_depot_id']])
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])

# MODIFIED: Enhanced neural network components with better architecture
class GraphAttentionLayer(nn.Module):
    """Improved Graph Attention Layer with better feature processing"""
    
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads
        
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)
        
        # ADDED: Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)  # ADDED: Dropout on attention weights
        
        out = torch.matmul(attention, V)
        out = out.view(batch_size, seq_len, -1)
        out = self.W_o(out)
        
        # ADDED: Residual connection and layer normalization
        return self.layer_norm(out + x) if x.size(-1) == out.size(-1) else self.layer_norm(out)

# MODIFIED: Enhanced BipartiteMatchingGAT with improved architecture
class BipartiteMatchingGAT(nn.Module):
    """Enhanced Graph Attention Network for drone-order matching"""
    
    def __init__(self, input_dim=14, hidden_dim=128, output_dim=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # ADDED: Input feature embedding with normalization
        self.input_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MODIFIED: Multiple GAT layers with residual connections
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout) 
            for _ in range(num_layers)
        ])
        
        # ADDED: Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # MODIFIED: Separate heads for different prediction tasks
        self.feasibility_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # Handle both single samples and batches
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension
        
        batch_size, seq_len, feature_dim = x.shape
        
        # Reshape for batch normalization
        x_reshaped = x.view(-1, feature_dim)
        x_embedded = self.input_embedding(x_reshaped)
        x_embedded = x_embedded.view(batch_size, seq_len, self.hidden_dim)
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x_embedded = gat(x_embedded)
        
        # Feature fusion
        x_fused = self.feature_fusion(x_embedded)
        
        # Prediction heads
        feasibility = self.feasibility_head(x_fused)
        value = self.value_head(x_fused)
        
        return feasibility.squeeze(-1), value.squeeze(-1)

# ADDED: Training utilities and model management
class ModelTrainer:
    """Trainer class for the GNN model with comprehensive training utilities"""
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        # ADDED: Loss tracking
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for features, labels in train_loader:
            self.optimizer.zero_grad()
            
        
            # Forward pass
            feasibility, values = self.model(features)
            
            # Calculate loss (using value prediction for now)
            loss = criterion(values.mean(dim=1) if len(values.shape) > 1 else values, labels.squeeze())
            
            # Backward pass
            loss.backward()
            
            # ADDED: Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                feasibility, values = self.model(features)
                loss = criterion(values.mean(dim=1) if len(values.shape) > 1 else values, labels.squeeze())
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, filepath)
    
    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])

# MODIFIED: Enhanced LightningAssignment with improved training integration
class LightningAssignment:
    """Enhanced Lightning Assignment with trained GNN model"""
    
    def __init__(self, model_path=None, feature_scaler=None):
        self.model = BipartiteMatchingGAT()
        self.feature_scaler = feature_scaler
        self.trainer = ModelTrainer(self.model)
        
        if model_path:
            self.trainer.load_model(model_path)
        self.model.eval()
        
    def encode_state(self, order: Order, drones: List[Drone], depots: List[Depot]) -> torch.Tensor:
        """Enhanced state encoding with better feature engineering"""
        features = []
        
        # Order node features (enhanced)
        order_features = [
            order.pickup_location[0], order.pickup_location[1],
            order.delivery_location[0], order.delivery_location[1],
            order.priority.value, 
            max(0, order.deadline - time.time()),  # Time urgency
            order.weight,
            self.calculate_distance(order.pickup_location, order.delivery_location)  # Order distance
        ]
        
        # System state features
        available_drones = [d for d in drones if d.status == DroneStatus.IDLE]
        system_features = [
            len(available_drones),  # Number of available drones
            len(drones),  # Total drones
            len(depots),  # Number of depots
            time.time()  # Current time
        ]
        
        # Combine features
        combined_features = order_features + system_features
        
        # Add padding if needed to match expected input dimension
        while len(combined_features) < 14:
            combined_features.append(0.0)
        
        features = torch.FloatTensor(combined_features).unsqueeze(0).unsqueeze(0)
        
        # Apply feature scaling if available
        if self.feature_scaler is not None:
            features_np = features.squeeze().numpy().reshape(1, -1)
            features_scaled = self.feature_scaler.transform(features_np)
            features = torch.FloatTensor(features_scaled).unsqueeze(0)
        
        return features
    
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def assign_order(self, order: Order, drones: List[Drone], depots: List[Depot]) -> Optional[Tuple[int, int]]:
        """Enhanced assignment using trained GNN model"""
        start_time = time.time()
        
        # Filter available drones
        available_drones = [d for d in drones if d.status == DroneStatus.IDLE]
        if not available_drones:
            print(f"No available drones for order {order.id}")
            return None
        
        print(f"GNN Lightning assignment for order {order.id}, {len(available_drones)} available drones")
        
        # Use GNN model for assignment scoring
        best_assignment = None
        best_score = float('-inf')
        
        try:
            # Encode current state
            state_features = self.encode_state(order, drones, depots)
            
            # Get model predictions
            with torch.no_grad():
                feasibility, values = self.model(state_features)
                gnn_score = values.item() if len(values.shape) == 0 else values.mean().item()
            
            # Use GNN score to guide assignment
            for drone in available_drones:
                # Check if drone can complete the order
                pickup_distance = self.calculate_distance(drone.location, order.pickup_location)
                delivery_distance = self.calculate_distance(order.pickup_location, order.delivery_location)
                
                # Find best depot for this drone-order combination
                best_depot = min(depots, key=lambda d: self.calculate_distance(order.delivery_location, d.location))
                depot_distance = self.calculate_distance(order.delivery_location, best_depot.location)
                
                total_distance = pickup_distance + delivery_distance + depot_distance
                
                # Battery constraint check (same as before)
                battery_needed = total_distance * 0.8 + 15.0
                safety_margin = drone.battery_capacity * 0.1
                
                if drone.current_battery >= battery_needed + safety_margin:
                    # MODIFIED: Combine GNN score with heuristic features
                    priority_weight = (5 - order.priority.value) * 10
                    urgency = max(0, order.deadline - time.time())
                    
                    # Combine GNN prediction with traditional heuristics
                    score = (gnn_score * 0.7 +  # 70% GNN prediction
                            (priority_weight - total_distance + urgency * 0.01) * 0.3)  # 30% heuristics
                    
                    if score > best_score:
                        best_score = score
                        best_assignment = (drone.id, best_depot.id)
                        
                    print(f"  Drone {drone.id}: GNN_score={gnn_score:.2f}, distance={total_distance:.2f}, "
                          f"combined_score={score:.2f}")
        
        except Exception as e:
            print(f"GNN prediction failed, falling back to heuristic: {e}")
            # Fallback to heuristic assignment
            for drone in available_drones:
                pickup_distance = self.calculate_distance(drone.location, order.pickup_location)
                delivery_distance = self.calculate_distance(order.pickup_location, order.delivery_location)
                
                best_depot = min(depots, key=lambda d: self.calculate_distance(order.delivery_location, d.location))
                depot_distance = self.calculate_distance(order.delivery_location, best_depot.location)
                
                total_distance = pickup_distance + delivery_distance + depot_distance
                battery_needed = total_distance * 0.8 + 15.0
                safety_margin = drone.battery_capacity * 0.1
                
                if drone.current_battery >= battery_needed + safety_margin:
                    priority_weight = (5 - order.priority.value) * 10
                    urgency = max(0, order.deadline - time.time())
                    score = total_distance - priority_weight + urgency * 0.01
                    
                    if best_assignment is None or score < best_score:
                        best_score = score
                        best_assignment = (drone.id, best_depot.id)
        
        # Check time constraint (100ms)
        elapsed_time = (time.time() - start_time) * 1000
        if best_assignment and elapsed_time <= 100:
            print(f"  → Assigned to drone {best_assignment[0]}, depot {best_assignment[1]} (took {elapsed_time:.1f}ms)")
            return best_assignment
        elif best_assignment:
            print(f"  → Assignment found but took too long ({elapsed_time:.1f}ms)")
        else:
            print(f"  → No feasible assignment found")
        
        return None

class ALNSOperator:
    def __init__(self):
        self.destroy_operators = [
            self.priority_aware_shaw_removal,
            self.random_removal,
            self.worst_removal
        ]
        self.repair_operators = [
            self.battery_aware_repair,
            self.greedy_repair
        ]
        
    def priority_aware_shaw_removal(self, orders: List[Order], num_orders: int) -> List[Order]:
        """Remove orders prioritizing low-priority ones"""
        orders_by_priority = sorted(orders, key=lambda x: x.priority.value, reverse=True)
        return orders_by_priority[:min(num_orders, len(orders_by_priority))]
    
    def random_removal(self, orders: List[Order], num_orders: int) -> List[Order]:
        """Randomly remove orders"""
        return random.sample(orders, min(num_orders, len(orders)))
    
    def worst_removal(self, orders: List[Order], num_orders: int) -> List[Order]:
        """Remove orders with worst delivery times"""
        orders_by_delay = sorted(orders, 
                               key=lambda x: (x.deadline - x.arrival_time) if x.delivery_time is None else 
                                           (x.delivery_time - x.arrival_time),
                               reverse=True)
        return orders_by_delay[:min(num_orders, len(orders_by_delay))]
    
    def battery_aware_repair(self, removed_orders: List[Order], optimizer) -> List[Tuple[Drone, Order]]:
        """Insert orders with battery-aware depot optimization"""
        assignments = []
        
        for order in removed_orders:
            best_assignment = None
            best_score = float('inf')
            
            for drone in optimizer.drones:
                if drone.status != DroneStatus.IDLE:
                    continue
                    
                if not optimizer.can_complete_order(drone, order):
                    continue
                
                # Find optimal depot for this assignment
                estimated_time, end_depot_id = optimizer.estimate_delivery_time(drone, order)
                
                # Score based on time and battery efficiency
                score = estimated_time + random.uniform(0, 5)  # Add some randomness
                
                if score < best_score:
                    best_score = score
                    best_assignment = (drone, order, end_depot_id)
            
            if best_assignment:
                drone, order, end_depot_id = best_assignment
                order.assigned_drone = drone.id
                order.end_depot_id = end_depot_id
                assignments.append((drone, order))
        
        return assignments
    
    def greedy_repair(self, removed_orders: List[Order], optimizer) -> List[Tuple[Drone, Order]]:
        """Greedy insertion of removed orders"""
        assignments = []
        
        for order in removed_orders:
            best_assignment = None
            best_time = float('inf')
            
            for drone in optimizer.drones:
                if drone.status != DroneStatus.IDLE:
                    continue
                    
                if not optimizer.can_complete_order(drone, order):
                    continue
                
                estimated_time, end_depot_id = optimizer.estimate_delivery_time(drone, order)
                
                if estimated_time < best_time:
                    best_time = estimated_time
                    best_assignment = (drone, order, end_depot_id)
            
            if best_assignment:
                drone, order, end_depot_id = best_assignment
                order.assigned_drone = drone.id
                order.end_depot_id = end_depot_id
                assignments.append((drone, order))
        
        return assignments

class RollingHorizonALNS:
    def __init__(self, time_horizon: float = 7200):
        self.time_horizon = time_horizon
        self.operators = ALNSOperator()
        self.last_optimization = 0
        
    def optimize_current_plan(self, optimizer) -> List[Tuple[Drone, Order]]:
        """Main ALNS optimization loop"""
        if not optimizer.pending_orders:
            return []
        
        current_time = time.time()
        
        # ALNS parameters
        max_iterations = min(100, len(optimizer.pending_orders) * 5)
        destroy_size = max(1, min(5, len(optimizer.pending_orders) // 3))
        
        best_assignments = []
        best_objective = -float('inf')
        
        # Adaptive weights
        destroy_weights = [1.0] * len(self.operators.destroy_operators)
        repair_weights = [1.0] * len(self.operators.repair_operators)
        
        for iteration in range(max_iterations):
            # Select operators
            destroy_idx = np.random.choice(len(destroy_weights), 
                                         p=np.array(destroy_weights)/sum(destroy_weights))
            repair_idx = np.random.choice(len(repair_weights), 
                                        p=np.array(repair_weights)/sum(repair_weights))
            
            # Destroy phase
            unassigned_orders = [o for o in optimizer.pending_orders if o.assigned_drone is None]
            if not unassigned_orders:
                break
                
            removed_orders = self.operators.destroy_operators[destroy_idx](
                unassigned_orders, destroy_size)
            
            # Repair phase
            assignments = self.operators.repair_operators[repair_idx](
                removed_orders, optimizer)
            
            # Evaluate solution
            objective = self.calculate_objective(assignments, optimizer)
            
            # Accept/reject solution
            if objective > best_objective:
                best_objective = objective
                best_assignments = assignments
                
                # Update operator weights
                destroy_weights[destroy_idx] *= 1.1
                repair_weights[repair_idx] *= 1.1
            
            # Time constraint (max 10 seconds)
            if time.time() - current_time > 10:
                break
        
        return best_assignments
    
    def calculate_objective(self, assignments: List[Tuple[Drone, Order]], optimizer) -> float:
        """Calculate solution objective"""
        if not assignments:
            return 0
        
        total_value = 0
        for drone, order in assignments:
            # Priority weight
            priority_bonus = (5 - order.priority.value) * 10
            
            # Time efficiency
            estimated_time, _ = optimizer.estimate_delivery_time(drone, order)
            time_penalty = estimated_time
            
            # Urgency
            urgency = max(0, order.deadline - optimizer.current_time - estimated_time)
            urgency_bonus = urgency * 0.1
            
            total_value += priority_bonus - time_penalty + urgency_bonus
        
        return total_value

# ADDED: Training and data loading functions
def load_training_data(filename: str) -> Tuple[List[Dict], Dict]:
    """Load training data from pickle file"""
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded training data: {len(data['training_data'])} samples")
        return data['training_data'], data
    except FileNotFoundError:
        print(f"Training data file {filename} not found. Please run greedy simulation first.")
        return [], {}
    except Exception as e:
        print(f"Error loading training data: {e}")
        return [], {}

def train_gnn_model(training_data: List[Dict], epochs=100, batch_size=32, validation_split=0.2):
    """Train the GNN model on collected data"""
    
    if len(training_data) < 10:
        print("Insufficient training data. Please run greedy simulation to collect more data.")
        return None, None
    
    print(f"Training GNN model on {len(training_data)} samples...")
    
    # Create feature scaler
    feature_scaler = StandardScaler()
    
    # Create dataset
    dataset = DroneRoutingDataset(training_data, feature_scaler)
    
    # Split data
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = BipartiteMatchingGAT(input_dim=15)
    trainer = ModelTrainer(model, learning_rate=0.001)
    criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 50
    
    for epoch in range(epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader, criterion)
        
        # Validate
        val_loss = trainer.validate(val_loader, criterion)
        
        # Learning rate scheduling
        trainer.scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            trainer.save_model('best_gnn_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Load best model
    trainer.load_model('best_gnn_model.pth')
    
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return trainer.model, feature_scaler

def plot_training_progress(trainer: ModelTrainer):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(trainer.train_losses, label='Training Loss')
    plt.plot(trainer.val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GNN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main HAD optimizer class that integrates with existing framework
class HierarchicalAnytimeDispatcher:
    """HAD optimizer that follows the same interface as the greedy optimizer"""
    
    def __init__(self, depots: List[Depot], drone_specs: Dict, trained_model_path=None, feature_scaler=None):
        self.depots = depots
        self.drone_specs = drone_specs
        self.drones = []
        self.pending_orders = []
        self.completed_orders = []
        self.current_time = 0.0
        
        # HAD-specific components with trained model
        self.lightning_assignment = LightningAssignment(trained_model_path, feature_scaler)
        self.rolling_horizon_alns = RollingHorizonALNS()
        self.virtual_queue = deque()
        
        # Background optimization
        self.optimization_lock = threading.Lock()
        self.optimization_thread = None
        self.last_optimization = 0
        
        # Initialize drones
        self._initialize_drones()
        
        # Performance metrics
        self.metrics = {
            'total_orders_completed': 0,
            'total_distance_traveled': 0.0,
            'average_delivery_time': 0.0,
            'orders_completed_on_time': 0,
            'total_orders_processed': 0,
            'lightning_assignments': 0,
            'alns_optimizations': 0,
            'gnn_assignments': 0  # ADDED: Track GNN-based assignments
        }
        
        # Start background optimization
        self.start_background_optimization()
    
    def _initialize_drones(self):
        """Initialize drones at depots - same as greedy approach"""
        drone_id = 0
        for depot in self.depots:
            for _ in range(depot.capacity):
                drone = Drone(
                    id=drone_id,
                    depot_id=depot.id,
                    location=depot.location,
                    status=DroneStatus.IDLE,
                    speed=self.drone_specs['speed'],
                    battery_capacity=self.drone_specs['battery_capacity'],
                    current_battery=self.drone_specs['battery_capacity'],
                    payload_capacity=self.drone_specs['payload_capacity']
                )
                self.drones.append(drone)
                depot.drones.append(drone_id)
                drone_id += 1
    
    def add_order(self, order: Order):
        """Add new order to the system"""
        self.pending_orders.append(order)
        self.pending_orders.sort(key=lambda x: (x.priority.value, x.deadline), reverse=True)
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def estimate_delivery_time(self, drone: Drone, order: Order) -> Tuple[float, int]:
        """Estimate total time to complete an order - same as greedy approach"""
        pickup_distance = self.distance(drone.location, order.pickup_location)
        delivery_distance = self.distance(order.pickup_location, order.delivery_location)
        
        nearest_depot = min(self.depots, 
                           key=lambda d: self.distance(order.delivery_location, d.location))
        nearest_depot_distance = self.distance(order.delivery_location, nearest_depot.location)
        
        total_distance = pickup_distance + delivery_distance + nearest_depot_distance
        return total_distance / drone.speed, nearest_depot.id
    
    def can_complete_order(self, drone: Drone, order: Order) -> bool:
        """Check if drone can complete the order - same logic as greedy approach"""
        if drone.status != DroneStatus.IDLE:
            return False
        
        if order.weight > drone.payload_capacity:
            return False
        
        pickup_distance = self.distance(drone.location, order.pickup_location)
        delivery_distance = self.distance(order.pickup_location, order.delivery_location)
        
        nearest_depot = min(self.depots, 
                           key=lambda d: self.distance(order.delivery_location, d.location))
        depot_distance = self.distance(order.delivery_location, nearest_depot.location)
        
        total_distance = pickup_distance + delivery_distance + depot_distance
        
        # Same battery model as greedy approach
        battery_per_distance = 0.8
        takeoff_landing_cost = 5.0
        
        battery_needed = (total_distance * battery_per_distance + 
                         3 * takeoff_landing_cost)
        
        safety_margin = drone.battery_capacity * 0.1
        
        if drone.current_battery < battery_needed + safety_margin:
            return False
        
        estimated_time, _ = self.estimate_delivery_time(drone, order)
        if self.current_time + estimated_time > order.deadline:
            return False
        
        return True
    
    def assign_orders_lightning(self):
        """Layer 1: Lightning Assignment using trained GNN (≤100ms)"""
        assignments = []
        processed_orders = []
        
        for order in self.pending_orders[:]:
            if order.assigned_drone is not None:
                continue
            
            start_time = time.time()
            
            # Lightning assignment with trained GNN
            result = self.lightning_assignment.assign_order(order, self.drones, self.depots)
            
            if result and (time.time() - start_time) * 1000 <= 100:
                drone_id, depot_id = result
                drone = next(d for d in self.drones if d.id == drone_id)
                
                order.assigned_drone = drone_id
                order.end_depot_id = depot_id
                assignments.append((drone, order))
                processed_orders.append(order)
                self.metrics['lightning_assignments'] += 1
                self.metrics['gnn_assignments'] += 1  # ADDED: Track GNN assignments
            else:
                # Add to virtual queue for later processing
                if order not in self.virtual_queue:
                    self.virtual_queue.append(order)
        
        # Remove processed orders
        for order in processed_orders:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
        
        return assignments
    
    def start_background_optimization(self):
        """Start background ALNS optimization thread"""
        def optimization_loop():
            while True:
                time.sleep(30)  # Run every 30 seconds (reduced for demo)
                
                with self.optimization_lock:
                    if self.virtual_queue or len(self.pending_orders) > 5:
                        # Process virtual queue first
                        queue_orders = list(self.virtual_queue)
                        self.virtual_queue.clear()
                        
                        # Add back to pending for ALNS processing
                        for order in queue_orders:
                            if order not in self.pending_orders:
                                self.pending_orders.append(order)
                        
                        # Run ALNS optimization
                        optimized_assignments = self.rolling_horizon_alns.optimize_current_plan(self)
                        
                        if optimized_assignments:
                            self.execute_assignments(optimized_assignments)
                            self.metrics['alns_optimizations'] += 1
                        
                        self.last_optimization = time.time()
        
        self.optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        self.optimization_thread.start()
    
    def update_drone_positions(self, dt: float):
        """Update drone positions and status - same as greedy approach"""
        for drone in self.drones:
            if drone.status == DroneStatus.IDLE or not drone.route:
                continue
            
            if drone.route:
                target = drone.route[0]
                distance_to_target = self.distance(drone.location, target)
                move_distance = drone.speed * dt
                
                if move_distance >= distance_to_target:
                    drone.location = target
                    drone.route.pop(0)
                    drone.total_distance += distance_to_target
                    
                    if drone.status == DroneStatus.FLYING_TO_PICKUP and drone.current_order:
                        if drone.location == drone.current_order.pickup_location:
                            drone.status = DroneStatus.FLYING_TO_DELIVERY
                            drone.route = [drone.current_order.delivery_location]
                            drone.current_order.pickup_time = self.current_time
                    
                    elif drone.status == DroneStatus.FLYING_TO_DELIVERY and drone.current_order:
                        if drone.location == drone.current_order.delivery_location:
                            drone.current_order.delivery_time = self.current_time
                            drone.current_order.completed = True
                            self.completed_orders.append(drone.current_order)
                            
                            if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                                end_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                            else:
                                end_depot = min(self.depots, 
                                               key=lambda d: self.distance(drone.location, d.location))
                            
                            drone.route = [end_depot.location]
                            drone.status = DroneStatus.RETURNING_TO_DEPOT
                            drone.target_depot_id = end_depot.id
                            drone.current_order = None
                    
                    elif drone.status == DroneStatus.RETURNING_TO_DEPOT:
                        drone.status = DroneStatus.IDLE
                        drone.current_battery = drone.battery_capacity
                        
                        if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                            old_depot = next(d for d in self.depots if d.id == drone.depot_id)
                            if drone.id in old_depot.drones:
                                old_depot.drones.remove(drone.id)
                            
                            new_depot = next(d for d in self.depots if d.id == drone.target_depot_id)
                            if drone.id not in new_depot.drones:
                                new_depot.drones.append(drone.id)
                            
                            drone.depot_id = drone.target_depot_id
                            drone.target_depot_id = None
                
                else:
                    # Move towards target
                    direction = ((target[0] - drone.location[0]) / distance_to_target,
                               (target[1] - drone.location[1]) / distance_to_target)
                    
                    new_x = drone.location[0] + direction[0] * move_distance
                    new_y = drone.location[1] + direction[1] * move_distance
                    drone.location = (new_x, new_y)
                    drone.total_distance += move_distance
    
    def execute_assignments(self, assignments: List[Tuple[Drone, Order]]):
        """Execute drone-order assignments"""
        for drone, order in assignments:
            drone.current_order = order
            drone.status = DroneStatus.FLYING_TO_PICKUP
            drone.route = [order.pickup_location]
    
    def step(self, dt: float, new_orders: List[Order] = None):
        """Execute one simulation step"""
        self.current_time += dt
        
        # Add new orders
        if new_orders:
            for order in new_orders:
                self.add_order(order)
        
        # Update drone positions
        self.update_drone_positions(dt)
        
        # Layer 1: Lightning Assignment for immediate orders using trained GNN
        lightning_assignments = self.assign_orders_lightning()
        self.execute_assignments(lightning_assignments)
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update performance metrics"""
        if self.completed_orders:
            self.metrics['total_orders_completed'] = len(self.completed_orders)
            self.metrics['total_distance_traveled'] = sum(drone.total_distance for drone in self.drones)
            
            delivery_times = [order.delivery_time - order.arrival_time 
                            for order in self.completed_orders if order.delivery_time]
            
            if delivery_times:
                self.metrics['average_delivery_time'] = np.mean(delivery_times)
            
            on_time_orders = sum(1 for order in self.completed_orders 
                               if order.delivery_time and order.delivery_time <= order.deadline)
            self.metrics['orders_completed_on_time'] = on_time_orders
            
            self.metrics['total_orders_processed'] = len(self.completed_orders) + len(self.pending_orders)

# Use the exact same visualization class from greedy approach
class DroneRoutingVisualizer:
    """Real-time visualization of the drone routing system"""
    
    def __init__(self, optimizer, bounds: Tuple[Tuple[float, float], Tuple[float, float]], algorithm_name="Trained HAD"):
        self.optimizer = optimizer
        self.bounds = bounds
        self.algorithm_name = algorithm_name
        
        # Setup plot
        self.fig, (self.ax_main, self.ax_metrics) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Main plot setup
        (min_lon, max_lon), (min_lat, max_lat) = bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title(f'{algorithm_name} Multi-Depot Drone Routing System')
        self.ax_main.grid(True, alpha=0.3)
        
        # Metrics plot setup
        self.ax_metrics.set_title(f'{algorithm_name} System Metrics')
        
        # Enhanced color schemes and visualization elements
        self.depot_colors = plt.cm.Set1(np.linspace(0, 1, len(optimizer.depots)))
        self.priority_colors = {
            OrderPriority.LOW: '#90EE90',        # Light Green
            OrderPriority.MEDIUM: '#FFA500',     # Orange
            OrderPriority.HIGH: '#FF4500',       # Red Orange
            OrderPriority.EMERGENCY: '#8B008B'   # Dark Magenta
        }
        
        # Pickup and delivery visualization colors
        self.pickup_color = '#2E8B57'     # Sea Green
        self.delivery_color = '#DC143C'   # Crimson
        
        # Animation data
        self.time_data = []
        self.completed_orders_data = []
        self.avg_delivery_time_data = []
        self.lightning_assignments_data = []
        self.alns_optimizations_data = []
        self.gnn_assignments_data = []  # ADDED: Track GNN assignments
        
    def update_visualization(self, frame):
        """Update visualization for animation"""
        self.ax_main.clear()
        
        # Plot setup
        (min_lon, max_lon), (min_lat, max_lat) = self.bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title(f'{self.algorithm_name} Multi-Depot Drone Routing System (t={self.optimizer.current_time:.1f})')
        self.ax_main.grid(True, alpha=0.3)
        
        # Plot depots with enhanced styling
        for i, depot in enumerate(self.optimizer.depots):
            self.ax_main.scatter(depot.location[0], depot.location[1], 
                               c=[self.depot_colors[i]], s=300, marker='s', 
                               label=f'Depot {depot.id}', edgecolors='black', 
                               linewidth=3, alpha=0.9, zorder=10)
            
            # Enhanced depot labels
            self.ax_main.annotate(f'Depot {depot.id}', depot.location, 
                                xytext=(8, 8), textcoords='offset points',
                                fontweight='bold', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=self.depot_colors[i], alpha=0.7),
                                zorder=11)
        
        # Plot drones
        for drone in self.optimizer.drones:
            color = self.depot_colors[drone.depot_id]
            marker = 'o' if drone.status == DroneStatus.IDLE else '^'
            alpha = 0.5 if drone.status == DroneStatus.IDLE else 1.0
            
            self.ax_main.scatter(drone.location[0], drone.location[1], 
                               c=[color], s=100, marker=marker, alpha=alpha, edgecolors='black')
            
            # Plot drone routes
            if drone.route:
                route_x = [drone.location[0]] + [pos[0] for pos in drone.route]
                route_y = [drone.location[1]] + [pos[1] for pos in drone.route]
                self.ax_main.plot(route_x, route_y, color=color, alpha=0.6, linestyle='--')
        
        # Plot pending orders with enhanced P/D visualization
        for order in self.optimizer.pending_orders:
            priority_color = self.priority_colors[order.priority]
            
            # Enhanced Pickup location with 'P' marker
            self.ax_main.scatter(order.pickup_location[0], order.pickup_location[1], 
                               c=self.pickup_color, s=120, marker='o', alpha=0.8, 
                               edgecolors='black', linewidth=2, zorder=5)
            
            # Add 'P' text on pickup location
            self.ax_main.annotate('P', order.pickup_location, 
                                ha='center', va='center', fontweight='bold', 
                                fontsize=9, color='white', zorder=6)
            
            # Enhanced Delivery location with 'D' marker  
            self.ax_main.scatter(order.delivery_location[0], order.delivery_location[1], 
                               c=self.delivery_color, s=120, marker='s', alpha=0.8, 
                               edgecolors='black', linewidth=2, zorder=5)
            
            # Add 'D' text on delivery location
            self.ax_main.annotate('D', order.delivery_location, 
                                ha='center', va='center', fontweight='bold', 
                                fontsize=9, color='white', zorder=6)
            
            # Connection line with priority color
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.5, linestyle='--', linewidth=2,
                            zorder=2)
        
        # Plot completed orders with enhanced visualization
        for order in self.optimizer.completed_orders[-20:]:  # Show last 20 completed
            priority_color = self.priority_colors[order.priority]
            
            # Faded pickup and delivery locations
            self.ax_main.scatter(order.pickup_location[0], order.pickup_location[1], 
                               c=self.pickup_color, s=60, marker='o', alpha=0.3, 
                               edgecolors='gray', linewidth=1, zorder=1)
            
            self.ax_main.scatter(order.delivery_location[0], order.delivery_location[1], 
                               c=self.delivery_color, s=60, marker='s', alpha=0.3, 
                               edgecolors='gray', linewidth=1, zorder=1)
            
            # Completed route line
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.2, linewidth=3, zorder=1)
            
            # Add checkmark for completed orders
            mid_x = (order.pickup_location[0] + order.delivery_location[0]) / 2
            mid_y = (order.pickup_location[1] + order.delivery_location[1]) / 2
            self.ax_main.annotate('✓', (mid_x, mid_y), 
                                ha='center', va='center', fontweight='bold', 
                                fontsize=10, color='green', alpha=0.5, zorder=3)
        
        # Enhanced legend with HAD-specific elements
        legend_elements = []
        
        # Algorithm-specific header
        legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='gray', 
                                        label=f'🤖 {self.algorithm_name} Algorithm', linewidth=0))
        
        # Depot legend
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                        markerfacecolor='gray', markersize=12, 
                                        label='📍 Depots', markeredgecolor='black'))
        
        # Order status legend
        legend_elements.extend([
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.pickup_color, markersize=10, 
                      label='P - Pickup Location', markeredgecolor='black'),
            plt.Line2D([0], [0], marker='s', color='w', 
                      markerfacecolor=self.delivery_color, markersize=10, 
                      label='D - Delivery Location', markeredgecolor='black'),
        ])
        
        # Priority legend
        legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='gray', 
                                        label='📦 Order Priority:', linewidth=0))
        for priority, color in self.priority_colors.items():
            symbol = "🚨" if priority == OrderPriority.EMERGENCY else "⚠️" if priority == OrderPriority.HIGH else "📦"
            legend_elements.append(plt.Line2D([0], [0], linestyle='-', color=color, 
                                            linewidth=4, label=f'  {symbol} {priority.name}'))
        
        # Drone status legend
        legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='gray', 
                                        label='🚁 Drone Status:', linewidth=0))
        drone_status_legend = [
            ('⭕ Idle', 'o', 0.6),
            ('▲ Active', '^', 1.0),
        ]
        
        for label, marker, alpha in drone_status_legend:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='gray', markersize=8, 
                                            label=f'  {label}', alpha=alpha))
        
        legend_elements.append(plt.Line2D([0], [0], linestyle='--', color='gray', 
                                        linewidth=2, label='  ➤ Flight Route'))
        
        self.ax_main.legend(handles=legend_elements, loc='upper right', 
                           bbox_to_anchor=(1.0, 1.0), fontsize=8, 
                           frameon=True, fancybox=True, shadow=True)
        
        # Update metrics plot with HAD-specific metrics
        self.time_data.append(self.optimizer.current_time)
        self.completed_orders_data.append(self.optimizer.metrics['total_orders_completed'])
        self.avg_delivery_time_data.append(self.optimizer.metrics['average_delivery_time'])
        
        # HAD-specific metrics
        if hasattr(self.optimizer, 'metrics'):
            self.lightning_assignments_data.append(self.optimizer.metrics.get('lightning_assignments', 0))
            self.alns_optimizations_data.append(self.optimizer.metrics.get('alns_optimizations', 0))
            self.gnn_assignments_data.append(self.optimizer.metrics.get('gnn_assignments', 0))  # ADDED
        
        self.ax_metrics.clear()
        self.ax_metrics.plot(self.time_data, self.completed_orders_data, 'b-', label='Completed Orders', linewidth=2)
        
        if max(self.avg_delivery_time_data) > 0:
            self.ax_metrics.plot(self.time_data, self.avg_delivery_time_data, 'r-', label='Avg Delivery Time', linewidth=2)
        
        # Add HAD-specific metrics
        if hasattr(self, 'lightning_assignments_data') and max(self.lightning_assignments_data) > 0:
            self.ax_metrics.plot(self.time_data, self.lightning_assignments_data, 'g--', label='Lightning Assignments', alpha=0.7)
        
        if hasattr(self, 'gnn_assignments_data') and max(self.gnn_assignments_data) > 0:
            self.ax_metrics.plot(self.time_data, self.gnn_assignments_data, 'purple', linestyle=':', label='GNN Assignments', alpha=0.8)
        
        if hasattr(self, 'alns_optimizations_data') and max(self.alns_optimizations_data) > 0:
            self.ax_metrics.plot(self.time_data, self.alns_optimizations_data, 'm--', label='ALNS Optimizations', alpha=0.7)
        
        self.ax_metrics.set_xlabel('Time')
        self.ax_metrics.set_ylabel('Count / Time')
        self.ax_metrics.set_title(f'{self.algorithm_name} Performance Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)

# MODIFIED: Simulation function with trained model
def run_had_simulation_with_trained_model(region='texas', size='medium', duration=300.0, dt=2.0, order_rate=0.5, 
                                          trained_model_path='best_gnn_model.pth', training_data_path='greedy_training_data.pkl'):
    """Run HAD simulation with trained GNN model"""
    
    # Load training data to get feature scaler
    training_data, data_info = load_training_data(training_data_path)
    feature_scaler = None
    
    if training_data:
        # Create feature scaler from training data
        try:
            temp_dataset = DroneRoutingDataset(training_data[:100])  # Use subset for scaler
            feature_scaler = StandardScaler()
            feature_scaler.fit(temp_dataset.features)
            print("Feature scaler created from training data")
        except Exception as e:
            print(f"Could not create feature scaler: {e}")
    
    # Generate problem instance using same generator
    instance = RealWorldDataGenerator.generate_instance(region, size, duration)
    
    # Same drone specifications
    drone_specs = {
        'speed': 2.0,
        'battery_capacity': 100.0,
        'payload_capacity': 10.0
    }
    
    # Initialize HAD optimizer with trained model
    try:
        optimizer = HierarchicalAnytimeDispatcher(instance['depots'], drone_specs, 
                                                 trained_model_path, feature_scaler)
        algorithm_name = "Trained HAD-GNN"
        print("Using trained GNN model for assignments")
    except Exception as e:
        print(f"Could not load trained model: {e}")
        print("Falling back to untrained HAD")
        optimizer = HierarchicalAnytimeDispatcher(instance['depots'], drone_specs)
        algorithm_name = "HAD (no training)"
    
    # Initialize visualizer with HAD label
    visualizer = DroneRoutingVisualizer(optimizer, instance['bounds'], algorithm_name)
    
    # Same order preparation
    orders = sorted(instance['orders'], key=lambda x: x.arrival_time)
    order_index = 0
    
    def animate(frame):
        nonlocal order_index
        
        # Dynamic order arrival - only add orders whose arrival time has passed
        new_orders = []
        while (order_index < len(orders) and 
               orders[order_index].arrival_time <= optimizer.current_time):
            new_orders.append(orders[order_index])
            order_index += 1
        
        # Additional random orders based on Poisson arrival process
        if frame > 10 and random.random() < order_rate * dt / 10:
            (min_lon, max_lon), (min_lat, max_lat) = instance['bounds']
            random_order = Order(
                id=len(orders) + frame * 1000,
                pickup_location=(random.uniform(min_lon, max_lon), 
                               random.uniform(min_lat, max_lat)),
                delivery_location=(random.uniform(min_lon, max_lon), 
                                 random.uniform(min_lat, max_lat)),
                priority=random.choice(list(OrderPriority)),
                arrival_time=optimizer.current_time,
                deadline=optimizer.current_time + random.uniform(60, 300),
                weight=random.uniform(0.5, 5.0)
            )
            new_orders.append(random_order)
        
        # Print debug info for order arrivals
        if new_orders:
            print(f"Time {optimizer.current_time:.1f}: {len(new_orders)} new orders arrived")
        
        # Step simulation
        optimizer.step(dt, new_orders)
        
        # Update visualization
        visualizer.update_visualization(frame)
        
        # Print metrics with HAD-specific information
        if frame % 15 == 0:  # Reduced frequency
            metrics = optimizer.metrics
            print(f"{algorithm_name} - Time: {optimizer.current_time:.1f}, "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Queue: {len(optimizer.virtual_queue)}, "
                  f"Avg Delivery: {metrics['average_delivery_time']:.2f}, "
                  f"Lightning: {metrics['lightning_assignments']}, "
                  f"GNN: {metrics['gnn_assignments']}, "
                  f"ALNS: {metrics['alns_optimizations']}")
    
    # Create animation
    ani = animation.FuncAnimation(visualizer.fig, animate, 
                                 frames=int(duration/dt), 
                                 interval=int(dt*100), 
                                 repeat=False, blit=False)
    
    plt.tight_layout()
    return ani, optimizer, visualizer

# ADDED: Main training and simulation pipeline
def main_training_pipeline():
    """Complete pipeline: Train GNN model and run simulation with trained model"""
    
    print("="*70)
    print("GNN TRAINING AND SIMULATION PIPELINE")
    print("="*70)
    
    # Step 1: Check if training data exists
    training_data_file = "greedy_training_data.pkl"
    training_data, data_info = load_training_data(training_data_file)
    print(f"Training data - {training_data_file}: {len(training_data)} samples found")
    
    if not training_data:
        print("No training data found. Please run greedy_approach_improved.py first to collect training data.")
        print("Example: python greedy_approach_improved.py")
        return
    
    # Step 2: Train GNN model
    print(f"\nStep 1: Training GNN model on {len(training_data)} samples...")
    trained_model, feature_scaler = train_gnn_model(training_data, epochs=50, batch_size=16)
    
    if trained_model is None:
        print("Training failed. Running simulation without trained model.")
        return
    
    # Step 3: Save trained model
    model_trainer = ModelTrainer(trained_model)
    model_trainer.save_model('trained_gnn_model.pth')
    print("Trained model saved as 'trained_gnn_model.pth'")
    
    # Step 4: Run simulation with trained model
    print("\nStep 2: Running HAD simulation with trained GNN model...")
    ani, optimizer, visualizer = run_had_simulation_with_trained_model(
        'texas', 'large', duration=150.0, dt=2.0, order_rate=0.3,
        trained_model_path='trained_gnn_model.pth',
        training_data_path=training_data_file
    )
    
    # Step 5: Show results
    def print_final_results():
        print("\n" + "="*70)
        print("TRAINED HAD-GNN SIMULATION RESULTS")
        print("="*70)
        
        metrics = optimizer.metrics
        print(f"Total Orders Completed: {metrics['total_orders_completed']}")
        print(f"Average Delivery Time: {metrics['average_delivery_time']:.2f}")
        print(f"Orders On Time: {metrics['orders_completed_on_time']}")
        print(f"Total Distance: {metrics['total_distance_traveled']:.2f}")
        print(f"Lightning Assignments: {metrics['lightning_assignments']}")
        print(f"GNN Assignments: {metrics['gnn_assignments']}")
        print(f"ALNS Optimizations: {metrics['alns_optimizations']}")
        
        if metrics['total_orders_completed'] > 0:
            efficiency = metrics['total_orders_completed'] / metrics['total_distance_traveled']
            print(f"Efficiency (Orders/Distance): {efficiency:.4f}")
    
    # Schedule results printing
    import threading
    timer = threading.Timer(160.0, print_final_results)
    timer.start()
    
    plt.show()
    return ani, optimizer, visualizer

# Example usage for training and comparison
if __name__ == "__main__":
    print("Hierarchical Anytime Dispatcher (HAD) with Trained GNN for Multi-Depot Drone Routing")
    print("This version trains a GNN model on data collected from the greedy approach")
    print("="*80)
    
    # Check command line arguments for different modes
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "train":
            # Training mode: Load data and train model
            print("Training mode: Training GNN model on collected data...")
            training_data, _ = load_training_data("greedy_training_data.pkl")
            if training_data:
                model, scaler = train_gnn_model(training_data, epochs=100)
                if model:
                    trainer = ModelTrainer(model)
                    trainer.save_model('trained_gnn_model.pth')
                    print("Model trained and saved successfully!")
            else:
                print("No training data found. Run greedy simulation first.")
        
        elif mode == "simulate":
            # Simulation mode: Run with trained model
            print("Simulation mode: Running HAD with trained GNN...")
            ani, optimizer, visualizer = run_had_simulation_with_trained_model('texas', 'large', duration=200.0)
            plt.show()
        
        elif mode == "pipeline":
            # Full pipeline mode
            main_training_pipeline()
        
        else:
            print("Unknown mode. Use: python GNN_ALNS_improved.py [train|simulate|pipeline]")
    
    else:
        # Default: Run full pipeline
        main_training_pipeline()
