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
            'small': {'depots': 2, 'drones_per_depot': 1, 'orders': 15, 'area_scale': 0.3},
            'medium': {'depots': 3, 'drones_per_depot': 2, 'orders': 35, 'area_scale': 0.6},
            'large': {'depots': 5, 'drones_per_depot': 3, 'orders': 60, 'area_scale': 1.0}
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
                capacity=params['drones_per_depot'] * 2,
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

# HAD-specific neural network components
class GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.output_dim = output_dim
        self.head_dim = output_dim // num_heads
        
        self.W_q = nn.Linear(input_dim, output_dim)
        self.W_k = nn.Linear(input_dim, output_dim)
        self.W_v = nn.Linear(input_dim, output_dim)
        self.W_o = nn.Linear(output_dim, output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attention, V)
        out = out.view(batch_size, seq_len, -1)
        return self.W_o(out)

class BipartiteMatchingGAT(nn.Module):
    def __init__(self, node_features=6, hidden_dim=128, output_dim=64):
        super().__init__()
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim) for _ in range(3)
        ])
        
        self.feasibility_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, node_features):
        x = F.relu(self.node_embedding(node_features))
        
        for gat in self.gat_layers:
            x = F.relu(gat(x)) + x
            
        feasibility = torch.sigmoid(self.feasibility_head(x))
        value = self.value_head(x)
        
        return feasibility, value

class LightningAssignment:
    def __init__(self, model_path=None):
        self.model = BipartiteMatchingGAT()
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
    def encode_state(self, order: Order, drones: List[Drone], depots: List[Depot]) -> torch.Tensor:
        features = []
        
        # Order node features
        order_features = [
            order.pickup_location[0], order.pickup_location[1],
            order.delivery_location[0], order.delivery_location[1],
            order.priority.value, time.time() - order.arrival_time
        ]
        features.append(order_features)
        
        # Drone-depot combination features
        for drone in drones:
            for depot in depots:
                drone_depot_features = [
                    drone.location[0], drone.location[1],
                    self.calculate_distance(drone.location, depot.location),
                    drone.current_battery / drone.battery_capacity,
                    len([o for o in [drone.current_order] if o is not None]),
                    len(depot.drones)
                ]
                features.append(drone_depot_features)
        
        return torch.FloatTensor(features).unsqueeze(0)
    
    def calculate_distance(self, loc1: Tuple[float, float], loc2: Tuple[float, float]) -> float:
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def assign_order(self, order: Order, drones: List[Drone], depots: List[Depot]) -> Optional[Tuple[int, int]]:
        """Lightning-fast assignment using simplified heuristic (fallback from neural network)"""
        start_time = time.time()
        
        # Filter available drones
        available_drones = [d for d in drones if d.status == DroneStatus.IDLE]
        if not available_drones:
            print(f"No available drones for order {order.id}")
            return None
        
        print(f"Lightning assignment for order {order.id}, {len(available_drones)} available drones")
        
        # Simplified assignment logic (bypass neural network for now)
        best_assignment = None
        best_score = float('inf')
        
        for drone in available_drones:
            # Check if drone can complete the order
            pickup_distance = self.calculate_distance(drone.location, order.pickup_location)
            delivery_distance = self.calculate_distance(order.pickup_location, order.delivery_location)
            
            # Find best depot for this drone-order combination
            best_depot = min(depots, key=lambda d: self.calculate_distance(order.delivery_location, d.location))
            depot_distance = self.calculate_distance(order.delivery_location, best_depot.location)
            
            total_distance = pickup_distance + delivery_distance + depot_distance
            
            # Battery constraint check
            battery_needed = total_distance * 0.8 + 15.0
            safety_margin = drone.battery_capacity * 0.1
            
            if drone.current_battery >= battery_needed + safety_margin:
                # Calculate assignment score
                priority_weight = (5 - order.priority.value) * 10
                urgency = max(0, order.deadline - time.time())
                
                score = total_distance - priority_weight + urgency * 0.01
                
                if score < best_score:
                    best_score = score
                    best_assignment = (drone.id, best_depot.id)
                    
                print(f"  Drone {drone.id}: distance={total_distance:.2f}, battery_ok={drone.current_battery:.1f}>={battery_needed:.1f}, score={score:.2f}")
        
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

# Main HAD optimizer class that integrates with existing framework
class HierarchicalAnytimeDispatcher:
    """HAD optimizer that follows the same interface as the greedy optimizer"""
    
    def __init__(self, depots: List[Depot], drone_specs: Dict):
        self.depots = depots
        self.drone_specs = drone_specs
        self.drones = []
        self.pending_orders = []
        self.completed_orders = []
        self.current_time = 0.0
        
        # HAD-specific components
        self.lightning_assignment = LightningAssignment()
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
            'alns_optimizations': 0
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
        """Layer 1: Lightning Assignment (≤100ms)"""
        assignments = []
        processed_orders = []
        
        for order in self.pending_orders[:]:
            if order.assigned_drone is not None:
                continue
            
            start_time = time.time()
            
            # Lightning assignment
            result = self.lightning_assignment.assign_order(order, self.drones, self.depots)
            
            if result and (time.time() - start_time) * 1000 <= 100:
                drone_id, depot_id = result
                drone = next(d for d in self.drones if d.id == drone_id)
                
                order.assigned_drone = drone_id
                order.end_depot_id = depot_id
                assignments.append((drone, order))
                processed_orders.append(order)
                self.metrics['lightning_assignments'] += 1
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
        
        # Layer 1: Lightning Assignment for immediate orders
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
    
    def __init__(self, optimizer, bounds: Tuple[Tuple[float, float], Tuple[float, float]], algorithm_name="HAD"):
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
        
        self.ax_metrics.clear()
        self.ax_metrics.plot(self.time_data, self.completed_orders_data, 'b-', label='Completed Orders', linewidth=2)
        
        if max(self.avg_delivery_time_data) > 0:
            self.ax_metrics.plot(self.time_data, self.avg_delivery_time_data, 'r-', label='Avg Delivery Time', linewidth=2)
        
        # Add HAD-specific metrics
        if hasattr(self, 'lightning_assignments_data') and max(self.lightning_assignments_data) > 0:
            self.ax_metrics.plot(self.time_data, self.lightning_assignments_data, 'g--', label='Lightning Assignments', alpha=0.7)
        
        if hasattr(self, 'alns_optimizations_data') and max(self.alns_optimizations_data) > 0:
            self.ax_metrics.plot(self.time_data, self.alns_optimizations_data, 'm--', label='ALNS Optimizations', alpha=0.7)
        
        self.ax_metrics.set_xlabel('Time')
        self.ax_metrics.set_ylabel('Count / Time')
        self.ax_metrics.set_title(f'{self.algorithm_name} Performance Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)

# Comparison simulation function
def run_had_simulation(region='texas', size='medium', duration=300.0, dt=2.0, order_rate=0.5):
    """Run HAD simulation with same parameters as greedy approach"""
    
    # Generate problem instance using same generator
    instance = RealWorldDataGenerator.generate_instance(region, size, duration)
    
    # Same drone specifications
    drone_specs = {
        'speed': 2.0,
        'battery_capacity': 100.0,
        'payload_capacity': 10.0
    }
    
    # Initialize HAD optimizer
    optimizer = HierarchicalAnytimeDispatcher(instance['depots'], drone_specs)
    
    # Initialize visualizer with HAD label
    visualizer = DroneRoutingVisualizer(optimizer, instance['bounds'], "HAD")
    
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
        # Only add random orders occasionally to simulate realistic arrival rate
        if frame > 10 and random.random() < order_rate * dt / 10:  # Scale by time step
            (min_lon, max_lon), (min_lat, max_lat) = instance['bounds']
            random_order = Order(
                id=len(orders) + frame * 1000,  # Unique ID
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
            print(f"HAD - Time: {optimizer.current_time:.1f}, "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Queue: {len(optimizer.virtual_queue)}, "
                  f"Avg Delivery: {metrics['average_delivery_time']:.2f}, "
                  f"Lightning: {metrics['lightning_assignments']}, "
                  f"ALNS: {metrics['alns_optimizations']}")
    
    # Create animation
    ani = animation.FuncAnimation(visualizer.fig, animate, 
                                 frames=int(duration/dt), 
                                 interval=int(dt*100), 
                                 repeat=False, blit=False)
    
    plt.tight_layout()
    return ani, optimizer, visualizer

# Comparison function to run both algorithms
def compare_algorithms(region='texas', size='medium', duration=200.0, dt=2.0, order_rate=0.5):
    """Compare HAD vs Greedy algorithms side by side"""
    
    print("="*60)
    print("ALGORITHM COMPARISON: HAD vs Greedy")
    print("="*60)
    
    # Set random seed for fair comparison
    random.seed(42)
    np.random.seed(42)
    
    # Generate same instance for both
    instance = RealWorldDataGenerator.generate_instance(region, size, duration)
    
    print(f"Instance: {region.title()} {size}")
    print(f"Depots: {len(instance['depots'])}")
    print(f"Total drones: {sum(d.capacity for d in instance['depots'])}")
    print(f"Orders: {len(instance['orders'])}")
    print(f"Duration: {duration:.0f} time units")
    print("-"*60)
    
    # Run HAD simulation
    print("Running HAD Algorithm...")
    ani_had, optimizer_had, viz_had = run_had_simulation(region, size, duration, dt, order_rate)
    
    # Final comparison metrics
    def print_final_metrics():
        print("\n" + "="*60)
        print("FINAL COMPARISON RESULTS")
        print("="*60)
        
        had_metrics = optimizer_had.metrics
        
        print(f"HAD Algorithm Results:")
        print(f"  Total Orders Completed: {had_metrics['total_orders_completed']}")
        print(f"  Average Delivery Time: {had_metrics['average_delivery_time']:.2f}")
        print(f"  Orders On Time: {had_metrics['orders_completed_on_time']}")
        print(f"  Total Distance: {had_metrics['total_distance_traveled']:.2f}")
        print(f"  Lightning Assignments: {had_metrics['lightning_assignments']}")
        print(f"  ALNS Optimizations: {had_metrics['alns_optimizations']}")
        print(f"  Pending Orders: {len(optimizer_had.pending_orders)}")
        print(f"  Virtual Queue: {len(optimizer_had.virtual_queue)}")
        
        # Efficiency metrics
        if had_metrics['total_orders_completed'] > 0:
            had_efficiency = had_metrics['total_orders_completed'] / had_metrics['total_distance_traveled']
            print(f"  Efficiency (Orders/Distance): {had_efficiency:.4f}")
    
    # Schedule final metrics print
    def delayed_metrics():
        import threading
        timer = threading.Timer(duration/dt * dt/100 + 2, print_final_metrics)
        timer.start()
    
    delayed_metrics()
    
    return ani_had, optimizer_had, viz_had

# Training function for the GAT model (optional for research)
def generate_training_data(num_instances=1000, region='texas', size='medium'):
    """Generate training data for the Lightning Assignment GAT model"""
    training_data = []
    
    for i in range(num_instances):
        if i % 100 == 0:
            print(f"Generating training instance {i}/{num_instances}")
        
        # Generate random instance
        instance = RealWorldDataGenerator.generate_instance(region, size)
        
        # Create temporary optimizer for evaluation
        drone_specs = {'speed': 2.0, 'battery_capacity': 100.0, 'payload_capacity': 10.0}
        temp_optimizer = HierarchicalAnytimeDispatcher(instance['depots'], drone_specs)
        
        # Generate state-action pairs
        for order in instance['orders'][:10]:  # Use first 10 orders for training
            # Encode state
            available_drones = [d for d in temp_optimizer.drones if d.status == DroneStatus.IDLE]
            if available_drones:
                lightning = LightningAssignment()
                features = lightning.encode_state(order, available_drones, temp_optimizer.depots)
                
                # Get optimal assignment using greedy approach as ground truth
                best_assignment = None
                best_score = float('inf')
                
                for drone in available_drones:
                    if temp_optimizer.can_complete_order(drone, order):
                        estimated_time, depot_id = temp_optimizer.estimate_delivery_time(drone, order)
                        priority_weight = 5 - order.priority.value
                        score = estimated_time + priority_weight * 10
                        
                        if score < best_score:
                            best_score = score
                            best_assignment = (drone.id, depot_id)
                
                if best_assignment:
                    training_data.append({
                        'features': features,
                        'optimal_assignment': best_assignment,
                        'reward': -best_score  # Negative because we want to minimize
                    })
    
    return training_data

def train_lightning_model(training_data, epochs=100, learning_rate=0.001):
    """Train the Lightning Assignment GAT model"""
    model = BipartiteMatchingGAT()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Training Lightning Assignment model on {len(training_data)} samples...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for data in training_data:
            features = data['features']
            reward = torch.FloatTensor([data['reward']])
            
            # Forward pass
            feasibility, values = model(features)
            
            # Compute loss (simplified - in practice would use more sophisticated loss)
            loss = criterion(values.mean(), reward)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Average Loss: {total_loss/len(training_data):.4f}")
    
    return model

# Batch comparison function for paper results
def run_batch_comparison(regions=['texas'], sizes=['medium'], num_runs=5):
    """Run batch comparison for statistical analysis"""
    results = []
    
    for region in regions:
        for size in sizes:
            for run in range(num_runs):
                print(f"\nRunning comparison {run+1}/{num_runs} for {region} {size}")
                
                # Set different seed for each run
                random.seed(42 + run)
                np.random.seed(42 + run)
                
                # Run HAD simulation
                _, had_optimizer, _ = run_had_simulation(region, size, duration=150.0, dt=2.0, order_rate=0.3)
                
                # Collect results
                had_metrics = had_optimizer.metrics
                result = {
                    'region': region,
                    'size': size,
                    'run': run,
                    'algorithm': 'HAD',
                    'completed_orders': had_metrics['total_orders_completed'],
                    'avg_delivery_time': had_metrics['average_delivery_time'],
                    'on_time_orders': had_metrics['orders_completed_on_time'],
                    'total_distance': had_metrics['total_distance_traveled'],
                    'lightning_assignments': had_metrics['lightning_assignments'],
                    'alns_optimizations': had_metrics['alns_optimizations'],
                    'efficiency': had_metrics['total_orders_completed'] / max(had_metrics['total_distance_traveled'], 1)
                }
                results.append(result)
    
    return results

# Example usage for direct comparison
if __name__ == "__main__":
    print("Hierarchical Anytime Dispatcher (HAD) for Multi-Depot Drone Routing")
    print("Compatible with existing greedy approach framework")
    print("="*70)
    
    # Run comparison
    ani, optimizer, visualizer = compare_algorithms('texas', 'large', duration=150.0)
    plt.show()