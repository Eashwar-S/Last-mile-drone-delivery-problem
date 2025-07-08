import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
import time
from collections import defaultdict
import heapq
import json
from datetime import datetime, timedelta
import pickle  # ADDED: For data collection
import os
import pickle
import random

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
    end_depot_id: Optional[int] = None  # Depot where delivery trip ends

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
    speed: float  # units per time
    battery_capacity: float
    current_battery: float
    payload_capacity: float
    current_payload: float = 0.0
    current_order: Optional[Order] = None
    route: List[Tuple[float, float]] = field(default_factory=list)
    total_distance: float = 0.0
    target_depot_id: Optional[int] = None  # Depot the drone is heading to
    # ADDED: For improved visualization
    full_route: List[Tuple[float, float]] = field(default_factory=list)  # Complete planned route
    route_progress: float = 0.0  # Progress along current segment (0-1)

class RealWorldDataGenerator:
    """Generate realistic datasets based on actual geographic regions"""
    
    REGIONS = {
        'arkansas': {
            'bounds': ((-94.6, -89.6), (33.0, 36.5)),
            'cities': [(-92.3, 34.7), (-94.1, 36.1), (-91.2, 35.2)],  # Little Rock, Fayetteville, Jonesboro
        },
        'north_carolina': {
            'bounds': ((-84.3, -75.4), (33.8, 36.6)),
            'cities': [(-80.8, 35.2), (-78.6, 35.8), (-82.6, 35.6)],  # Charlotte, Raleigh, Asheville
        },
        'utah': {
            'bounds': ((-114.1, -109.0), (37.0, 42.0)),
            'cities': [(-111.9, 40.8), (-111.7, 39.3), (-111.5, 37.1)],  # Salt Lake City, Provo, St. George
        },
        'texas': {
            'bounds': ((-106.6, -93.5), (25.8, 36.5)),
            'cities': [(-97.7, 30.3), (-95.4, 29.8), (-96.8, 32.8)],  # Austin, Houston, Dallas
        }
    }
    
    @staticmethod
    def generate_instance(region: str, size: str, time_horizon: float = 480.0) -> Dict:
        """Generate problem instance based on region and size"""
        if region not in RealWorldDataGenerator.REGIONS:
            raise ValueError(f"Region {region} not supported")
        
        region_data = RealWorldDataGenerator.REGIONS[region]
        (min_lon, max_lon), (min_lat, max_lat) = region_data['bounds']
        
        # Scale factors based on instance size
        size_params = {
            'small': {'depots': 5, 'drones_per_depot': 1, 'orders': 150, 'area_scale': 0.3},
            'medium': {'depots': 10, 'drones_per_depot': 1, 'orders': 350, 'area_scale': 0.6},
            'large': {'depots': 15, 'drones_per_depot': 1, 'orders': 600, 'area_scale': 1.0}
        }
        
        params = size_params[size]
        
        # Generate depots near major cities with better geographic distribution
        depots = []
        city_clusters = []
        
        # Create clusters of cities for better depot distribution
        if len(region_data['cities']) >= params['depots']:
            # Use actual city locations with some spacing
            for i in range(params['depots']):
                base_lon, base_lat = region_data['cities'][i]
                city_clusters.append((base_lon, base_lat))
        else:
            # Create well-distributed points across the region
            region_width = max_lon - min_lon
            region_height = max_lat - min_lat
            
            # Create a grid-like distribution
            import math
            grid_size = math.ceil(math.sqrt(params['depots']))
            
            for i in range(params['depots']):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                # Add some randomness to avoid perfect grid
                lon_offset = (grid_x + random.uniform(-0.3, 0.3)) * region_width / grid_size
                lat_offset = (grid_y + random.uniform(-0.3, 0.3)) * region_height / grid_size
                
                depot_lon = min_lon + lon_offset
                depot_lat = min_lat + lat_offset
                
                # Ensure within bounds
                depot_lon = max(min_lon, min(max_lon, depot_lon))
                depot_lat = max(min_lat, min(max_lat, depot_lat))
                
                city_clusters.append((depot_lon, depot_lat))
        
        # Create depots at the distributed locations
        for i, (base_lon, base_lat) in enumerate(city_clusters):
            # Add small random variation
            depot_lon = base_lon + random.uniform(-0.1, 0.6)
            depot_lat = base_lat + random.uniform(-0.1, 0.6)
            
            depots.append(Depot(
                id=i,
                location=(depot_lon, depot_lat),
                capacity=params['drones_per_depot'] * 2,
                charging_stations=params['drones_per_depot']
            ))
        
        # Generate orders with realistic distribution
        orders = []
        for i in range(params['orders']):
            # Orders tend to cluster around populated areas
            pickup_lon = random.uniform(min_lon, max_lon)
            pickup_lat = random.uniform(min_lat, max_lat)
            
            # Delivery locations can be anywhere but biased towards cities
            delivery_lon = random.uniform(min_lon, max_lon)
            delivery_lat = random.uniform(min_lat, max_lat)
            
            # Priority distribution (most orders are medium priority)
            priority_weights = [0.2, 0.6, 0.15, 0.05]  # LOW, MEDIUM, HIGH, EMERGENCY
            priority = random.choices(list(OrderPriority), weights=priority_weights)[0]
            
            # Arrival time (orders arrive throughout the day)
            arrival_time = random.expovariate(time_horizon / params['orders'])
            
            # Deadline based on priority
            deadline_multipliers = {
                OrderPriority.EMERGENCY: 0.5,
                OrderPriority.HIGH: 1.0,
                OrderPriority.MEDIUM: 2.0,
                OrderPriority.LOW: 4.0
            }
            
            deadline = arrival_time + random.uniform(60, 240) * deadline_multipliers[priority]
            
            orders.append(Order(
                id=i,
                pickup_location=(pickup_lon, pickup_lat),
                delivery_location=(delivery_lon, delivery_lat),
                priority=priority,
                arrival_time=arrival_time,
                deadline=min(deadline, time_horizon),
                weight=random.uniform(0.5, 5.0)
            ))
        
        return {
            'region': region,
            'size': size,
            'depots': depots,
            'orders': orders,
            'bounds': region_data['bounds'],
            'time_horizon': time_horizon
        }

class MultiDepotDroneRoutingOptimizer:
    """Advanced optimizer for multi-depot drone routing with dynamic orders"""
    
    def __init__(self, depots: List[Depot], drone_specs: Dict):
        self.depots = depots
        self.drone_specs = drone_specs
        self.drones = []
        self.pending_orders = []
        self.completed_orders = []
        self.current_time = 0.0
        
        # ADDED: Data collection for GNN training
        self.training_data = []
        self.assignment_history = []
        
        # Initialize drones
        self._initialize_drones()
        
        # Performance metrics
        self.metrics = {
            'total_orders_completed': 0,
            'total_distance_traveled': 0.0,
            'average_delivery_time': 0.0,
            'orders_completed_on_time': 0,
            'total_orders_processed': 0
        }
    
    def _initialize_drones(self):
        """Initialize drones at depots"""
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
        """Estimate total time to complete an order and return best depot ID"""
        pickup_distance = self.distance(drone.location, order.pickup_location)
        delivery_distance = self.distance(order.pickup_location, order.delivery_location)
        
        # Find nearest depot for ending trip (can be different from start depot)
        nearest_depot = min(self.depots, 
                           key=lambda d: self.distance(order.delivery_location, d.location))
        nearest_depot_distance = self.distance(order.delivery_location, nearest_depot.location)
        
        total_distance = pickup_distance + delivery_distance + nearest_depot_distance
        return total_distance / drone.speed, nearest_depot.id
    
    def can_complete_order(self, drone: Drone, order: Order) -> bool:
        """Check if drone can complete the order within battery constraints"""
        if drone.status != DroneStatus.IDLE:
            return False
        
        if order.weight > drone.payload_capacity:
            return False
        
        # Calculate total distance required
        pickup_distance = self.distance(drone.location, order.pickup_location)
        delivery_distance = self.distance(order.pickup_location, order.delivery_location)
        
        # Find nearest depot for ending trip (optimized depot selection)
        nearest_depot = min(self.depots, 
                           key=lambda d: self.distance(order.delivery_location, d.location))
        depot_distance = self.distance(order.delivery_location, nearest_depot.location)
        
        total_distance = pickup_distance + delivery_distance + depot_distance
        
        # Battery consumption model (units per distance + takeoff/landing costs)
        battery_per_distance = 0.8  # Base consumption per distance unit
        takeoff_landing_cost = 5.0  # Fixed cost for takeoff/landing operations
        
        # Total battery needed: travel + 3 takeoff/landing operations (pickup, delivery, depot)
        battery_needed = (total_distance * battery_per_distance + 
                         3 * takeoff_landing_cost)
        
        # Add safety margin (10% of battery capacity)
        safety_margin = drone.battery_capacity * 0.1
        
        # Check battery constraint
        if drone.current_battery < battery_needed + safety_margin:
            return False
        
        # Check time constraint
        estimated_time, _ = self.estimate_delivery_time(drone, order)
        if self.current_time + estimated_time > order.deadline:
            return False
        
        return True
    
    # ADDED: Data collection method for GNN training
    def collect_training_data(self, order: Order, available_drones: List[Drone], chosen_drone: Drone, chosen_depot: int):
        """Collect training data for GNN model"""
        # Create state representation
        state_features = {
            'order': {
                'pickup_location': order.pickup_location,
                'delivery_location': order.delivery_location,
                'priority': order.priority.value,
                'arrival_time': order.arrival_time,
                'deadline': order.deadline,
                'weight': order.weight,
                'urgency': max(0, order.deadline - self.current_time)
            },
            'drones': [],
            'depots': [],
            'current_time': self.current_time
        }
        
        # Add drone features
        for drone in self.drones:
            drone_features = {
                'id': drone.id,
                'depot_id': drone.depot_id,
                'location': drone.location,
                'status': drone.status.value,
                'battery_level': drone.current_battery / drone.battery_capacity,
                'available': drone.status == DroneStatus.IDLE,
                'distance_to_pickup': self.distance(drone.location, order.pickup_location),
                'can_complete': self.can_complete_order(drone, order)
            }
            state_features['drones'].append(drone_features)
        
        # Add depot features
        for depot in self.depots:
            depot_features = {
                'id': depot.id,
                'location': depot.location,
                'capacity': depot.capacity,
                'available_drones': len([d for d in self.drones if d.depot_id == depot.id and d.status == DroneStatus.IDLE]),
                'distance_to_delivery': self.distance(depot.location, order.delivery_location)
            }
            state_features['depots'].append(depot_features)
        
        # Record the assignment decision
        assignment_data = {
            'state': state_features,
            'action': {
                'chosen_drone_id': chosen_drone.id,
                'chosen_depot_id': chosen_depot
            },
            'reward': self.calculate_assignment_reward(order, chosen_drone, chosen_depot)
        }
        
        self.training_data.append(assignment_data)
        
        # Also keep a simpler assignment history
        self.assignment_history.append({
            'timestamp': self.current_time,
            'order_id': order.id,
            'drone_id': chosen_drone.id,
            'start_depot': chosen_drone.depot_id,
            'end_depot': chosen_depot,
            'estimated_time': self.estimate_delivery_time(chosen_drone, order)[0]
        })
    
    # ADDED: Reward calculation for training data
    def calculate_assignment_reward(self, order: Order, drone: Drone, end_depot: int) -> float:
        """Calculate reward for the assignment decision"""
        estimated_time, _ = self.estimate_delivery_time(drone, order)
        
        # Positive reward components
        priority_bonus = (5 - order.priority.value) * 10  # Higher priority = higher bonus
        urgency_bonus = max(0, order.deadline - self.current_time - estimated_time) * 0.1
        
        # Negative reward components
        time_penalty = estimated_time
        battery_efficiency = (drone.current_battery / drone.battery_capacity) * 5
        
        total_reward = priority_bonus + urgency_bonus + battery_efficiency - time_penalty
        return total_reward
    
    def assign_orders_greedy(self):
        """Greedy assignment strategy with priority consideration and cross-depot optimization"""
        assignments = []
        
        for order in self.pending_orders[:]:
            if order.assigned_drone is not None:
                continue
            
            best_drone = None
            best_score = float('inf')
            best_end_depot = None
            
            # ADDED: Collect available drones for data collection
            available_drones = [drone for drone in self.drones if self.can_complete_order(drone, order)]
            
            for drone in self.drones:
                if not self.can_complete_order(drone, order):
                    continue
                
                # Calculate assignment score with cross-depot optimization
                estimated_time, end_depot_id = self.estimate_delivery_time(drone, order)
                priority_weight = 5 - order.priority.value  # Higher priority = lower weight
                urgency = max(0, order.deadline - self.current_time - estimated_time)
                
                # Calculate total distance for this assignment
                pickup_distance = self.distance(drone.location, order.pickup_location)
                delivery_distance = self.distance(order.pickup_location, order.delivery_location)
                end_depot = next(d for d in self.depots if d.id == end_depot_id)
                depot_distance = self.distance(order.delivery_location, end_depot.location)
                
                total_distance = pickup_distance + delivery_distance + depot_distance
                
                # Score combines distance, priority, urgency, and depot efficiency
                depot_efficiency_bonus = 0
                if end_depot_id != drone.depot_id:
                    # Small bonus for cross-depot operations (load balancing)
                    depot_efficiency_bonus = -2.0
                
                score = (total_distance + priority_weight * 10 - urgency * 0.1 + 
                        depot_efficiency_bonus)
                
                if score < best_score:
                    best_score = score
                    best_drone = drone
                    best_end_depot = end_depot_id
            
            if best_drone:
                order.assigned_drone = best_drone.id
                order.end_depot_id = best_end_depot  # Store the optimal end depot
                assignments.append((best_drone, order))
                self.pending_orders.remove(order)
                
                # ADDED: Collect training data for this assignment
                self.collect_training_data(order, available_drones, best_drone, best_end_depot)
        
        return assignments
    
    def update_drone_positions(self, dt: float):
        """Update drone positions and status with smoother animation"""
        for drone in self.drones:
            if drone.status == DroneStatus.IDLE or not drone.route:
                continue
            
            # MODIFIED: Smoother movement animation
            if drone.route:
                target = drone.route[0]
                distance_to_target = self.distance(drone.location, target)
                move_distance = drone.speed * dt
                
                if move_distance >= distance_to_target:
                    # Reached target
                    drone.location = target
                    drone.route.pop(0)
                    drone.total_distance += distance_to_target
                    drone.route_progress = 0.0  # Reset progress for next segment
                    
                    # Update status based on current objective
                    if drone.status == DroneStatus.FLYING_TO_PICKUP and drone.current_order:
                        if drone.location == drone.current_order.pickup_location:
                            drone.status = DroneStatus.FLYING_TO_DELIVERY
                            drone.route = [drone.current_order.delivery_location]
                            # ADDED: Update full route for visualization
                            if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                                end_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                                drone.full_route = [drone.current_order.delivery_location, end_depot.location]
                            drone.current_order.pickup_time = self.current_time
                    
                    elif drone.status == DroneStatus.FLYING_TO_DELIVERY and drone.current_order:
                        if drone.location == drone.current_order.delivery_location:
                            # Order completed
                            drone.current_order.delivery_time = self.current_time
                            drone.current_order.completed = True
                            self.completed_orders.append(drone.current_order)
                            
                            # Go to optimal end depot (may be different from start depot)
                            if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                                end_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                            else:
                                # Fallback: find nearest depot
                                end_depot = min(self.depots, 
                                               key=lambda d: self.distance(drone.location, d.location))
                            
                            drone.route = [end_depot.location]
                            drone.full_route = [end_depot.location]  # ADDED: Update full route
                            drone.status = DroneStatus.RETURNING_TO_DEPOT
                            drone.target_depot_id = end_depot.id  # Track target depot
                            drone.current_order = None
                    
                    elif drone.status == DroneStatus.RETURNING_TO_DEPOT:
                        # Arrived at target depot (may be different from original depot)
                        drone.status = DroneStatus.IDLE
                        drone.current_battery = drone.battery_capacity  # Instant recharge for simulation
                        drone.full_route = []  # ADDED: Clear full route
                        
                        # Update drone's depot assignment if it ended at a different depot
                        if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                            # Remove from old depot
                            old_depot = next(d for d in self.depots if d.id == drone.depot_id)
                            if drone.id in old_depot.drones:
                                old_depot.drones.remove(drone.id)
                            
                            # Add to new depot
                            new_depot = next(d for d in self.depots if d.id == drone.target_depot_id)
                            if drone.id not in new_depot.drones:
                                new_depot.drones.append(drone.id)
                            
                            # Update drone's depot assignment
                            drone.depot_id = drone.target_depot_id
                            drone.target_depot_id = None
                
                else:
                    # MODIFIED: Smoother movement with progress tracking
                    direction = ((target[0] - drone.location[0]) / distance_to_target,
                               (target[1] - drone.location[1]) / distance_to_target)
                    
                    new_x = drone.location[0] + direction[0] * move_distance
                    new_y = drone.location[1] + direction[1] * move_distance
                    drone.location = (new_x, new_y)
                    drone.total_distance += move_distance
                    
                    # ADDED: Update route progress for smoother visualization
                    drone.route_progress = min(1.0, drone.route_progress + move_distance / distance_to_target)
    
    def execute_assignments(self, assignments: List[Tuple[Drone, Order]]):
        """Execute drone-order assignments with full route planning"""
        for drone, order in assignments:
            drone.current_order = order
            drone.status = DroneStatus.FLYING_TO_PICKUP
            drone.route = [order.pickup_location]
            
            # ADDED: Plan full route for visualization
            if hasattr(order, 'end_depot_id') and order.end_depot_id is not None:
                end_depot = next(d for d in self.depots if d.id == order.end_depot_id)
                drone.full_route = [order.pickup_location, order.delivery_location, end_depot.location]
            else:
                # Fallback: find nearest depot
                end_depot = min(self.depots, key=lambda d: self.distance(order.delivery_location, d.location))
                drone.full_route = [order.pickup_location, order.delivery_location, end_depot.location]
            
            drone.route_progress = 0.0
    
    def step(self, dt: float, new_orders: List[Order] = None):
        """Execute one simulation step"""
        self.current_time += dt
        
        # Add new orders
        if new_orders:
            for order in new_orders:
                self.add_order(order)
        
        # Update drone positions
        self.update_drone_positions(dt)
        
        # Assign pending orders
        assignments = self.assign_orders_greedy()
        self.execute_assignments(assignments)
        
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
    
    # ADDED: Method to save training data
    def save_training_data(self, filename: str = "drone_routing_training_data.pkl"):
        """Save collected training data to file"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'training_data': self.training_data,
                'assignment_history': self.assignment_history,
                'depots': [(d.id, d.location, d.capacity) for d in self.depots],
                'drone_specs': self.drone_specs
            }, f)
        print(f"Training data saved to {filename} ({len(self.training_data)} samples)")

class DroneRoutingVisualizer:
    """Real-time visualization of the drone routing system with improved animations"""
    
    def __init__(self, optimizer: MultiDepotDroneRoutingOptimizer, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        self.optimizer = optimizer
        self.bounds = bounds
        
        # Setup plot
        self.fig, (self.ax_main, self.ax_metrics) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Main plot setup
        (min_lon, max_lon), (min_lat, max_lat) = bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title('Multi-Depot Drone Routing System')
        self.ax_main.grid(True, alpha=0.3)
        
        # Metrics plot setup
        self.ax_metrics.set_title('System Metrics')
        
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
        
    def update_visualization(self, frame):
        """Update visualization for animation with improved route display"""
        self.ax_main.clear()
        
        # Plot setup
        (min_lon, max_lon), (min_lat, max_lat) = self.bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title(f'Multi-Depot Drone Routing System (t={self.optimizer.current_time:.1f})')
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
        
        # MODIFIED: Enhanced drone visualization with full route display
        for drone in self.optimizer.drones:
            color = self.depot_colors[drone.depot_id]
            
            # ADDED: Different markers based on status for better visibility
            if drone.status == DroneStatus.IDLE:
                marker = 'o'
                alpha = 0.6
                size = 80
            elif drone.status == DroneStatus.FLYING_TO_PICKUP:
                marker = '^'
                alpha = 1.0
                size = 120
            elif drone.status == DroneStatus.FLYING_TO_DELIVERY:
                marker = '>'
                alpha = 1.0
                size = 120
            else:  # RETURNING_TO_DEPOT
                marker = 'v'
                alpha = 0.8
                size = 100
            
            # Plot drone with enhanced triangle marker
            self.ax_main.scatter(drone.location[0], drone.location[1], 
                               c=[color], s=size, marker=marker, alpha=alpha, 
                               edgecolors='black', linewidth=2, zorder=8)
            
            # ADDED: Plot full planned route with arrows for active drones
            if drone.full_route and drone.status != DroneStatus.IDLE:
                # Create complete route including current position
                complete_route = [drone.location] + drone.full_route
                route_x = [pos[0] for pos in complete_route]
                route_y = [pos[1] for pos in complete_route]
                
                # Plot route line with gradient effect
                self.ax_main.plot(route_x, route_y, color=color, alpha=0.7, 
                                linestyle='-', linewidth=3, zorder=4)
                
                # ADDED: Add arrows along the route to show direction
                for i in range(len(complete_route) - 1):
                    start = complete_route[i]
                    end = complete_route[i + 1]
                    
                    # Calculate arrow position (midpoint)
                    arrow_x = (start[0] + end[0]) / 2
                    arrow_y = (start[1] + end[1]) / 2
                    
                    # Calculate arrow direction
                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    
                    # Add directional arrow
                    self.ax_main.annotate('', xy=(arrow_x + dx*0.1, arrow_y + dy*0.1),
                                        xytext=(arrow_x - dx*0.1, arrow_y - dy*0.1),
                                        arrowprops=dict(arrowstyle='->', color=color, 
                                                      lw=2, alpha=0.8), zorder=5)
            
            # ADDED: Current route segment for immediate target
            elif drone.route and drone.status != DroneStatus.IDLE:
                target = drone.route[0]
                self.ax_main.plot([drone.location[0], target[0]], 
                                [drone.location[1], target[1]], 
                                color=color, alpha=0.6, linestyle='--', linewidth=2)
                
                # Add arrow to immediate target
                mid_x = (drone.location[0] + target[0]) / 2
                mid_y = (drone.location[1] + target[1]) / 2
                dx = target[0] - drone.location[0]
                dy = target[1] - drone.location[1]
                
                self.ax_main.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                                    xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                                    arrowprops=dict(arrowstyle='->', color=color, 
                                                  lw=1.5, alpha=0.7))
        
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
            
            # Connection line with priority color and arrow
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.5, linestyle='--', linewidth=2,
                            zorder=2)
            
            # ADDED: Arrow showing pickup to delivery direction
            mid_x = (order.pickup_location[0] + order.delivery_location[0]) / 2
            mid_y = (order.pickup_location[1] + order.delivery_location[1]) / 2
            dx = order.delivery_location[0] - order.pickup_location[0]
            dy = order.delivery_location[1] - order.pickup_location[1]
            
            self.ax_main.annotate('', xy=(mid_x + dx*0.1, mid_y + dy*0.1),
                                xytext=(mid_x - dx*0.1, mid_y - dy*0.1),
                                arrowprops=dict(arrowstyle='->', color=priority_color, 
                                              lw=1.5, alpha=0.6))
        
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
        
        # Enhanced legend with better organization
        legend_elements = []
        
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
        
        # MODIFIED: Enhanced drone status legend with new markers
        legend_elements.append(plt.Line2D([0], [0], linestyle='-', color='gray', 
                                        label='🚁 Drone Status:', linewidth=0))
        drone_status_legend = [
            ('⭕ Idle', 'o', 0.6),
            ('▲ To Pickup', '^', 1.0),
            ('▶ To Delivery', '>', 1.0),
            ('▼ Returning', 'v', 0.8),
        ]
        
        for label, marker, alpha in drone_status_legend:
            legend_elements.append(plt.Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='gray', markersize=8, 
                                            label=f'  {label}', alpha=alpha))
        
        legend_elements.extend([
            plt.Line2D([0], [0], linestyle='-', color='gray', 
                      linewidth=3, label='  ➤ Planned Route'),
            plt.Line2D([0], [0], linestyle='--', color='gray', 
                      linewidth=2, label='  ➤ Current Target')
        ])
        
        self.ax_main.legend(handles=legend_elements, loc='upper right', 
                           bbox_to_anchor=(1.0, 1.0), fontsize=8, 
                           frameon=True, fancybox=True, shadow=True)
        
        # Update metrics plot
        self.time_data.append(self.optimizer.current_time)
        self.completed_orders_data.append(self.optimizer.metrics['total_orders_completed'])
        self.avg_delivery_time_data.append(self.optimizer.metrics['average_delivery_time'])
        
        self.ax_metrics.clear()
        self.ax_metrics.plot(self.time_data, self.completed_orders_data, 'b-', label='Completed Orders')
        if max(self.avg_delivery_time_data) > 0:
            self.ax_metrics.plot(self.time_data, self.avg_delivery_time_data, 'r-', label='Avg Delivery Time')
        
        self.ax_metrics.set_xlabel('Time')
        self.ax_metrics.set_ylabel('Count / Time')
        self.ax_metrics.set_title('Performance Metrics')
        self.ax_metrics.legend()
        self.ax_metrics.grid(True, alpha=0.3)

# MODIFIED: Main simulation function with slower animation and data collection
def run_simulation(region='texas', size='medium', duration=300.0, dt=1.0, order_rate=0.9):
    """Run the complete simulation with data collection"""
    
    # Generate problem instance
    instance = RealWorldDataGenerator.generate_instance(region, size, duration)
    
    # Drone specifications
    drone_specs = {
        'speed': 1.5,  # MODIFIED: Reduced speed for better visualization
        'battery_capacity': 100.0,
        'payload_capacity': 10.0
    }
    
    # Initialize optimizer
    optimizer = MultiDepotDroneRoutingOptimizer(instance['depots'], drone_specs)
    
    # Initialize visualizer
    visualizer = DroneRoutingVisualizer(optimizer, instance['bounds'])
    
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
        
        # Also add some random orders based on order rate
        if random.random() < order_rate * dt / 10:  # MODIFIED: Adjusted for slower animation
            (min_lon, max_lon), (min_lat, max_lat) = instance['bounds']
            random_order = Order(
                id=len(orders) + len(new_orders) + frame * 1000,  # Unique ID
                pickup_location=(random.uniform(min_lon, max_lon), 
                               random.uniform(min_lat, max_lat)),
                delivery_location=(random.uniform(min_lon, max_lon), 
                                 random.uniform(min_lat, max_lat)),
                priority=random.choice(list(OrderPriority)),
                arrival_time=optimizer.current_time,
                deadline=optimizer.current_time + random.uniform(60, 300)
            )
            new_orders.append(random_order)
        
        # Step simulation
        optimizer.step(dt, new_orders)
        
        # Update visualization
        visualizer.update_visualization(frame)
        
        # Print metrics periodically
        if frame % 20 == 0:  # MODIFIED: Adjusted frequency for slower animation
            metrics = optimizer.metrics
            print(f"Time: {optimizer.current_time:.1f}, "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Training samples: {len(optimizer.training_data)}, "
                  f"Avg Delivery: {metrics['average_delivery_time']:.2f}")
    
    # MODIFIED: Slower animation with increased interval
    ani = animation.FuncAnimation(visualizer.fig, animate, 
                                 frames=int(duration/dt), 
                                 interval=int(dt*200),  # MODIFIED: Slower animation (200ms per frame)
                                 repeat=False, blit=False)
    
    plt.tight_layout()

    # plt.tight_layout()

    # Set up a callback to save data when animation ends
    def on_animation_complete():
        optimizer.save_training_data("greedy_training_data.pkl")
        print("Training data saved after animation completion.")

    # Connect the callback to figure close event
    visualizer.fig.canvas.mpl_connect('close_event', lambda evt: on_animation_complete())

    plt.show()
    # return ani, optimizer, visualizer



# MODIFIED: Main simulation function with slower animation and data collection
import matplotlib.pyplot as plt
from matplotlib import animation
import random

def run_test_simulation(region, size, instance, duration, dt, order_rate):
    """Run the complete simulation with data collection and automatic closure."""

    drone_specs = {
        'speed': 1.5,
        'battery_capacity': 100.0,
        'payload_capacity': 10.0
    }

    optimizer = MultiDepotDroneRoutingOptimizer(instance['depots'], drone_specs)
    visualizer = DroneRoutingVisualizer(optimizer, instance['bounds'])

    orders = sorted(instance['orders'], key=lambda x: x.arrival_time)
    order_index = 0
    final_output = None
    total_frames = int(duration / dt)

    def animate(frame):
        nonlocal order_index, final_output

        new_orders = []
        while (order_index < len(orders) and 
               orders[order_index].arrival_time <= optimizer.current_time):
            new_orders.append(orders[order_index])
            order_index += 1

        if random.random() < order_rate * dt / 10:
            (min_lon, max_lon), (min_lat, max_lat) = instance['bounds']
            random_order = Order(
                id=len(orders) + len(new_orders) + frame * 1000,
                pickup_location=(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)),
                delivery_location=(random.uniform(min_lon, max_lon), random.uniform(min_lat, max_lat)),
                priority=random.choice(list(OrderPriority)),
                arrival_time=optimizer.current_time,
                deadline=optimizer.current_time + random.uniform(60, 300)
            )
            new_orders.append(random_order)

        optimizer.step(dt, new_orders)
        visualizer.update_visualization(frame)

        if frame % 20 == 0:
            metrics = optimizer.metrics
            print(f"Time: {optimizer.current_time:.1f}, "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Training samples: {len(optimizer.training_data)}, "
                  f"Avg Delivery: {metrics['average_delivery_time']:.2f}")

        # Auto-close after final frame
        if frame == total_frames - 1:
            optimizer.save_training_data("greedy_training_data.pkl")
            print("Training data saved after animation completion.")
            metrics = optimizer.metrics
            final_output = (f"Time: {optimizer.current_time:.1f}, "
                            f"Completed: {metrics['total_orders_completed']}, "
                            f"Pending: {len(optimizer.pending_orders)}, "
                            f"Training samples: {len(optimizer.training_data)}, "
                            f"Avg Delivery: {metrics['average_delivery_time']:.2f}")
            print(final_output)
            plt.close(visualizer.fig)

    ani = animation.FuncAnimation(visualizer.fig, animate,
                                  frames=total_frames,
                                  interval=int(dt * 200),
                                  repeat=False, blit=False)

    plt.tight_layout()
    plt.show()

    return final_output



    





# === CONFIGURABLE PARAMETERS ===
REGIONS = ['arkansas', 'north_carolina', 'utah', 'texas']
SIZES = ['small', 'medium', 'large']
CACHE_DIR = "cached_instances"

SIM_DURATION = 350.0  #fixed
DT = 2.0   # fixed
ORDER_RATE_RANGE = (0.2, 0.5)  # Random range for testing variability

# === UTILITY FUNCTIONS ===
def save_instance(instance, filename):
    with open(filename, 'wb') as f:
        pickle.dump(instance, f)

def load_instance(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def generate_and_save_instance(region, size):
    instance = RealWorldDataGenerator.generate_instance(region, size)
    filename = os.path.join(CACHE_DIR, f"{region}_{size}.pkl")
    save_instance(instance, filename)
    return instance

def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)

# === MAIN TESTING SCRIPT ===
if __name__ == "__main__":
    print("=== Multi-Depot Drone Routing: Instance Generation, Saving, and Simulation ===\n")
    ensure_cache_dir()

    results_file = "results_greedy.txt"
    with open(results_file, "w") as txt:
        for region in REGIONS:
            for size in SIZES:
                print(f"\n▶️ Generating and saving instance for {region.title()} ({size})")
                instance = generate_and_save_instance(region, size)
                depots_count = len(instance['depots'])
                drones_count = sum(len(d.drones) for d in instance['depots'])
                orders_count = len(instance['orders'])

                print(f"   - Depots: {depots_count}")
                print(f"   - Drones: {drones_count}")
                print(f"   - Orders: {orders_count}")

                # Random order rate for this scenario
                order_rate = 0.3  #round(random.uniform(*ORDER_RATE_RANGE), 2)
                print(f"   - Order Rate for simulation: {order_rate}")

                # Run simulation and get final output
                print("   - Starting simulation...")
                final_output = run_test_simulation(region, size, instance,
                                                   duration=SIM_DURATION,
                                                   dt=DT,
                                                   order_rate=order_rate)
                # Save result to text file
                txt.write(f"Region: {region}, Size: {size}, Order Rate: {order_rate}\n")
                txt.write(f"Depots: {depots_count}, Drones: {drones_count}, Orders: {orders_count}\n")
                txt.write(f"Result: {final_output}\n")
                txt.write("=" * 60 + "\n")

    print(f"\n✅ All scenarios tested. Results saved to: {results_file}")







# # Example usage and testing
# if __name__ == "__main__":
#     print("Multi-Depot Dynamic Drone Routing System with Enhanced Visualization and Data Collection")
#     print("=" * 80)
    
#     # Generate and display instance information
#     for region in ['arkansas', 'north_carolina', 'utah', 'texas']:
#         for size in ['small', 'medium', 'large']:
#             instance = RealWorldDataGenerator.generate_instance(region, size)
#             print(f"{region.title()} {size}: "
#                   f"{len(instance['depots'])} depots, "
#                   f"{sum(len(d.drones) for d in instance['depots'])} drones, "
#                   f"{len(instance['orders'])} orders")
    
#     print("\nStarting simulation with data collection...")
    
#     # MODIFIED: Run simulation with slower animation and data collection
#     run_simulation('texas', 'large', instance, duration=350.0, dt=2.0, order_rate=0.3)
