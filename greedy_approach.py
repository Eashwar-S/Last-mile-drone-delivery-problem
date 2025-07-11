import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random
import time
import math
import heapq
import pickle
import csv
import os
import glob
from datetime import datetime
from collections import defaultdict

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
    end_depot_id: Optional[int] = None  # ADDED: Track optimal end depot

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
    speed_mps: float  # meters per second
    battery_duration_seconds: float  # total battery duration
    current_battery_seconds: float  # remaining battery time
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
    target_depot_id: Optional[int] = None  # ADDED: Track target depot for cross-depot operations
    
    # Animation support
    full_route: List[Tuple[float, float]] = field(default_factory=list)
    route_progress: float = 0.0

class ImprovedDroneRoutingOptimizer:
    """Improved optimizer with cross-depot operations and real-world drone constraints"""
    
    def __init__(self, instance: Dict):
        self.instance = instance
        self.depots = instance['depots']
        self.drone_specs = instance['drone_specs']
        self.time_horizon = instance['time_horizon']
        self.total_possible_score = instance['total_possible_score']
        
        self.drones = []
        self.pending_orders = []
        self.completed_orders = []
        self.failed_orders = []  # Orders that couldn't be completed on time
        self.current_time = 0.0
        
        # Assignment tracking for GNN training
        self.assignment_times = []  # Track time taken for each assignment
        self.gnn_training_data = []
        self.cross_depot_assignments = 0  # ADDED: Track cross-depot operations
        
        # Initialize drones (one per depot)
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
            'cross_depot_operations': 0  # ADDED: Track cross-depot efficiency
        }
    
    def _initialize_drones(self):
        """Initialize one drone per depot"""
        for depot in self.depots:
            drone = Drone(
                id=depot.id,  # One-to-one mapping
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
        
        R = 6371000  # Earth's radius in meters
        return R * c
    
    def calculate_flight_time(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate flight time between coordinates (returns seconds)"""
        distance_m = self.haversine_distance(coord1, coord2)
        return distance_m / self.drone_specs['cruising_speed_mps']
    
    def calculate_total_mission_time_with_depot(self, drone: Drone, order: Order, end_depot: Depot) -> float:
        """Calculate total mission time with specific end depot (returns seconds)"""
        # Flight times
        to_pickup_time = self.calculate_flight_time(drone.location, order.pickup_location)
        pickup_to_delivery_time = self.calculate_flight_time(order.pickup_location, order.delivery_location)
        delivery_to_depot_time = self.calculate_flight_time(order.delivery_location, end_depot.location)
        
        # Service and operational times
        takeoff_landing_time = self.drone_specs['takeoff_landing_time_seconds'] * 3
        service_time = self.drone_specs['service_time_seconds'] * 2
        
        total_time = (to_pickup_time + pickup_to_delivery_time + delivery_to_depot_time + 
                     takeoff_landing_time + service_time)
        
        return total_time
    
    def find_optimal_depot(self, delivery_location: Tuple[float, float]) -> Depot:
        """Find the optimal depot based on delivery location"""
        return min(self.depots, 
                  key=lambda d: self.haversine_distance(delivery_location, d.location))
    
    def can_complete_order(self, drone: Drone, order: Order) -> bool:
        """Check if drone can complete order within all constraints using optimal depot"""
        # Check drone availability
        if drone.status != DroneStatus.IDLE:
            return False
        
        # Check payload capacity
        if order.weight > drone.payload_capacity:
            return False
        
        # FIXED: Use optimal depot for feasibility check
        optimal_depot = self.find_optimal_depot(order.delivery_location)
        
        # Check battery constraint
        mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_depot)
        if mission_time > drone.current_battery_seconds * 0.9:  # 10% safety margin
            return False
        
        # Check time constraint (can we complete before deadline?)
        estimated_completion_time = self.current_time + mission_time / 60  # Convert to minutes
        if estimated_completion_time > order.deadline:
            return False
        
        return True
    
    def add_order(self, order: Order):
        """Add new order to pending list"""
        self.pending_orders.append(order)
        # Sort by priority (emergency first) then by deadline
        self.pending_orders.sort(key=lambda x: (-x.priority.value, x.deadline))
        
        # Track total orders by priority
        priority_key = f"{order.priority.name.lower()}_total"
        if priority_key in self.metrics:
            self.metrics[priority_key] += 1
    
    def collect_gnn_training_data(self, order: Order, available_drones: List[Drone], 
                                chosen_drone: Drone, chosen_end_depot_id: int, assignment_time: float):
        """Collect enhanced training data for GNN model with cross-depot information"""
        
        # State representation - Current system state when assignment decision is made
        state_features = {
            'timestamp': self.current_time,
            'order_features': {
                'id': order.id,
                'priority_value': order.priority.value,
                'priority_name': order.priority.name,
                'weight': order.weight,
                'score': order.score,
                'arrival_time': order.arrival_time,
                'deadline': order.deadline,
                'time_until_deadline': order.deadline - self.current_time,
                'urgency_ratio': (order.deadline - self.current_time) / (order.deadline - order.arrival_time) if order.deadline > order.arrival_time else 0,
                'pickup_location': order.pickup_location,
                'delivery_location': order.delivery_location,
                'order_distance': self.haversine_distance(order.pickup_location, order.delivery_location)
            },
            'system_state': {
                'total_pending_orders': len(self.pending_orders),
                'total_completed_orders': len(self.completed_orders),
                'total_failed_orders': len(self.failed_orders),
                'current_score': self.metrics['total_score'],
                'current_utilization': self.metrics['drone_utilization'],
                'cross_depot_operations': self.metrics['cross_depot_operations'],  # ADDED
                'pending_orders_by_priority': {
                    'emergency': len([o for o in self.pending_orders if o.priority == OrderPriority.EMERGENCY]),
                    'high': len([o for o in self.pending_orders if o.priority == OrderPriority.HIGH]),
                    'medium': len([o for o in self.pending_orders if o.priority == OrderPriority.MEDIUM]),
                    'low': len([o for o in self.pending_orders if o.priority == OrderPriority.LOW])
                }
            },
            'drone_candidates': []
        }
        
        # Add features for all available drones (candidates)
        for drone in available_drones:
            optimal_depot = self.find_optimal_depot(order.delivery_location)
            mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_depot)
            distance_to_pickup = self.haversine_distance(drone.location, order.pickup_location)
            order_distance = self.haversine_distance(order.pickup_location, order.delivery_location)
            distance_delivery_to_depot = self.haversine_distance(order.delivery_location, optimal_depot.location)
            
            drone_features = {
                'drone_id': drone.id,
                'depot_id': drone.depot_id,
                'location': drone.location,
                'battery_level': drone.current_battery_seconds / drone.battery_duration_seconds,
                'distance_to_pickup': distance_to_pickup,
                'distance_delivery_to_depot': distance_delivery_to_depot,
                'total_mission_distance': distance_to_pickup + order_distance + distance_delivery_to_depot,
                'estimated_mission_time_minutes': mission_time / 60,
                'battery_usage_ratio': mission_time / drone.current_battery_seconds,
                'can_complete_on_time': self.current_time + mission_time/60 <= order.deadline,
                'time_efficiency': max(0, (order.deadline - self.current_time - mission_time/60)) / order.deadline,
                'optimal_end_depot_id': optimal_depot.id,  # ADDED
                'is_cross_depot_operation': optimal_depot.id != drone.depot_id,  # ADDED
                'is_chosen': drone.id == chosen_drone.id
            }
            state_features['drone_candidates'].append(drone_features)
        
        # Action taken - Which drone was chosen and optimal depot
        action = {
            'chosen_drone_id': chosen_drone.id,
            'chosen_end_depot_id': chosen_end_depot_id,
            'assignment_time_seconds': assignment_time,
            'is_cross_depot': chosen_end_depot_id != chosen_drone.depot_id  # ADDED
        }
        
        # Enhanced reward calculation for better GNN training objectives
        reward_components = self._calculate_comprehensive_reward(order, chosen_drone, chosen_end_depot_id, assignment_time)
        
        training_sample = {
            'state': state_features,
            'action': action,
            'rewards': reward_components,
            'instance_info': {
                'region': self.instance['region'],
                'size': self.instance['size'],
                'total_possible_score': self.total_possible_score
            }
        }
        
        self.gnn_training_data.append(training_sample)
    
    def _calculate_comprehensive_reward(self, order: Order, drone: Drone, end_depot_id: int, assignment_time: float) -> Dict:
        """Calculate comprehensive reward components for GNN training with cross-depot benefits"""
        end_depot = next(d for d in self.depots if d.id == end_depot_id)
        mission_time = self.calculate_total_mission_time_with_depot(drone, order, end_depot)
        
        # 1. Score maximization reward (primary objective)
        score_reward = order.score
        
        # 2. Time efficiency reward
        completion_time = self.current_time + mission_time / 60
        time_efficiency = max(0, (order.deadline - completion_time) / (order.deadline - order.arrival_time))
        time_reward = time_efficiency * 5
        
        # 3. Resource efficiency reward
        battery_efficiency = 1 - (mission_time / drone.current_battery_seconds)
        resource_reward = battery_efficiency * 3
        
        # 4. Distance efficiency reward (minimize travel)
        total_distance = (self.haversine_distance(drone.location, order.pickup_location) +
                         self.haversine_distance(order.pickup_location, order.delivery_location) +
                         self.haversine_distance(order.delivery_location, end_depot.location))
        distance_penalty = total_distance / 10000  # Normalize
        
        # 5. Priority urgency reward
        urgency_multiplier = {
            OrderPriority.EMERGENCY: 3.0,
            OrderPriority.HIGH: 2.0,
            OrderPriority.MEDIUM: 1.5,
            OrderPriority.LOW: 1.0
        }[order.priority]
        
        # 6. Assignment speed reward (faster decision making)
        assignment_speed_reward = max(0, 2 - assignment_time)
        
        # 7. ADDED: Cross-depot efficiency reward
        cross_depot_reward = 0
        if end_depot_id != drone.depot_id:
            # Reward for optimal depot selection and load balancing
            cross_depot_reward = 3.0
        
        # Combined reward with different objectives
        rewards = {
            'score_reward': score_reward,
            'time_efficiency_reward': time_reward,
            'resource_efficiency_reward': resource_reward,
            'distance_penalty': -distance_penalty,
            'urgency_multiplier': urgency_multiplier,
            'assignment_speed_reward': assignment_speed_reward,
            'cross_depot_reward': cross_depot_reward,  # ADDED
            'total_reward': ((score_reward + time_reward + resource_reward - distance_penalty + cross_depot_reward) 
                           * urgency_multiplier + assignment_speed_reward),
            'normalized_total_reward': ((score_reward + time_reward + resource_reward - distance_penalty + cross_depot_reward) 
                                      * urgency_multiplier + assignment_speed_reward) / 25  # Normalize
        }
        
        return rewards
    
    def assign_orders_greedy(self):
        """Enhanced greedy assignment with cross-depot optimization and timing"""
        assignments = []
        
        # Process orders by priority and deadline
        for order in self.pending_orders[:]:
            if order.assigned_drone is not None:
                continue
            
            assignment_start_time = time.time()
            
            best_drone = None
            best_score = -float('inf')
            best_end_depot_id = None
            available_drones = [drone for drone in self.drones if self.can_complete_order(drone, order)]
            
            if not available_drones:
                continue
            
            for drone in available_drones:
                # FIXED: Find optimal end depot for each drone-order combination
                optimal_end_depot = self.find_optimal_depot(order.delivery_location)
                
                # Calculate mission time with optimal depot
                mission_time = self.calculate_total_mission_time_with_depot(drone, order, optimal_end_depot)
                
                # Base score from order priority and score
                base_score = order.score * 2  # Emphasize order score
                
                # Time efficiency bonus (faster completion = higher score)
                completion_time = self.current_time + mission_time / 60
                time_efficiency = max(0, (order.deadline - completion_time)) / order.deadline
                time_bonus = time_efficiency * 5
                
                # FIXED: Distance efficiency with optimal depot
                total_distance = (
                    self.haversine_distance(drone.location, order.pickup_location) +
                    self.haversine_distance(order.pickup_location, order.delivery_location) +
                    self.haversine_distance(order.delivery_location, optimal_end_depot.location)  # FIXED
                )
                distance_penalty = total_distance / 100000  # Normalize to reasonable scale
                
                # Battery efficiency bonus
                battery_efficiency = drone.current_battery_seconds / drone.battery_duration_seconds
                battery_bonus = battery_efficiency * 2
                
                # Priority urgency multiplier
                urgency_multiplier = {
                    OrderPriority.EMERGENCY: 3.0,
                    OrderPriority.HIGH: 2.0,
                    OrderPriority.MEDIUM: 1.5,
                    OrderPriority.LOW: 1.0
                }[order.priority]
                
                # ADDED: Cross-depot efficiency bonus
                cross_depot_bonus = 0
                if optimal_end_depot.id != drone.depot_id:
                    # Significant bonus for cross-depot operations that improve efficiency
                    cross_depot_bonus = 5.0
                    
                    # Additional bonus based on depot load balancing
                    current_depot_load = len([d for d in self.drones if d.depot_id == drone.depot_id and d.status != DroneStatus.IDLE])
                    target_depot_load = len([d for d in self.drones if d.depot_id == optimal_end_depot.id and d.status != DroneStatus.IDLE])
                    
                    if target_depot_load < current_depot_load:
                        cross_depot_bonus += 2.0  # Extra bonus for load balancing
                
                # Future opportunity cost (consider remaining orders)
                future_penalty = 0
                if len(self.pending_orders) > 5:  # If many orders pending
                    future_penalty = -0.5  # Small penalty for not reserving capacity
                
                # Final score calculation with cross-depot optimization
                assignment_score = ((base_score + time_bonus + battery_bonus - distance_penalty + 
                                   cross_depot_bonus + future_penalty) * urgency_multiplier)
                
                if assignment_score > best_score:
                    best_score = assignment_score
                    best_drone = drone
                    best_end_depot_id = optimal_end_depot.id
            
            if best_drone:
                assignment_end_time = time.time()
                assignment_duration = assignment_end_time - assignment_start_time
                self.assignment_times.append(assignment_duration)
                
                order.assigned_drone = best_drone.id
                order.end_depot_id = best_end_depot_id  # FIXED: Store optimal end depot
                assignments.append((best_drone, order))
                self.pending_orders.remove(order)
                
                # Track cross-depot operations
                if best_end_depot_id != best_drone.depot_id:
                    self.cross_depot_assignments += 1
                
                # Collect GNN training data with cross-depot information
                self.collect_gnn_training_data(order, available_drones, best_drone, best_end_depot_id, assignment_duration)
        
        return assignments
    
    def update_drone_positions(self, dt: float):
        """Update drone positions and handle state transitions with cross-depot support"""
        for drone in self.drones:
            # Handle charging
            if drone.status == DroneStatus.CHARGING:
                if drone.charging_start_time is None:
                    drone.charging_start_time = self.current_time
                
                charging_duration = (self.current_time - drone.charging_start_time) * 60  # Convert to seconds
                if charging_duration >= drone.charging_time_seconds:
                    # Charging complete
                    drone.current_battery_seconds = drone.battery_duration_seconds
                    drone.status = DroneStatus.IDLE
                    drone.charging_start_time = None
                continue
            
            # Handle idle drones
            if drone.status == DroneStatus.IDLE:
                drone.total_idle_time += dt
                
                # Check if need to charge
                if drone.current_battery_seconds < drone.battery_duration_seconds * 0.2:  # 20% threshold
                    drone.status = DroneStatus.CHARGING
                    drone.charging_start_time = self.current_time
                continue
            
            # Handle moving drones
            if not drone.route:
                continue
            
            target = drone.route[0]
            distance_to_target = self.haversine_distance(drone.location, target)
            
            # Convert movement: dt is in minutes, speed is in m/s
            move_distance_meters = drone.speed_mps * dt * 60  # Convert dt to seconds
            
            if move_distance_meters >= distance_to_target:
                # Reached target
                drone.location = target
                drone.route.pop(0)
                drone.total_distance_meters += distance_to_target
                
                # Calculate flight time and battery consumption
                flight_time_seconds = distance_to_target / drone.speed_mps
                flight_time_minutes = flight_time_seconds / 60
                drone.total_flight_time += flight_time_minutes
                drone.current_battery_seconds -= flight_time_seconds
                
                # Handle state transitions based on location
                if drone.status == DroneStatus.FLYING_TO_PICKUP and drone.current_order:
                    pickup_distance = self.haversine_distance(drone.location, drone.current_order.pickup_location)
                    if pickup_distance < 50:  # Within 50m of pickup
                        drone.status = DroneStatus.FLYING_TO_DELIVERY
                        drone.route = [drone.current_order.delivery_location]
                        
                        # FIXED: Update full route to show optimal depot
                        if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                            end_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                            drone.full_route = [drone.current_order.delivery_location, end_depot.location]
                        else:
                            # Fallback to nearest depot
                            optimal_depot = self.find_optimal_depot(drone.current_order.delivery_location)
                            drone.full_route = [drone.current_order.delivery_location, optimal_depot.location]
                        
                        drone.current_order.pickup_time = self.current_time
                        
                        # Add service time battery consumption
                        service_time_seconds = self.drone_specs['service_time_seconds']
                        drone.current_battery_seconds -= service_time_seconds
                
                elif drone.status == DroneStatus.FLYING_TO_DELIVERY and drone.current_order:
                    delivery_distance = self.haversine_distance(drone.location, drone.current_order.delivery_location)
                    if delivery_distance < 50:  # Within 50m of delivery
                        # Order completed successfully
                        drone.current_order.delivery_time = self.current_time
                        drone.current_order.completed = True
                        self.completed_orders.append(drone.current_order)
                        
                        # Add to total score and priority-specific counts
                        self.metrics['total_score'] += drone.current_order.score
                        priority_key = f"{drone.current_order.priority.name.lower()}_completed"
                        if priority_key in self.metrics:
                            self.metrics[priority_key] += 1
                        
                        # FIXED: Return to optimal depot (from order assignment)
                        if hasattr(drone.current_order, 'end_depot_id') and drone.current_order.end_depot_id is not None:
                            target_depot = next(d for d in self.depots if d.id == drone.current_order.end_depot_id)
                        else:
                            # Fallback: find nearest depot to delivery location
                            target_depot = self.find_optimal_depot(drone.current_order.delivery_location)
                        
                        drone.route = [target_depot.location]
                        drone.full_route = [target_depot.location]
                        drone.status = DroneStatus.RETURNING_TO_DEPOT
                        drone.target_depot_id = target_depot.id  # FIXED: Use optimal depot
                        drone.current_order = None
                        
                        # Add service time battery consumption
                        service_time_seconds = self.drone_specs['service_time_seconds']
                        drone.current_battery_seconds -= service_time_seconds
                
                elif drone.status == DroneStatus.RETURNING_TO_DEPOT:
                    # FIXED: Check arrival at target depot (which may be different from original)
                    if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                        target_depot_location = next(d.location for d in self.depots if d.id == drone.target_depot_id)
                    else:
                        target_depot_location = next(d.location for d in self.depots if d.id == drone.depot_id)
                    
                    depot_distance = self.haversine_distance(drone.location, target_depot_location)
                    if depot_distance < 50:  # Within 50m of target depot
                        drone.status = DroneStatus.IDLE
                        drone.full_route = []
                        
                        # FIXED: Update drone's depot assignment to optimal depot
                        if hasattr(drone, 'target_depot_id') and drone.target_depot_id is not None:
                            old_depot_id = drone.depot_id
                            new_depot_id = drone.target_depot_id
                            
                            if old_depot_id != new_depot_id:
                                # Cross-depot operation completed
                                self.metrics['cross_depot_operations'] += 1
                                
                                # Remove from old depot's drone list (conceptual - for future extensions)
                                # In this simplified version, we just update the drone's depot_id
                                
                                # Update drone's depot assignment
                                drone.depot_id = new_depot_id
                                drone.target_depot_id = None
                                
                                print(f"Drone {drone.id} completed cross-depot operation: {old_depot_id} → {new_depot_id}")
                        
                        # Add landing time battery consumption
                        landing_time_seconds = self.drone_specs['takeoff_landing_time_seconds']
                        drone.current_battery_seconds -= landing_time_seconds
            
            else:
                # Move towards target
                bearing = math.atan2(target[1] - drone.location[1], target[0] - drone.location[0])
                
                # Convert movement to lat/lon degrees (approximate)
                lat_change = (move_distance_meters * math.sin(bearing)) / 111320  # meters to degrees lat
                lon_change = (move_distance_meters * math.cos(bearing)) / (111320 * math.cos(math.radians(drone.location[1])))  # meters to degrees lon
                
                new_lat = drone.location[1] + lat_change
                new_lon = drone.location[0] + lon_change
                drone.location = (new_lon, new_lat)
                
                drone.total_distance_meters += move_distance_meters
                
                # Calculate flight time and battery consumption for partial movement
                flight_time_seconds = dt * 60  # dt in minutes to seconds
                flight_time_minutes = dt
                drone.total_flight_time += flight_time_minutes
                drone.current_battery_seconds -= flight_time_seconds
    
    def execute_assignments(self, assignments: List[Tuple[Drone, Order]]):
        """Execute drone-order assignments with cross-depot planning"""
        for drone, order in assignments:
            drone.current_order = order
            drone.status = DroneStatus.FLYING_TO_PICKUP
            drone.route = [order.pickup_location]
            
            # FIXED: Plan full route to optimal depot
            if hasattr(order, 'end_depot_id') and order.end_depot_id is not None:
                end_depot = next(d for d in self.depots if d.id == order.end_depot_id)
                drone.full_route = [order.pickup_location, order.delivery_location, end_depot.location]
            else:
                # Fallback: find nearest depot
                optimal_depot = self.find_optimal_depot(order.delivery_location)
            
                drone.full_route = [order.pickup_location, order.delivery_location, optimal_depot.location]
                order.end_depot_id = optimal_depot.id
            
            drone.mission_start_time = self.current_time
            drone.route_progress = 0.0
            
            # Add takeoff time battery consumption
            takeoff_time_seconds = self.drone_specs['takeoff_landing_time_seconds']
            drone.current_battery_seconds -= takeoff_time_seconds
            
            # Debug print for cross-depot operations
            if order.end_depot_id != drone.depot_id:
                print(f"Cross-depot assignment: Drone {drone.id} (Depot {drone.depot_id}) → Order {order.id} → End Depot {order.end_depot_id}")
    
    def handle_expired_orders(self):
        """Move expired orders to failed list"""
        expired_orders = []
        for order in self.pending_orders:
            if self.current_time > order.deadline:
                expired_orders.append(order)
                self.failed_orders.append(order)
        
        for order in expired_orders:
            self.pending_orders.remove(order)
    
    def step(self, dt: float, new_orders: List[Order] = None):
        """Execute one simulation step"""
        self.current_time += dt
        
        # Add new orders that have arrived
        if new_orders:
            for order in new_orders:
                self.add_order(order)
        
        # Handle expired orders
        self.handle_expired_orders()
        
        # Update drone positions and states
        self.update_drone_positions(dt)
        
        # Assign pending orders with cross-depot optimization
        assignments = self.assign_orders_greedy()
        self.execute_assignments(assignments)
        
        # Update metrics
        self._update_metrics()
    
    def _update_metrics(self):
        """Update comprehensive performance metrics"""
        # Basic counts
        self.metrics['total_orders_completed'] = len(self.completed_orders)
        self.metrics['total_orders_failed'] = len(self.failed_orders)
        
        # Score metrics
        self.metrics['score_ratio'] = self.metrics['total_score'] / self.total_possible_score if self.total_possible_score > 0 else 0
        
        # Distance and time metrics
        self.metrics['total_distance_km'] = sum(drone.total_distance_meters for drone in self.drones) / 1000
        self.metrics['total_flight_time_hours'] = sum(drone.total_flight_time for drone in self.drones) / 60
        self.metrics['total_idle_time_hours'] = sum(drone.total_idle_time for drone in self.drones) / 60
        
        # Assignment time metrics
        if self.assignment_times:
            self.metrics['average_assignment_time'] = np.mean(self.assignment_times)
        
        # Delivery time
        if self.completed_orders:
            delivery_times = [order.delivery_time - order.arrival_time 
                            for order in self.completed_orders if order.delivery_time]
            if delivery_times:
                self.metrics['average_delivery_time'] = np.mean(delivery_times)
        
        # Utilization
        total_time = self.metrics['total_flight_time_hours'] + self.metrics['total_idle_time_hours']
        if total_time > 0:
            self.metrics['drone_utilization'] = self.metrics['total_flight_time_hours'] / total_time
        
        # Priority-specific completion rates
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
        """Save final metrics as single row to CSV file"""
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
            'cross_depot_operations': self.metrics['cross_depot_operations'],  # ADDED
            'cross_depot_assignments': self.cross_depot_assignments,  # ADDED
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
            'low_completion_rate': self.metrics['low_completion_rate']
        }
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = final_metrics.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerow(final_metrics)
        
        print(f"Metrics saved to {filename}")
    
    def save_gnn_training_data(self, filename: str):
        """Save GNN training data to pickle file"""
        if not self.gnn_training_data:
            print("No GNN training data to save")
            return
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        training_data = {
            'assignment_decisions': self.gnn_training_data,
            'instance_metadata': {
                'region': self.instance['region'],
                'size': self.instance['size'],
                'total_depots': len(self.depots),
                'total_orders': len(self.instance['orders']),
                'time_horizon': self.time_horizon,
                'drone_specs': self.drone_specs
            },
            'final_metrics': self.metrics,
            'solution_quality': {
                'total_score_achieved': self.metrics['total_score'],
                'total_possible_score': self.total_possible_score,
                'score_ratio': self.metrics['score_ratio'],
                'orders_completed': self.metrics['total_orders_completed'],
                'orders_failed': self.metrics['total_orders_failed'],
                'cross_depot_operations': self.metrics['cross_depot_operations']  # ADDED
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(training_data, f)
        
        print(f"GNN training data saved to {filename} ({len(self.gnn_training_data)} samples)")
        print(f"Cross-depot assignments made: {self.cross_depot_assignments}")
    
    def print_final_report(self):
        """Print comprehensive final performance report"""
        print("\n" + "="*80)
        print("IMPROVED GREEDY DRONE ROUTING PERFORMANCE REPORT")
        print("="*80)
        
        print(f"Instance: {self.instance['region']} {self.instance['size']}")
        print(f"Time Horizon: {self.time_horizon:.1f} minutes")
        print(f"Total Depots: {len(self.depots)}")
        print(f"Total Drones: {len(self.drones)}")
        
        print(f"\nSCORE PERFORMANCE:")
        print(f"  Total Score Achieved: {self.metrics['total_score']}")
        print(f"  Total Possible Score: {self.total_possible_score}")
        print(f"  Score Ratio: {self.metrics['score_ratio']:.3f} ({self.metrics['score_ratio']*100:.1f}%)")
        
        print(f"\nORDER STATISTICS:")
        print(f"  Orders Completed: {self.metrics['total_orders_completed']}")
        print(f"  Orders Failed: {self.metrics['total_orders_failed']}")
        print(f"  Orders Pending: {len(self.pending_orders)}")
        print(f"  Total Orders: {len(self.instance['orders'])}")
        
        completion_rate = self.metrics['total_orders_completed'] / len(self.instance['orders']) if len(self.instance['orders']) > 0 else 0
        print(f"  Overall Completion Rate: {completion_rate:.3f} ({completion_rate*100:.1f}%)")
        
        print(f"\nCROSS-DEPOT OPERATIONS:")
        print(f"  Cross-Depot Assignments: {self.cross_depot_assignments}")
        print(f"  Cross-Depot Operations Completed: {self.metrics['cross_depot_operations']}")
        cross_depot_rate = (self.cross_depot_assignments / max(1, self.metrics['total_orders_completed'])) * 100
        print(f"  Cross-Depot Rate: {cross_depot_rate:.1f}%")
        
        print(f"\nPRIORITY-SPECIFIC COMPLETION:")
        print(f"  Emergency: {self.metrics['emergency_completed']}/{self.metrics['emergency_total']} ({self.metrics['emergency_completion_rate']*100:.1f}%)")
        print(f"  High: {self.metrics['high_completed']}/{self.metrics['high_total']} ({self.metrics['high_completion_rate']*100:.1f}%)")
        print(f"  Medium: {self.metrics['medium_completed']}/{self.metrics['medium_total']} ({self.metrics['medium_completion_rate']*100:.1f}%)")
        print(f"  Low: {self.metrics['low_completed']}/{self.metrics['low_total']} ({self.metrics['low_completion_rate']*100:.1f}%)")
        
        print(f"\nOPERATIONAL METRICS:")
        print(f"  Total Distance: {self.metrics['total_distance_km']:.1f} km")
        print(f"  Total Flight Time: {self.metrics['total_flight_time_hours']:.2f} hours")
        print(f"  Total Idle Time: {self.metrics['total_idle_time_hours']:.2f} hours")
        print(f"  Drone Utilization: {self.metrics['drone_utilization']:.3f} ({self.metrics['drone_utilization']*100:.1f}%)")
        print(f"  Average Delivery Time: {self.metrics['average_delivery_time']:.2f} minutes")
        print(f"  Average Assignment Time: {self.metrics['average_assignment_time']*1000:.2f} ms")
        
        print(f"\nDRONE DETAILS:")
        depot_distribution = defaultdict(int)
        for drone in self.drones:
            depot_distribution[drone.depot_id] += 1
            battery_pct = (drone.current_battery_seconds / drone.battery_duration_seconds) * 100
            print(f"  Drone {drone.id}: Current Depot={drone.depot_id}, Distance={drone.total_distance_meters/1000:.1f}km, "
                  f"Flight={drone.total_flight_time:.1f}min, Idle={drone.total_idle_time:.1f}min, "
                  f"Battery={battery_pct:.1f}%, Status={drone.status.value}")
        
        print(f"\nDEPOT LOAD DISTRIBUTION:")
        for depot_id, count in depot_distribution.items():
            print(f"  Depot {depot_id}: {count} drones")

class ImprovedDroneRoutingVisualizer:
    """Enhanced visualizer with cross-depot operation indicators"""
    
    def __init__(self, optimizer: ImprovedDroneRoutingOptimizer, bounds: Tuple[Tuple[float, float], Tuple[float, float]]):
        self.optimizer = optimizer
        self.bounds = bounds
        
        # Setup plot
        self.fig, ((self.ax_main, self.ax_score), (self.ax_completion, self.ax_utilization)) = plt.subplots(2, 2, figsize=(20, 12))
        
        # Main plot setup
        (min_lon, max_lon), (min_lat, max_lat) = bounds
        self.ax_main.set_xlim(min_lon, max_lon)
        self.ax_main.set_ylim(min_lat, max_lat)
        self.ax_main.set_xlabel('Longitude')
        self.ax_main.set_ylabel('Latitude')
        self.ax_main.set_title('Cross-Depot Enhanced Drone Routing System')
        self.ax_main.grid(True, alpha=0.3)
        
        # Metrics plots setup
        self.ax_score.set_title('Score Performance')
        self.ax_completion.set_title('Order Completion')
        self.ax_utilization.set_title('Drone Utilization & Cross-Depot Ops')
        
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
        self.utilization_data = []
        self.cross_depot_data = []  # ADDED
    
    def update_visualization(self, frame):
        """Update all visualization components with cross-depot indicators"""
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
        
        self.ax_main.set_title(f'Cross-Depot Enhanced Routing - t={self.optimizer.current_time:.1f}min - Score: {current_score}/{max_score} ({score_pct:.1f}%) - Cross-Depot: {cross_depot_ops}')
        self.ax_main.grid(True, alpha=0.3)
        
        # Plot depots
        for i, depot in enumerate(self.optimizer.depots):
            self.ax_main.scatter(depot.location[0], depot.location[1], 
                               c=[self.depot_colors[i]], s=400, marker='s', 
                               label=f'Depot {depot.id}', edgecolors='black', 
                               linewidth=3, alpha=0.9, zorder=10)
            
            # Count drones currently at this depot
            drones_at_depot = len([d for d in self.optimizer.drones if d.depot_id == depot.id])
            
            self.ax_main.annotate(f'D{depot.id}({drones_at_depot})', depot.location, 
                                xytext=(8, 8), textcoords='offset points',
                                fontweight='bold', fontsize=10,
                                bbox=dict(boxstyle='round,pad=0.3', 
                                         facecolor=self.depot_colors[i], alpha=0.7),
                                zorder=11)
        
        # Plot drones with enhanced status indicators and cross-depot highlighting
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
            
            # ADDED: Highlight cross-depot operations
            edge_color = 'red' if (hasattr(drone, 'target_depot_id') and 
                                  drone.target_depot_id is not None and 
                                  drone.target_depot_id != drone.depot_id) else 'black'
            edge_width = 3 if edge_color == 'red' else 2
            
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
            
            # Plot planned routes with cross-depot indication
            if drone.full_route and drone.status != DroneStatus.IDLE and drone.status != DroneStatus.CHARGING:
                complete_route = [drone.location] + drone.full_route
                route_x = [pos[0] for pos in complete_route]
                route_y = [pos[1] for pos in complete_route]
                
                # Use different line style for cross-depot operations
                line_style = '--' if (hasattr(drone, 'target_depot_id') and 
                                    drone.target_depot_id is not None and 
                                    drone.target_depot_id != drone.depot_id) else '-'
                line_width = 3 if line_style == '--' else 2
                
                self.ax_main.plot(route_x, route_y, color=color, alpha=0.7, 
                                linestyle=line_style, linewidth=line_width, zorder=4)
        
        # Plot orders
        for order in self.optimizer.pending_orders:
            priority_color = self.priority_colors[order.priority]
            
            # Pickup location
            self.ax_main.scatter(order.pickup_location[0], order.pickup_location[1], 
                               c='#2E8B57', s=100, marker='o', alpha=0.8, 
                               edgecolors='black', linewidth=1, zorder=5)
            self.ax_main.annotate('P', order.pickup_location, 
                                ha='center', va='center', fontweight='bold', 
                                fontsize=8, color='white', zorder=6)
            
            # Delivery location
            self.ax_main.scatter(order.delivery_location[0], order.delivery_location[1], 
                               c='#DC143C', s=100, marker='s', alpha=0.8, 
                               edgecolors='black', linewidth=1, zorder=5)
            self.ax_main.annotate('D', order.delivery_location, 
                                ha='center', va='center', fontweight='bold', 
                                fontsize=8, color='white', zorder=6)
            
            # Connection line
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.6, linestyle='--', linewidth=2)
        
        # Plot recent completed orders (faded)
        for order in self.optimizer.completed_orders[-10:]:
            priority_color = self.priority_colors[order.priority]
            
            self.ax_main.scatter(order.pickup_location[0], order.pickup_location[1], 
                               c='#2E8B57', s=50, marker='o', alpha=0.3, zorder=1)
            self.ax_main.scatter(order.delivery_location[0], order.delivery_location[1], 
                               c='#DC143C', s=50, marker='s', alpha=0.3, zorder=1)
            self.ax_main.plot([order.pickup_location[0], order.delivery_location[0]],
                            [order.pickup_location[1], order.delivery_location[1]],
                            color=priority_color, alpha=0.2, linewidth=1)
        
        # Update metrics plots
        self.time_data.append(self.optimizer.current_time)
        self.score_data.append(self.optimizer.metrics['score_ratio'])
        self.completion_data.append(self.optimizer.metrics['total_orders_completed'])
        self.utilization_data.append(self.optimizer.metrics['drone_utilization'])
        self.cross_depot_data.append(self.optimizer.metrics['cross_depot_operations'])  # ADDED
        
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
        
        # Utilization and Cross-Depot plot
        self.ax_utilization.clear()
        self.ax_utilization.plot(self.time_data, [u * 100 for u in self.utilization_data], 'm-', linewidth=2, label='Utilization %')
        if max(self.cross_depot_data) > 0:
            self.ax_utilization.plot(self.time_data, self.cross_depot_data, 'orange', linestyle='--', linewidth=2, label='Cross-Depot Ops')
        self.ax_utilization.set_ylabel('Percentage / Count')
        self.ax_utilization.set_xlabel('Time (minutes)')
        self.ax_utilization.set_title('Utilization & Cross-Depot Operations')
        self.ax_utilization.grid(True, alpha=0.3)
        self.ax_utilization.legend()

def run_improved_simulation(instance_file: str, dt: float = 1.0, save_metrics: bool = True, show_animation: bool = True):
    """Run simulation with improved cross-depot optimizer"""
    
    print(f"Loading instance from {instance_file}...")
    
    # Load instance
    with open(instance_file, 'rb') as f:
        instance = pickle.load(f)
    
    print(f"Loaded {instance['region']} {instance['size']} instance")
    print(f"Depots: {len(instance['depots'])}, Orders: {len(instance['orders'])}")
    print(f"Total possible score: {instance['total_possible_score']}")
    print(f"Time horizon: {instance['time_horizon']:.1f} minutes")
    
    # Initialize optimizer
    optimizer = ImprovedDroneRoutingOptimizer(instance)
    
    # Initialize visualizer
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
        
        # Print progress
        if frame % 50 == 0:
            metrics = optimizer.metrics
            print(f"Time: {optimizer.current_time:.1f}, "
                  f"Score: {metrics['total_score']}/{optimizer.total_possible_score} "
                  f"({metrics['score_ratio']*100:.1f}%), "
                  f"Completed: {metrics['total_orders_completed']}, "
                  f"Failed: {metrics['total_orders_failed']}, "
                  f"Pending: {len(optimizer.pending_orders)}, "
                  f"Cross-Depot: {metrics['cross_depot_operations']}")
    
    # Run simulation
    frames = int(instance['time_horizon'] / dt)
    
    if show_animation:
        ani = animation.FuncAnimation(visualizer.fig, animate, 
                                     frames=frames, 
                                     interval=int(dt*100),
                                     repeat=False, blit=False)
        plt.tight_layout()

        # Create output_videos folder if it doesn't exist
        # os.makedirs("output_videos", exist_ok=True)

        # video_path = f"output_videos/greedy.mp4"

        # # Save animation
        # print(f"Saving animation to {video_path}...")
        # ani.save(video_path, writer='ffmpeg', fps=10)
        # print("Animation saved.")


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
        metrics_file = f"greedy_approach_results/metrics_{instance_name}_{timestamp}.csv"
        optimizer.save_metrics_to_csv(metrics_file)
        
        # Save GNN training data
        gnn_file = f"gnn_training_data/{instance_name}_sol.pkl"
        optimizer.save_gnn_training_data(gnn_file)
    
    return optimizer

def run_batch_experiments():
    """Run experiments on all instances in the instances folder"""
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('gnn_training_data', exist_ok=True)
    
    # Find all instance files
    instance_files = glob.glob('instances/*.pkl')
    
    if not instance_files:
        print("No instance files found in 'instances' folder!")
        return
    
    print(f"Found {len(instance_files)} instances to process")
    
    # Summary results
    summary_results = []
    
    for instance_file in instance_files:
        print(f"\n{'='*80}")
        print(f"Processing: {os.path.basename(instance_file)}")
        print(f"{'='*80}")
        
        try:
            # Run simulation without animation
            optimizer = run_improved_simulation(instance_file, dt=0.5, 
                                               save_metrics=False, show_animation=False)
            
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
                'cross_depot_assignments': optimizer.cross_depot_assignments,
                'emergency_completed': optimizer.metrics['emergency_completed'],
                'high_completed': optimizer.metrics['high_completed'],
                'medium_completed': optimizer.metrics['medium_completed'],
                'low_completed': optimizer.metrics['low_completed'],
                'emergency_completion_rate': optimizer.metrics['emergency_completion_rate'],
                'high_completion_rate': optimizer.metrics['high_completion_rate'],
                'medium_completion_rate': optimizer.metrics['medium_completion_rate'],
                'low_completion_rate': optimizer.metrics['low_completion_rate']
            }
            
            summary_results.append(summary_data)
            print(f"✓ Successfully processed {instance_name}")
            
        except Exception as e:
            print(f"✗ Error processing {instance_file}: {e}")
            continue
    
    # Save summary results
    if summary_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"greedy_approach_results/greedy_approach.csv"
        
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = summary_results[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in summary_results:
                writer.writerow(row)
        
        print(f"\n{'='*80}")
        print(f"CROSS-DEPOT ENHANCED BATCH PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed {len(summary_results)}/{len(instance_files)} instances successfully")
        print(f"Summary results saved to: {summary_file}")
        print(f"Individual metrics saved to: greedy_approach_results/")
        print(f"GNN training data saved to: gnn_training_data/")
        
        # Print cross-depot performance summary
        if summary_results:
            avg_score_ratio = np.mean([r['score_ratio'] for r in summary_results])
            avg_completion_rate = np.mean([r['completion_rate'] for r in summary_results])
            total_cross_depot_ops = sum([r['cross_depot_operations'] for r in summary_results])
            total_cross_depot_assignments = sum([r['cross_depot_assignments'] for r in summary_results])
            
            print(f"\nCROSS-DEPOT ENHANCED PERFORMANCE SUMMARY:")
            print(f"  Average Score Ratio: {avg_score_ratio:.3f} ({avg_score_ratio*100:.1f}%)")
            print(f"  Average Completion Rate: {avg_completion_rate:.3f} ({avg_completion_rate*100:.1f}%)")
            print(f"  Total Cross-Depot Assignments: {total_cross_depot_assignments}")
            print(f"  Total Cross-Depot Operations Completed: {total_cross_depot_ops}")
            
            if total_cross_depot_assignments > 0:
                cross_depot_success_rate = (total_cross_depot_ops / total_cross_depot_assignments) * 100
                print(f"  Cross-Depot Success Rate: {cross_depot_success_rate:.1f}%")

def main():
    """Main function to run improved cross-depot greedy approach"""
    print("Cross-Depot Enhanced Drone Routing System with Real-World Constraints")
    print("=" * 80)
    
    # Check for sample instance
    sample_instance = "instances/arkansas_medium_0.pkl"
    
    if os.path.exists(sample_instance):
        print(f"Found sample instance: {sample_instance}")
        print("Options:")
        print("1. Run sample instance with visualization")
        print("2. Run all instances in batch mode (no visualization)")
        
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            print(f"Running cross-depot enhanced simulation with {sample_instance}")
            optimizer = run_improved_simulation(sample_instance, dt=1, show_animation=True)
        elif choice == "2":
            print("Running batch experiments on all instances...")
            run_batch_experiments()
        else:
            print("Invalid choice. Running sample instance with visualization...")
            optimizer = run_improved_simulation(sample_instance, dt=0.5, show_animation=True)
    
    else:
        print(f"Sample instance file {sample_instance} not found.")
        print("Checking for instances folder...")
        
        if os.path.exists('instances') and glob.glob('instances/*.pkl'):
            print("Found instances folder with .pkl files")
            print("Running batch experiments...")
            run_batch_experiments()
        else:
            print("No instances found.")
            print("Please run the instance generator first to create instances.")
            print("Expected files:")
            print("  - sample_small_instance.pkl")
            print("  - sample_medium_instance.pkl") 
            print("  - sample_large_instance.pkl")
            print("  - instances/ folder with .pkl files")

if __name__ == "__main__":
    main()