import numpy as np
import pickle
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import os

class OrderPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EMERGENCY = 4

@dataclass
class Order:
    id: int
    pickup_location: Tuple[float, float]
    delivery_location: Tuple[float, float]
    priority: OrderPriority
    arrival_time: float
    deadline: float
    weight: float = 1.0
    score: int = 0  # Score for completing this order

@dataclass
class Depot:
    id: int
    location: Tuple[float, float]

class DroneInstanceGenerator:
    """Generate realistic drone routing instances with feasibility constraints"""
    
    # Real-world drone specifications (Zipline 2 reference)
    DRONE_SPECS = {
        'cruising_speed_mph': 70,  # 70 mph
        'cruising_speed_mps': 70 * 1609.34 / 3600,  # Convert to meters per second
        'battery_duration_minutes': 21,  # 21 minutes flight time
        'charging_time_minutes': 30,  # 30 minutes charging time
        'max_payload_kg': 1.75,  # Typical small package delivery capacity
        'takeoff_landing_time_seconds': 30,  # Time for takeoff/landing operations
        'service_time_seconds': 60,  # Time spent at pickup/delivery location
    }
    
    # Geographic regions with much smaller, drone-feasible bounds
    # These represent small urban/suburban areas suitable for drone delivery
    REGIONS = {
        'arkansas': {
            'bounds': ((-92.35, -92.25), (34.65, 34.75)),  # ~11km x 11km around Little Rock
            'cities': [(-92.3, 34.7)],  # Central Little Rock area
        },
        'north_carolina': {
            'bounds': ((-80.85, -80.75), (35.15, 35.25)),  # ~11km x 11km around Charlotte
            'cities': [(-80.8, 35.2)],  # Central Charlotte area
        },
        'utah': {
            'bounds': ((-111.95, -111.85), (40.75, 40.85)),  # ~11km x 11km around Salt Lake City
            'cities': [(-111.9, 40.8)],  # Central Salt Lake City area
        },
        'texas': {
            'bounds': ((-97.75, -97.65), (30.25, 30.35)),  # ~11km x 11km around Austin
            'cities': [(-97.7, 30.3)],  # Central Austin area
        }
    }
    
    @staticmethod
    def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate distance between two coordinates using Haversine formula (in meters)"""
        lat1, lon1 = math.radians(coord1[1]), math.radians(coord1[0])
        lat2, lon2 = math.radians(coord2[1]), math.radians(coord2[0])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth's radius in meters
        R = 6371000
        return R * c
    
    @staticmethod
    def calculate_flight_time(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate flight time between two coordinates in seconds"""
        distance_m = DroneInstanceGenerator.haversine_distance(coord1, coord2)
        speed_mps = DroneInstanceGenerator.DRONE_SPECS['cruising_speed_mps']
        return distance_m / speed_mps
    
    @staticmethod
    def calculate_total_mission_time(pickup_coord: Tuple[float, float], 
                                   delivery_coord: Tuple[float, float],
                                   depot_coord: Tuple[float, float]) -> float:
        """Calculate total time for pickup-delivery-return mission in seconds"""
        
        # Flight times
        to_pickup_time = DroneInstanceGenerator.calculate_flight_time(depot_coord, pickup_coord)
        pickup_to_delivery_time = DroneInstanceGenerator.calculate_flight_time(pickup_coord, delivery_coord)
        delivery_to_depot_time = DroneInstanceGenerator.calculate_flight_time(delivery_coord, depot_coord)
        
        # Service times
        takeoff_landing_time = DroneInstanceGenerator.DRONE_SPECS['takeoff_landing_time_seconds'] * 3  # 3 operations
        service_time = DroneInstanceGenerator.DRONE_SPECS['service_time_seconds'] * 2  # pickup + delivery
        
        total_time = (to_pickup_time + pickup_to_delivery_time + delivery_to_depot_time + 
                     takeoff_landing_time + service_time)
        
        return total_time
    
    @staticmethod
    def is_mission_feasible(pickup_coord: Tuple[float, float], 
                           delivery_coord: Tuple[float, float],
                           depot_coord: Tuple[float, float]) -> bool:
        """Check if a mission is feasible within battery constraints"""
        total_time = DroneInstanceGenerator.calculate_total_mission_time(
            pickup_coord, delivery_coord, depot_coord)
        
        max_flight_time = DroneInstanceGenerator.DRONE_SPECS['battery_duration_minutes'] * 60  # Convert to seconds
        
        # Add 10% safety margin
        return total_time <= max_flight_time * 0.9
    
    @staticmethod
    def generate_feasible_depot_locations(region: str, num_depots: int, area_scale: float = 1.0) -> List[Depot]:
        """Generate depot locations ensuring good coverage within scaled area"""
        if region not in DroneInstanceGenerator.REGIONS:
            raise ValueError(f"Region {region} not supported")
        
        region_data = DroneInstanceGenerator.REGIONS[region]
        (min_lon, max_lon), (min_lat, max_lat) = region_data['bounds']
        
        # Apply area scaling to reduce the operational area further if needed
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        width = (max_lon - min_lon) * area_scale
        height = (max_lat - min_lat) * area_scale
        
        min_lon = center_lon - width / 2
        max_lon = center_lon + width / 2
        min_lat = center_lat - height / 2
        max_lat = center_lat + height / 2
        
        print(f"Scaled area: {width*111:.1f}km x {height*111:.1f}km")
        
        depots = []
        
        if num_depots == 1:
            # Single depot at center
            depots.append(Depot(id=0, location=(center_lon, center_lat)))
        
        elif num_depots <= 4:
            # Place depots in corners/edges for small numbers
            positions = [
                (min_lon + width*0.2, min_lat + height*0.2),  # Bottom-left
                (max_lon - width*0.2, min_lat + height*0.2),  # Bottom-right  
                (min_lon + width*0.2, max_lat - height*0.2),  # Top-left
                (max_lon - width*0.2, max_lat - height*0.2),  # Top-right
            ]
            
            for i in range(num_depots):
                depot_lon, depot_lat = positions[i]
                # Add small random variation
                depot_lon += random.uniform(-width*0.05, width*0.05)
                depot_lat += random.uniform(-height*0.05, height*0.05)
                
                depots.append(Depot(id=i, location=(depot_lon, depot_lat)))
        
        else:
            # Grid-based distribution for larger numbers
            grid_size = math.ceil(math.sqrt(num_depots))
            
            for i in range(num_depots):
                grid_x = i % grid_size
                grid_y = i // grid_size
                
                # Calculate position with some randomization
                lon_offset = (grid_x + 0.5 + random.uniform(-0.3, 0.3)) * width / grid_size
                lat_offset = (grid_y + 0.5 + random.uniform(-0.3, 0.3)) * height / grid_size
                
                depot_lon = min_lon + lon_offset
                depot_lat = min_lat + lat_offset
                
                # Ensure within bounds
                depot_lon = max(min_lon, min(max_lon, depot_lon))
                depot_lat = max(min_lat, min(max_lat, depot_lat))
                
                depots.append(Depot(id=i, location=(depot_lon, depot_lat)))
        
        return depots
    
    @staticmethod
    def generate_feasible_orders(depots: List[Depot], num_orders: int, 
                               region: str, time_horizon: float, area_scale: float = 1.0) -> List[Order]:
        """Generate orders ensuring all are feasible from at least one depot"""
        if region not in DroneInstanceGenerator.REGIONS:
            raise ValueError(f"Region {region} not supported")
        
        region_data = DroneInstanceGenerator.REGIONS[region]
        (min_lon, max_lon), (min_lat, max_lat) = region_data['bounds']
        
        # Apply area scaling
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        
        width = (max_lon - min_lon) * area_scale
        height = (max_lat - min_lat) * area_scale
        
        min_lon = center_lon - width / 2
        max_lon = center_lon + width / 2
        min_lat = center_lat - height / 2
        max_lat = center_lat + height / 2
        
        orders = []
        max_attempts_per_order = 100  # Increased attempts per order
        
        # Score mapping for priorities
        priority_scores = {
            OrderPriority.LOW: 1,
            OrderPriority.MEDIUM: 3,
            OrderPriority.HIGH: 7,
            OrderPriority.EMERGENCY: 15
        }
        
        # Pre-calculate maximum feasible distance for any depot
        max_feasible_distance = 0
        battery_seconds = DroneInstanceGenerator.DRONE_SPECS['battery_duration_minutes'] * 60
        speed_mps = DroneInstanceGenerator.DRONE_SPECS['cruising_speed_mps']
        
        # Account for service times and safety margin
        service_time = DroneInstanceGenerator.DRONE_SPECS['service_time_seconds'] * 2
        takeoff_landing_time = DroneInstanceGenerator.DRONE_SPECS['takeoff_landing_time_seconds'] * 3
        safety_margin = battery_seconds * 0.1
        
        available_flight_time = battery_seconds - service_time - takeoff_landing_time - safety_margin
        max_feasible_distance = (available_flight_time * speed_mps) / 3  # Divided by 3 for round trip
        
        print(f"Maximum feasible distance per leg: {max_feasible_distance/1000:.2f} km")
        
        successful_orders = 0
        total_attempts = 0
        
        for order_id in range(num_orders):
            attempts = 0
            order_created = False
            
            while attempts < max_attempts_per_order and not order_created:
                total_attempts += 1
                
                # Generate pickup location (bias towards depot locations for higher success rate)
                if random.random() < 0.6:  # 60% chance to be near a depot
                    # Pick a random depot and generate location nearby
                    depot = random.choice(depots)
                    # Generate within 2km of depot
                    max_offset = min(0.02, width/4, height/4)  # 0.02 degrees ≈ 2km
                    pickup_lon = depot.location[0] + random.uniform(-max_offset, max_offset)
                    pickup_lat = depot.location[1] + random.uniform(-max_offset, max_offset)
                else:
                    # Random location in area
                    pickup_lon = random.uniform(min_lon, max_lon)
                    pickup_lat = random.uniform(min_lat, max_lat)
                
                pickup_location = (pickup_lon, pickup_lat)
                
                # Generate delivery location with distance constraint
                max_delivery_distance = max_feasible_distance * 0.7  # Conservative estimate
                
                # Try to place delivery within feasible range
                delivery_attempts = 0
                delivery_location = None
                
                while delivery_attempts < 20:
                    if random.random() < 0.4:  # 40% chance near a depot
                        depot = random.choice(depots)
                        max_offset = min(0.02, width/4, height/4)
                        delivery_lon = depot.location[0] + random.uniform(-max_offset, max_offset)
                        delivery_lat = depot.location[1] + random.uniform(-max_offset, max_offset)
                    else:
                        delivery_lon = random.uniform(min_lon, max_lon)
                        delivery_lat = random.uniform(min_lat, max_lat)
                    
                    potential_delivery = (delivery_lon, delivery_lat)
                    
                    # Check if pickup-delivery distance is reasonable
                    pickup_delivery_distance = DroneInstanceGenerator.haversine_distance(
                        pickup_location, potential_delivery)
                    
                    if pickup_delivery_distance <= max_delivery_distance:
                        delivery_location = potential_delivery
                        break
                    
                    delivery_attempts += 1
                
                if delivery_location is None:
                    attempts += 1
                    continue
                
                # Check if this order is feasible from at least one depot
                feasible = False
                min_mission_time = float('inf')
                
                for depot in depots:
                    if DroneInstanceGenerator.is_mission_feasible(
                        pickup_location, delivery_location, depot.location):
                        feasible = True
                        mission_time = DroneInstanceGenerator.calculate_total_mission_time(
                            pickup_location, delivery_location, depot.location)
                        min_mission_time = min(min_mission_time, mission_time)
                        break
                
                if feasible:
                    # Generate order properties
                    priority_weights = [0.3, 0.5, 0.15, 0.05]  # Favor easier orders
                    priority = random.choices(list(OrderPriority), weights=priority_weights)[0]
                    
                    # Arrival time uniformly distributed
                    arrival_time = random.uniform(0, time_horizon * 0.7)  # Orders arrive in first 70%
                    
                    # Deadline based on mission time and priority with generous multipliers
                    base_time_allowance = min_mission_time / 60  # Convert to minutes
                    priority_multipliers = {
                        OrderPriority.EMERGENCY: 1.5,  # More generous
                        OrderPriority.HIGH: 2.0,
                        OrderPriority.MEDIUM: 3.0,
                        OrderPriority.LOW: 5.0
                    }
                    
                    time_allowance = base_time_allowance * priority_multipliers[priority]
                    deadline = arrival_time + time_allowance
                    
                    # Ensure deadline is within time horizon
                    deadline = min(deadline, time_horizon)
                    
                    order = Order(
                        id=order_id,
                        pickup_location=pickup_location,
                        delivery_location=delivery_location,
                        priority=priority,
                        arrival_time=arrival_time,
                        deadline=deadline,
                        weight=random.uniform(0.1, 1.5),  # Weight in kg
                        score=priority_scores[priority]
                    )
                    
                    orders.append(order)
                    successful_orders += 1
                    order_created = True
                    
                    if successful_orders % 10 == 0:
                        print(f"Generated {successful_orders}/{num_orders} orders...")
                
                attempts += 1
            
            if not order_created:
                print(f"Warning: Could not generate feasible order {order_id} after {max_attempts_per_order} attempts")
        
        print(f"Successfully generated {successful_orders}/{num_orders} orders in {total_attempts} total attempts")
        
        # Sort orders by arrival time
        orders.sort(key=lambda x: x.arrival_time)
        return orders
    
    @staticmethod
    def generate_instance(region: str, size: str, instance_id: int = 0) -> Dict:
        """Generate a complete problem instance"""
        
        # Instance parameters with area scaling
        size_params = {
            'small': {'depots': 5, 'orders': 100, 'time_horizon': 480.0, 'area_scale': 0.3},
            'medium': {'depots': 10, 'orders': 200, 'time_horizon': 600.0, 'area_scale': 0.6},
            'large': {'depots': 15, 'orders': 400, 'time_horizon': 720.0, 'area_scale': 1.0}
        }
        
        if size not in size_params:
            raise ValueError(f"Size {size} not supported. Use 'small', 'medium', or 'large'")
        
        params = size_params[size]
        
        print(f"Generating {size} instance for {region} with {params['depots']} depots and {params['orders']} orders...")
        print(f"Area scale: {params['area_scale']} (smaller scale = smaller area)")
        
        # Generate depots with area scaling
        depots = DroneInstanceGenerator.generate_feasible_depot_locations(
            region, params['depots'], params['area_scale'])
        
        # Generate orders with area scaling
        orders = DroneInstanceGenerator.generate_feasible_orders(
            depots, params['orders'], region, params['time_horizon'], params['area_scale'])
        
        # Calculate total possible score
        total_possible_score = sum(order.score for order in orders)
        
        # Calculate area statistics
        region_data = DroneInstanceGenerator.REGIONS[region]
        (min_lon, max_lon), (min_lat, max_lat) = region_data['bounds']
        
        # Apply scaling to get actual operational area
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
        width = (max_lon - min_lon) * params['area_scale']
        height = (max_lat - min_lat) * params['area_scale']
        
        actual_bounds = (
            (center_lon - width/2, center_lon + width/2),
            (center_lat - height/2, center_lat + height/2)
        )
        
        instance = {
            'instance_id': instance_id,
            'region': region,
            'size': size,
            'depots': depots,
            'orders': orders,
            'bounds': actual_bounds,  # Use scaled bounds
            'original_bounds': region_data['bounds'],  # Keep original for reference
            'time_horizon': params['time_horizon'],
            'area_scale': params['area_scale'],
            'drone_specs': DroneInstanceGenerator.DRONE_SPECS,
            'total_possible_score': total_possible_score,
            'generation_stats': {
                'num_depots': len(depots),
                'num_orders': len(orders),
                'feasible_orders': len(orders),
                'operational_area_km2': (width * 111) * (height * 111),  # Approximate km²
                'avg_order_distance': np.mean([
                    DroneInstanceGenerator.haversine_distance(order.pickup_location, order.delivery_location) 
                    for order in orders
                ]) / 1000 if orders else 0,  # Convert to km
                'max_order_distance': max([
                    DroneInstanceGenerator.haversine_distance(order.pickup_location, order.delivery_location) 
                    for order in orders
                ]) / 1000 if orders else 0,  # Convert to km
            }
        }
        
        print(f"Generated instance with {len(orders)} feasible orders")
        print(f"Operational area: {instance['generation_stats']['operational_area_km2']:.1f} km²")
        print(f"Total possible score: {total_possible_score}")
        print(f"Average order distance: {instance['generation_stats']['avg_order_distance']:.2f} km")
        print(f"Maximum order distance: {instance['generation_stats']['max_order_distance']:.2f} km")
        
        return instance
    
    @staticmethod
    def save_instance(instance: Dict, filename: str):
        """Save instance to pickle file"""
        # Only create directory if filename contains a directory path
        if os.path.dirname(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'wb') as f:
            pickle.dump(instance, f)
        
        print(f"Instance saved to {filename}")
    
    @staticmethod
    def load_instance(filename: str) -> Dict:
        """Load instance from pickle file"""
        with open(filename, 'rb') as f:
            instance = pickle.load(f)
        
        print(f"Instance loaded from {filename}")
        print(f"Region: {instance['region']}, Size: {instance['size']}")
        print(f"Depots: {len(instance['depots'])}, Orders: {len(instance['orders'])}")
        print(f"Total possible score: {instance['total_possible_score']}")
        
        return instance
    
    @staticmethod
    def generate_all_instances():
        """Generate all required instances"""
        regions = ['arkansas', 'north_carolina', 'utah', 'texas']
        sizes = ['small', 'medium', 'large']
        instances_per_config = 5
        
        os.makedirs('instances', exist_ok=True)
        
        for region in regions:
            for size in sizes:
                for instance_num in range(instances_per_config):
                    instance = DroneInstanceGenerator.generate_instance(region, size, instance_num)
                    filename = f"instances/{region}_{size}_{instance_num}.pkl"
                    DroneInstanceGenerator.save_instance(instance, filename)
                    
                    print(f"Generated {region} {size} instance {instance_num}")
                    print("-" * 50)

def main():
    """Generate sample instances for testing"""
    
    print("Drone Instance Generator - Scaled for Feasible Operations")
    print("=" * 60)
    print(f"Drone Specifications:")
    print(f"  Cruising Speed: {DroneInstanceGenerator.DRONE_SPECS['cruising_speed_mph']} mph")
    print(f"  Battery Duration: {DroneInstanceGenerator.DRONE_SPECS['battery_duration_minutes']} minutes")
    print(f"  Charging Time: {DroneInstanceGenerator.DRONE_SPECS['charging_time_minutes']} minutes")
    print(f"  Max Payload: {DroneInstanceGenerator.DRONE_SPECS['max_payload_kg']} kg")
    print()
    
    print("Area Scaling:")
    print("  Small instances: 30% of base area (~3.3km x 3.3km)")
    print("  Medium instances: 60% of base area (~6.6km x 6.6km)")  
    print("  Large instances: 100% of base area (~11km x 11km)")
    print()
    
    # Calculate theoretical maximum range
    battery_seconds = DroneInstanceGenerator.DRONE_SPECS['battery_duration_minutes'] * 60
    speed_mps = DroneInstanceGenerator.DRONE_SPECS['cruising_speed_mps']
    service_time = DroneInstanceGenerator.DRONE_SPECS['service_time_seconds'] * 2
    takeoff_landing_time = DroneInstanceGenerator.DRONE_SPECS['takeoff_landing_time_seconds'] * 3
    
    available_flight_time = battery_seconds - service_time - takeoff_landing_time - (battery_seconds * 0.1)
    max_range_km = (available_flight_time * speed_mps) / 1000 / 3  # Round trip + safety
    
    print(f"Theoretical maximum delivery range: {max_range_km:.2f} km")
    print()
    
    # Generate a sample instance for testing
    print("Generating sample instances...")
    
    # Generate one instance of each size for testing
    for size in ['small', 'medium', 'large']:
        try:
            print(f"\nGenerating {size} instance...")
            instance = DroneInstanceGenerator.generate_instance('texas', size, 0)
            filename = f"sample_{size}_instance.pkl"
            DroneInstanceGenerator.save_instance(instance, filename)
            
            # Test loading
            loaded_instance = DroneInstanceGenerator.load_instance(filename)
            print(f"✓ Successfully generated and verified {size} instance")
            
        except Exception as e:
            print(f"✗ Failed to generate {size} instance: {e}")
    
    print(f"\nTo generate all instances for all regions, uncomment the line below:")
    print("# DroneInstanceGenerator.generate_all_instances()")

if __name__ == "__main__":
    # main()
    
    # print(f"  Charging Time: {DroneInstanceGenerator.DRONE_SPECS['charging_time_minutes']} minutes")
    # print(f"  Max Payload: {DroneInstanceGenerator.DRONE_SPECS['max_payload_kg']} kg")
    # print()
    
    # # Generate a sample instance for testing
    # print("Generating sample instances...")
    
    # # Generate one instance of each size for testing
    # for size in ['small', 'medium', 'large']:
    #     instance = DroneInstanceGenerator.generate_instance('texas', size, 0)
    #     filename = f"sample_{size}_instance.pkl"
    #     DroneInstanceGenerator.save_instance(instance, filename)
        
    #     # Test loading
    #     loaded_instance = DroneInstanceGenerator.load_instance(filename)
    #     print(f"Successfully generated and loaded {size} instance")
    #     print()
    
    # print("To generate all instances, uncomment the line below:")
    # print("# DroneInstanceGenerator.generate_all_instances()")

    # Uncomment to generate all instances (this will take some time)
    DroneInstanceGenerator.generate_all_instances()