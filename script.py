

import simpy
import numpy as np
import pandas as pd
from scipy.stats import lognorm
import random
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Simulation Parameters
SIM_TIME = 24 * 60  # Total simulation time in minutes
CLOSING_TIMES = [20 * 60, 22 * 60, 24 * 60]  # 8 PM, 10 PM, 12 AM in minutes
HOLDING_BAY_COUNTS = [11, 15, 18, 22]  # Different numbers of holding bays to test

# Arrival Rates (per minute)
ARRIVAL_RATES = {
    'CATH': [
        (6 * 60, 10 * 60, 10 / 60),  # 6 AM to 10 AM
        (10 * 60, 14 * 60, 8 / 60),  # 10 AM to 2 PM
        (14 * 60, 18 * 60, 5 / 60),  # 2 PM to 6 PM
    ],
    'EP': [
        (6 * 60, 10 * 60, 15 / 60),  # 6 AM to 10 AM
        (10 * 60, 14 * 60, 10 / 60),  # 10 AM to 2 PM
        (14 * 60, 18 * 60, 8 / 60),  # 2 PM to 6 PM
    ],
}

# 1. Load and model input data
data = pd.read_csv('Data set for Lab and recovery durations.csv')

# Fit distributions to CATH Lab time
cath_lab_times = data['CATH Lab time (in minutes)'].dropna()
cath_shape, cath_loc, cath_scale = lognorm.fit(cath_lab_times, floc=0)

# Fit distributions to EP Lab time
ep_lab_times = data['EP Lab time (in minutes)'].dropna()
ep_shape, ep_loc, ep_scale = lognorm.fit(ep_lab_times, floc=0)

# Fit distribution to Recovery time
recovery_times = data['Recovery time (in minutes)'].dropna()
recovery_shape, recovery_loc, recovery_scale = lognorm.fit(recovery_times, floc=0)

def generate_procedure_time(patient_type):
    """Generate a procedure time based on the patient type and fitted distributions."""
    if patient_type == 'CATH':
        return lognorm(cath_shape, loc=cath_loc, scale=cath_scale).rvs()
    elif patient_type == 'EP':
        return lognorm(ep_shape, loc=ep_loc, scale=ep_scale).rvs()
    else:
        raise ValueError("Unknown patient type")

def generate_recovery_time():
    """Generate a recovery time based on the fitted distribution."""
    return lognorm(recovery_shape, loc=recovery_loc, scale=recovery_scale).rvs()

# Optional: Plot the fitted distributions for verification
def plot_fitted_distributions():
    # Plot CATH Lab times
    sns.histplot(cath_lab_times, bins=30, kde=False, stat='density', label='Data')
    x = np.linspace(min(cath_lab_times), max(cath_lab_times), 100)
    pdf = lognorm.pdf(x, cath_shape, loc=cath_loc, scale=cath_scale)
    plt.plot(x, pdf, label='Fitted Lognormal')
    plt.title('CATH Lab Times Distribution')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Plot EP Lab times
    sns.histplot(ep_lab_times, bins=30, kde=False, stat='density', label='Data', color='orange')
    x = np.linspace(min(ep_lab_times), max(ep_lab_times), 100)
    pdf = lognorm.pdf(x, ep_shape, loc=ep_loc, scale=ep_scale)
    plt.plot(x, pdf, label='Fitted Lognormal', color='red')
    plt.title('EP Lab Times Distribution')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Plot Recovery times
    sns.histplot(recovery_times, bins=30, kde=False, stat='density', label='Data', color='green')
    x = np.linspace(min(recovery_times), max(recovery_times), 100)
    pdf = lognorm.pdf(x, recovery_shape, loc=recovery_loc, scale=recovery_scale)
    plt.plot(x, pdf, label='Fitted Lognormal', color='black')
    plt.title('Recovery Times Distribution')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# Uncomment the following line to plot the distributions
# plot_fitted_distributions()

# 2. Adjust the patient process
def patient(env, name, patient_type, hospital):
    """Simulates a patient moving through the hospital system."""
    arrival_time = env.now
    hospital.stats['arrivals'] += 1

    # Check if holding bays are open
    if not hospital.holding_bays_open:
        # Holding bays are closed; patient cannot be admitted
        hospital.stats['cancelled'] += 1  # Procedure is cancelled
        return  # Patient leaves the system

    # Request a holding bay for preparation
    with hospital.holding_bays.request(priority=1) as request:
        result = yield request | env.timeout(60)  # Wait up to 60 minutes

        if request in result:
            # Holding bay allocated
            preparation_time = np.random.exponential(20)
            yield env.timeout(preparation_time)

            # Wait until lab unit is available
            with hospital.lab_units.request() as lab_request:
                yield lab_request
                # Release holding bay during procedure
                hospital.holding_bays.release(request)

                # Procedure time based on patient type
                procedure_duration = generate_procedure_time(patient_type)
                yield env.timeout(procedure_duration)

                # Check if holding bays are open for recovery
                if hospital.holding_bays_open:
                    # Request a holding bay for recovery
                    with hospital.holding_bays.request(priority=0) as recovery_request:
                        yield recovery_request
                        # Release lab unit
                        hospital.lab_units.release(lab_request)
                        recovery_duration = generate_recovery_time()
                        yield env.timeout(recovery_duration)
                        # Release holding bay after recovery
                        hospital.holding_bays.release(recovery_request)
                        hospital.stats['processed'] += 1
                else:
                    # Holding bays are closed; transfer patient to hospital for recovery
                    # Release lab unit
                    hospital.lab_units.release(lab_request)
                    # Record that patient was transferred
                    hospital.stats['transferred'] += 1
                    hospital.stats['processed'] += 1  # Considered as processed
        else:
            # Patient leaves due to waiting too long
            hospital.stats['reneged'] += 1

# 3. Define the Hospital Environment
class Hospital:
    """Represents the hospital environment."""
    def __init__(self, env, num_holding_bays, closing_time):
        self.env = env
        self.holding_bays = simpy.PriorityResource(env, capacity=num_holding_bays)
        self.lab_units = simpy.Resource(env, capacity=12)
        self.closing_time = closing_time
        self.holding_bays_open = True  # Holding bays are initially open
        self.stats = {
            'arrivals': 0,
            'processed': 0,
            'reneged': 0,
            'cancelled': 0,
            'transferred': 0,  # To track patients transferred to hospital
            'holding_bay_idle_time': 0,
            'lab_unit_idle_time': 0,
        }
        self.last_event_time = 0
        env.process(self.close_holding_bays())

    def close_holding_bays(self):
        """Closes the holding bays at the specified closing time."""
        yield self.env.timeout(self.closing_time)
        self.holding_bays_open = False  # Holding bays are now closed

# 4. Define the Arrival Process
def arrival_process(env, patient_type, hospital):
    """Generates patients arriving at the hospital."""
    while True:
        current_time = env.now
        arrival_rate = get_arrival_rate(patient_type, current_time)
        if arrival_rate == 0 or current_time >= hospital.closing_time:
            # No arrivals during this time or after closing time
            yield env.timeout(1)
            continue
        interarrival_time = random.expovariate(arrival_rate)
        yield env.timeout(interarrival_time)
        # Check again if the holding bays are open
        if env.now >= hospital.closing_time:
            break  # Stop generating patients
        patient_id = f"{patient_type}_{hospital.stats['arrivals'] + 1}"
        env.process(patient(env, patient_id, patient_type, hospital))

# 5. Get Arrival Rate Based on Time
def get_arrival_rate(patient_type, current_time):
    """Returns the arrival rate based on the time of day."""
    for start, end, rate in ARRIVAL_RATES[patient_type]:
        if start <= current_time < end:
            return rate
    return 0

# 6. Run the Simulation
def run_simulation(num_holding_bays, closing_time):
    """Runs the simulation and returns collected statistics."""
    env = simpy.Environment()
    hospital = Hospital(env, num_holding_bays, closing_time)
    env.process(arrival_process(env, 'CATH', hospital))
    env.process(arrival_process(env, 'EP', hospital))
    env.run(until=SIM_TIME)
    return hospital.stats

# 7. Main Loop for Experimentation
# Collect results for different scenarios
results = []
for num_bays in HOLDING_BAY_COUNTS:
    for closing in CLOSING_TIMES:
        stats = run_simulation(num_bays, closing)
        results.append({
            'Holding Bays': num_bays,
            'Closing Time': int(closing / 60),
            'Total Arrivals': stats['arrivals'],
            'Processed': stats['processed'],
            'Reneged': stats['reneged'],
            'Cancelled': stats['cancelled'],
            'Transferred': stats.get('transferred', 0),
            # Add additional metrics as needed
        })
        print(f"Simulation with {num_bays} holding bays and closing time at {int(closing / 60)}:00")
        print(f"Total Arrivals: {stats['arrivals']}")
        print(f"Processed: {stats['processed']}")
        print(f"Reneged: {stats['reneged']}")
        print(f"Cancelled: {stats['cancelled']}")
        print(f"Transferred: {stats.get('transferred', 0)}")
        print("-" * 50)

# 8. Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# 9. Save results to CSV (optional)
results_df.to_csv('simulation_results.csv', index=False)

# 10. Print the results DataFrame
print(results_df)
